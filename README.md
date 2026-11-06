# Sector analysis for differential dynamic microscopy

Differential dynamic microscopy (DDM) measures how particles move from a microscope video, without tracking them. Most
DDM models average the image structure function (ISF) over all directions, which throws away the direction of motion.
Existing sector methods keep some direction, but treat the model as constant across each sector, which fails for
directed motion.

This package fits **angular sector models** that keep the phase variation across a sector. From a video of particles
drifting in two dimensions, it gives the drift speed, diffusion coefficient, speed spread and drift direction. It also
includes radial 2D (Bessel) and 3D (sinc) models for comparison. It was written for colloidal beads drifting under
gravity, and is meant for any 2D directed motion, such as bacteria at an interface.

The theory and tests are in the accompanying report, *Sector Analysis for Colloidal Motion in Two Dimensions using
Differential Dynamic Microscopy*. Model names here are the names used in the report.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```python
from sector_ddm import load_video, compute_isf, characteristic_time, fit_models

frames = load_video("my_video.avi")
isf = compute_isf(frames, fps=50, pixel_size=0.119, sector_half_width=20, drift_angle_deg=0)

fits = fit_models(isf, ["exp*J0", "sectors_spread"], q_range=(1.0, 4.0))

f = fits["sectors_spread"]
f.v              # (speed, standard error) in µm/s
f.D              # (diffusion coefficient, standard error) in µm²/s
f.speed_spread   # std/mean of the speeds
f.bic            # lower is better
```

## Functions

### Computing the ISF (`sector_ddm/isf.py`)

**`load_video(path)`** reads a video into an `(n, height, width)` grayscale array.

**`compute_isf(frames, fps, pixel_size, ...)`** returns an `ISF`, with the radial ISF and the two sector ISFs.

| Argument | Default | Meaning |
|---|---|---|
| `fps` | required | Camera frame rate |
| `pixel_size` | required | µm per pixel |
| `sector_half_width` | 20 | Degrees either side of each sector's centre line. `None` computes the radial ISF only |
| `drift_angle_deg` | 0 | Expected drift direction, from image +x towards +y (clockwise on screen). The `parallel` sector is centred here and the `perp` sector 90° from it |
| `max_couples` | 300 | Most frame pairs averaged per lag |
| `points_per_decade` | 30 | Density of the log-spaced lags |
| `frame_index` | `None` | For videos with dropped frames: the camera-frame number of each frame. Only pairs an exact number of camera frames apart are used |

**`ISF`** holds `radial`, `parallel`, `perp`, `lags` (s) and `q` (µm⁻¹). `isf.save(path)` and `ISF.load(path)` store it,
so the slow step only runs once. To use an ISF computed elsewhere, build one directly with `ISF(radial=..., lags=..., q=...)`.

### Choosing the q range (`sector_ddm/fitting.py`)

**`characteristic_time(isf, model="sinc")`** fits a simple model at every q and returns an array of `A, B, tau_c`. Use
`model="sinc"` for drifting samples and `model="exp"` for purely diffusing ones. Plot `tau_c` against `isf.q` on log axes
and choose `q_range` where it is a clean straight line. The range matters: outside it the fits describe noise.

### Fitting (`sector_ddm/fitting.py`)

**`fit_models(isf, names, q_range)`** fits each named model at every q in `q_range = (q_min, q_max)` and returns
`{name: Fit}`. All the models named in one call use the same q, so compare models within a single call.

**`fit(isf, name, q_range)`** does the same for one model and returns its `Fit`.

**`Fit`** holds the results:

| Attribute | Meaning |
|---|---|
| `v` | Mean speed and standard error, from Ω = vq, over the q where the drift is resolved (Ω τ_max ≥ 1). NaN without drift |
| `D` | Diffusion coefficient and standard error, from Γ = Dq² |
| `speed_spread` | Median over q of std/mean of the speeds, for models with σ |
| `angle` | Median and interquartile range over q of the drift angle in degrees, for `sectors_angle` |
| `bic`, `bic_per_q` | Bayesian information criterion, mean over q and per q. Compare radial models with radial models, and sector models with sector models |
| `v_perp`, `D_parallel` | For `sectors_separate`: the values from the other sector |
| `q`, `per_q` | The q fitted and every fitted parameter at each q |
| `curve(i, where)` | The fitted ISF at `q[i]`, to plot over the data. `where` is `"radial"`, `"parallel"` or `"perp"` |

## Models

Each is fitted as ISF(q, τ) = A(q)[1 − f(q, τ)] + B(q). `MODELS` in `sector_ddm/models.py` lists them.

| Name | Data | Motion it describes | Fitted at each q, besides A, B |
|---|---|---|---|
| `exp` | radial | Diffusion only | Γ |
| `sinc` | radial | Drift in random 3D directions | Ω |
| `J0` | radial | Drift at one speed in the plane | Ω |
| `exp*sinc` | radial | `sinc` + diffusion | Γ, Ω |
| `exp*J0` | radial | `J0` + diffusion | Γ, Ω |
| `exp*J0*u_spread` | radial | `exp*J0` + speed spread | Γ, Ω, σ |
| `sectors_single` | both sectors | One speed along the drift direction + diffusion | Γ, Ω |
| `sectors_spread` | both sectors | `sectors_single` + speed spread | Γ, Ω, σ |
| `sectors_angle` | both sectors | `sectors_spread` + free drift angle θ | Γ, Ω, σ, θ |
| `sectors_separate` | each sector alone | The spread model on each sector, with its own Γ, Ω, σ | Γ, Ω, σ per sector |

The sector models also fit a separate A and B per sector. Speeds are lognormal with mean 1 and log-width σ.

**Which model to use.** Start with `sectors_spread` if you know the drift direction, and set `drift_angle_deg` to it.
Use `sectors_angle` if you need to measure the direction. Use `exp*J0*u_spread` if you need only the speed and D, since
a radial fit needs no direction.

## Limitations

- The ISF is even in q, so the sign of the drift is lost: θ and θ + 180° look the same, and `sectors_angle` reports 0–90°.
- `sectors_angle` is least accurate when the drift is within about 10° of a sector's centre line, where it can stay at 0°.
- D is overestimated if the sample also moves through other means, such as flow or vibration.
- A very low σ means the data show no measurable spread, not that the spread is exactly zero.

## Code layout

```
sector_ddm/isf.py       frames to ISF
sector_ddm/models.py    the f(q, τ) of each model
sector_ddm/fitting.py   fits, and v, D, spread, angle and BIC from them
```

## Credits

- **Method:** differential dynamic microscopy was introduced by R. Cerbino and V. Trappe, "Differential dynamic
  microscopy: Probing wave vector dependent dynamics with a microscope", Phys. Rev. Lett. **100**, 188102 (2008).
- **Ideas and starting point:** [samuelmcdermott/DDM](https://github.com/samuelmcdermott/DDM). The ISF code in
  `sector_ddm/isf.py` derives from that repository.
- **Code generation:** the code in this repository was written with help from Anthropic's claude code.

## Licence

GPL-2.0, see `LICENSE`. This is the licence of [samuelmcdermott/DDM](https://github.com/samuelmcdermott/DDM), which this
code derives from.
