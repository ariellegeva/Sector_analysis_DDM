"""Fit the models to an ISF, q by q, and turn the fits into v, D, speed spread, drift angle and BIC.

Each fit is in log space (residual log ISF - log model) from several starting points, keeping the best. Starts come
from a first sinc fit at each q, and from the simpler models a model contains (Model.needs).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import leastsq, least_squares

from .models import ANGLE_NODES, MODELS, RADIAL_F, sector_f, speed_cv

SIGMA_MAX = 2.0               # upper bound on the lognormal log-width (speed std/mean up to ~7)
OMEGA_STARTS = (0.5, 0.77, 1, 1.3, 2)   # × 1/tau_c of the first fit
GAMMA_STARTS = (0.01, 0.1, 1)           # × Omega start
SIGMA_STARTS = (0.05, 0.3, 0.8)
THETA_STARTS = (5.0, 20.0, 45.0, 70.0, 85.0)   # degrees

_COLS = ("A", "B", "A_perp", "B_perp", "Gamma", "Omega", "sigma", "theta", "Gamma_parallel", "Omega_perp", "sigma_perp",
         "mse")
_C = {c: i for i, c in enumerate(_COLS)}


# ---------------------------------------------------------------- results

def power_fit(q, y, power):
    """y = C q^power in log space, points with y <= 0 left out. Returns C and its standard error."""
    ok = y > 0
    r = np.log(y[ok]) - power * np.log(q[ok])
    C = np.exp(np.mean(r))
    return C, C * np.std(r, ddof=1) / np.sqrt(len(r)) if len(r) >= 2 else np.nan


@dataclass
class Fit:
    """One model fitted over a q range. per_q holds an array per parameter over the q in .q (see .per_q keys):
    parameters the model does not have are 0 (Gamma, Omega, sigma, theta) or NaN (the rest)."""
    model: str
    q: np.ndarray = field(repr=False)
    per_q: dict = field(repr=False)
    tau_max: float = field(repr=False)
    n_lags: int = field(repr=False)
    _isf: object = field(default=None, repr=False)

    def _power(self, column, power, q_mask=None):
        q, y = self.q, self.per_q[column]
        if q_mask is not None:
            q, y = q[q_mask], y[q_mask]
        return power_fit(q, y, power) if len(q) >= 2 else (np.nan, np.nan)

    @property
    def _resolved(self):
        """q with Omega tau_max >= 1: the drift phase is resolved within the video."""
        return self.per_q["Omega"] * self.tau_max >= 1

    @property
    def v(self):
        """Mean drift speed [µm/s] and its standard error, over the q where the drift is resolved. NaN without drift."""
        if "Omega" not in MODELS[self.model].has:
            return np.nan, np.nan
        return self._power("Omega", 1, self._resolved)

    @property
    def D(self):
        """Diffusion coefficient [µm²/s] and its standard error. sectors_separate: from the perp sector."""
        return self._power("Gamma", 2) if "Gamma" in MODELS[self.model].has else (np.nan, np.nan)

    @property
    def v_perp(self):
        """sectors_separate: the speed from the perp sector, over all q."""
        return self._power("Omega_perp", 1)

    @property
    def D_parallel(self):
        """sectors_separate: the diffusion coefficient from the parallel sector."""
        return self._power("Gamma_parallel", 2)

    @property
    def speed_spread(self):
        """Speed std/mean, median over q."""
        return float(np.median(speed_cv(self.per_q["sigma"])))

    @property
    def angle(self):
        """Drift angle from the expected direction [degrees]: median and interquartile range over q. Always in 0-90°,
        since the ISF cannot tell +theta from -theta."""
        deg = np.degrees(self.per_q["theta"])
        return float(np.median(deg)), float(np.subtract(*np.percentile(deg, [75, 25])))

    @property
    def bic_per_q(self):
        """n ln(mse) + k ln(n) at each q, for independent Gaussian residuals; lower is better. Compare models only on
        the same data: radial models with radial models, sector models with sector models."""
        m = MODELS[self.model]
        n = self.n_lags * (2 if m.data == "sector" else 1)
        return n * np.log(self.per_q["mse"]) + m.k * np.log(n)

    @property
    def bic(self):
        return float(np.mean(self.bic_per_q))

    def curve(self, q_index, where="radial"):
        """The fitted ISF at self.q[q_index] on self's lags. where: "radial", or "parallel" or "perp" for sector models.
        Radial models fit the radial ISF only, sector models the sectors only."""
        F = {c: self.per_q[c][q_index] for c in _COLS}
        run = self._isf
        t = run.lags[:-1]
        phis = _sector_angles(run)
        if where == "radial":
            f = RADIAL_F[self.model](t, F["Gamma"], F["Omega"], F["sigma"])
        elif self.model == "sectors_separate":
            f = (sector_f(t, F["Gamma_parallel"], F["Omega"], F["sigma"], 0.0, phis["parallel"]) if where == "parallel"
                 else sector_f(t, F["Gamma"], F["Omega_perp"], F["sigma_perp"], 0.0, phis["perp"]))
        else:
            f = sector_f(t, F["Gamma"], F["Omega"], F["sigma"], F["theta"], phis[where])
        A, B = (F["A_perp"], F["B_perp"]) if where == "perp" else (F["A"], F["B"])
        return A * (1 - f) + B


# ---------------------------------------------------------------- first fit

def characteristic_time(isf, model="sinc", q_indices=None):
    """tau_c(q) from a fit of ISF = A (1 - f(tau / tau_c)) + B at every q, with f = sinc(tau / tau_c) (model="sinc",
    for drifting samples) or exp(-tau / tau_c) (model="exp"). Returns an (nq, 3) array of A, B, tau_c.

    Plot tau_c against q on log axes and choose q_range where it is a clean straight line.
    q_indices: fit only these q (the rest are NaN)."""
    f = {"exp": lambda t, c: np.exp(-t / c), "sinc": lambda t, c: np.sin(t / c) / (t / c)}[model]
    log_model = lambda p, t: np.log(np.maximum(p[0] * (1 - f(t, p[2])) + p[1], 1e-10))
    t = isf.lags[:-1]
    out = np.full((isf.radial.shape[-1], 3), np.nan)
    for iq in range(isf.radial.shape[-1]) if q_indices is None else q_indices:
        ddm = isf.radial[:-1, iq]
        if np.all(np.isfinite(ddm)) and (ddm > 0).all():
            out[iq] = leastsq(lambda p, t, logd: log_model(p, t) - logd, [np.ptp(isf.radial), ddm.min(), 1],
                              args=(t, np.log(ddm)))[0]
    if model == "sinc":
        out[:, 2] = np.abs(out[:, 2])   # sinc is even: -tau_c is the same fit as +tau_c
    return out


# ---------------------------------------------------------------- fitting one q

def _best(resid, starts, upper):
    best = None
    for p0 in starts:
        r = least_squares(resid, p0, bounds=(0, upper), x_scale="jac")
        if best is None or r.cost < best.cost:
            best = r
    return best


def _fit_radial(t, isf, f, has, starts):
    """A (1 - f(t, G, O, S)) + B; parameters A, B and those in has. starts: dicts."""
    free = [p for p in ("Gamma", "Omega", "sigma") if p in has]
    logi = np.log(isf)

    def unpack(p):
        return p[0], p[1], {"Gamma": 0.0, "Omega": 0.0, "sigma": 0.0, **dict(zip(free, p[2:]))}

    def resid(p):
        A, B, v = unpack(p)
        return np.log(np.maximum(A * (1 - f(t, v["Gamma"], v["Omega"], v["sigma"])) + B, 1e-10)) - logi

    upper = [np.inf] * (2 + len(free))
    if "sigma" in free:
        upper[-1] = SIGMA_MAX
    r = _best(resid, [[s["A"], s["B"]] + [s[p] for p in free] for s in starts], upper)
    A, B, v = unpack(r.x)
    return dict(A=A, B=B, **v, mse=2 * r.cost / len(t))


def _fit_joint(t, ix, iy, phis, has, starts):
    """Both sectors, own A, B each, shared Gamma, Omega (+ sigma, theta). starts: dicts."""
    free = ["A", "B", "A_perp", "B_perp", "Gamma", "Omega"] + [p for p in ("sigma", "theta") if p in has]
    logx, logy = np.log(ix), np.log(iy)

    def unpack(p):
        return {"sigma": 0.0, "theta": 0.0, **dict(zip(free, p))}

    def resid(p):
        v = unpack(p)
        fx = v["A"] * (1 - sector_f(t, v["Gamma"], v["Omega"], v["sigma"], v["theta"], phis["parallel"])) + v["B"]
        fy = v["A_perp"] * (1 - sector_f(t, v["Gamma"], v["Omega"], v["sigma"], v["theta"], phis["perp"])) + v["B_perp"]
        return np.concatenate([np.log(np.maximum(fx, 1e-10)) - logx, np.log(np.maximum(fy, 1e-10)) - logy])

    upper = [np.inf] * 6 + [SIGMA_MAX, np.pi / 2][:len(free) - 6]
    r = _best(resid, [[s[p] for p in free] for s in starts], upper)
    return dict(unpack(r.x), mse=2 * r.cost / (2 * len(t)))


def _fit_alone(t, isf, phis, starts):
    """Spread model with drift along the expected direction, on one sector. starts: [A, B, Gamma, Omega, sigma]."""
    logi = np.log(isf)
    resid = lambda p: np.log(np.maximum(p[0] * (1 - sector_f(t, p[2], p[3], p[4], 0.0, phis)) + p[1], 1e-10)) - logi
    r = _best(resid, starts, [np.inf] * 4 + [SIGMA_MAX])
    return dict(zip(("A", "B", "Gamma", "Omega", "sigma"), r.x), mse=2 * r.cost / len(t))


def _row(**v):
    out = np.full(len(_COLS), np.nan)
    for c in ("Gamma", "Omega", "sigma", "theta"):
        out[_C[c]] = 0.0
    for c, x in v.items():
        out[_C[c]] = x
    return out


def _sector_angles(isf):
    return {sec: np.radians(centre + isf.half_width * ANGLE_NODES) for sec, centre in (("parallel", 0.0), ("perp", 90.0))}


def _fit_q(name, isf, iq, first, fits):
    """Fit model name at q index iq. first: the first-fit array. fits: the models in name's needs."""
    m = MODELS[name]
    t = isf.lags[:-1]
    A0, B0, tau_c = first[iq, 0], first[iq, 1], abs(first[iq, 2])
    if m.data == "radial":
        data = isf.radial[:-1, iq]
        if name == "exp*J0*u_spread":
            starts = [dict(zip(_COLS, fits[n][iq]), sigma=s) for n in ("exp*J0", "exp*sinc") for s in SIGMA_STARTS]
        else:
            A0 = A0 if A0 > 0 else np.ptp(data)
            B0 = B0 if B0 > 0 else data.min()
            grid = [dict(A=A0, B=B0, Omega=k / tau_c, Gamma=g * k / tau_c, sigma=0.0)
                    for k in OMEGA_STARTS for g in GAMMA_STARTS]
            starts = grid if "Gamma" in m.has else grid[::len(GAMMA_STARTS)]   # no Gamma: one start per Omega
        return _row(**_fit_radial(t, data, RADIAL_F[name], m.has, starts))

    ix, iy = isf.parallel[:-1, iq], isf.perp[:-1, iq]
    phis = _sector_angles(isf)
    if name == "sectors_single":
        base = dict(A=np.ptp(ix), B=ix.min(), A_perp=np.ptp(iy), B_perp=iy.min())
        starts = [dict(base, Omega=k / tau_c, Gamma=g * k / tau_c) for k in OMEGA_STARTS for g in GAMMA_STARTS]
        return _row(**_fit_joint(t, ix, iy, phis, m.has, starts))
    if name == "sectors_spread":
        starts = [dict(zip(_COLS, fits["sectors_single"][iq]), sigma=s) for s in SIGMA_STARTS]
        return _row(**_fit_joint(t, ix, iy, phis, m.has, starts))
    J = dict(zip(_COLS, fits["sectors_spread"][iq]))
    if name == "sectors_angle":
        # also Omega from tau_c: with the drift mostly in the perp sector, sectors_spread (theta = 0) finds Omega ~ 0,
        # and with no drift term theta does nothing, so it would never move from its start
        starts = [dict(J, Omega=O, theta=np.radians(th)) for th in THETA_STARTS
                  for O in (J["Omega"],) + tuple(k / tau_c for k in OMEGA_STARTS)]
        return _row(**_fit_joint(t, ix, iy, phis, m.has, starts))
    sigmas = (J["sigma"], 0.05, 0.5)   # sectors_separate
    x = _fit_alone(t, ix, phis["parallel"], [[J["A"], J["B"], J["Gamma"], J["Omega"], s] for s in sigmas])
    # drift is weak across the drift direction, so Omega gets more starts there
    y = _fit_alone(t, iy, phis["perp"], [[J["A_perp"], J["B_perp"], J["Gamma"], J["Omega"] * k, s]
                                         for s in sigmas for k in (1, 0.3, 3)])
    return _row(A=x["A"], B=x["B"], A_perp=y["A"], B_perp=y["B"], Gamma=y["Gamma"], Omega=x["Omega"], sigma=x["sigma"],
                Gamma_parallel=x["Gamma"], Omega_perp=y["Omega"], sigma_perp=y["sigma"], mse=(x["mse"] + y["mse"]) / 2)


# ---------------------------------------------------------------- fitting

def fit_models(isf, names, q_range):
    """Fit the named models over q_min <= q <= q_max [µm⁻¹]. q_range: (q_min, q_max). Returns {name: Fit}.

    Models that start from others (Model.needs) fit those first, and share them when several are named. Each Fit keeps
    the q where every model named here could be fitted, so results within one call are on the same q."""
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise KeyError(f"unknown models {unknown}; choose from {list(MODELS)}")
    if any(MODELS[n].data == "sector" for n in names) and isf.parallel is None:
        raise ValueError("sector models need an ISF computed with sector_half_width")
    q_range = np.flatnonzero((isf.q >= q_range[0]) & (isf.q <= q_range[1]))
    if len(q_range) < 2:
        raise ValueError("q_range holds fewer than 2 q values")
    first = characteristic_time(isf, "sinc", q_range)
    fits = {}

    def usable(m, iq):
        data = [isf.radial] if m.data == "radial" else [isf.parallel, isf.perp]
        return all(np.all(np.isfinite(d[:-1, iq])) and (d[:-1, iq] > 0).all() for d in data)

    def get(name):
        if name in fits:
            return
        m = MODELS[name]
        for n in m.needs:
            get(n)
        out = np.full((len(isf.q), len(_COLS)), np.nan)
        for iq in q_range:
            if usable(m, iq) and np.isfinite(first[iq, 2]) and all(np.isfinite(fits[n][iq, -1]) for n in m.needs):
                out[iq] = _fit_q(name, isf, iq, first, fits)
        fits[name] = out

    for name in names:
        get(name)
    ok = q_range[np.all([np.isfinite(fits[n][q_range, -1]) for n in names], axis=0)]
    return {n: Fit(n, isf.q[ok], {c: fits[n][ok, i] for i, c in enumerate(_COLS)}, isf.lags[:-1][-1],
                   len(isf.lags) - 1, isf) for n in names}


def fit(isf, name, q_range):
    """Fit one model. Returns a Fit."""
    return fit_models(isf, [name], q_range)[name]
