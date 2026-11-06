"""From frames to the image structure function (ISF).

ISF(q, dt) = < | FFT[ I(x, t + dt) - I(x, t) ] |^2 >_t

averaged over all directions of q (radial), and optionally over two opposite pairs of wedges in q space: one centred
on the expected drift direction (parallel) and one perpendicular to it (perp).

Derived from https://github.com/samuelmcdermott/DDM (GPL-2.0).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ISF:
    """An ISF, ready for fitting. radial, parallel and perp have shape (number of lags, number of q)."""
    radial: np.ndarray
    lags: np.ndarray                  # s
    q: np.ndarray                     # µm⁻¹
    parallel: np.ndarray = None       # None if computed without sectors
    perp: np.ndarray = None
    half_width: float = None          # degrees, half the opening angle of each sector

    def save(self, path):
        arrays = {k: v for k, v in vars(self).items() if v is not None}
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(**{k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files})


# ---------------------------------------------------------------- frames

def load_video(path):
    """All frames of a video as an (n, height, width) grayscale array. Needs opencv-python."""
    import cv2
    video = cv2.VideoCapture(str(path))
    n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = np.empty((n, h, w), dtype=np.int16)   # preallocated: a list + np.array would need twice the memory
    count = 0
    while count < n:
        ok, image = video.read()
        if not ok:
            break
        frames[count] = image.mean(axis=2).astype(np.int16)
        count += 1
    return frames[:count]


# ---------------------------------------------------------------- time-averaged spectra

def _spectrum_diff(im0, im1):
    return np.abs(np.fft.fft2(im1 - im0.astype(float)))**2


def _time_averaged(frames, lag, max_couples):
    """Average spectrum of up to max_couples frame pairs lag frames apart."""
    increment = max([(len(frames) - lag) / max_couples, 1])
    starts = np.arange(0, len(frames) - lag, increment).astype(int)
    starts = starts[starts + lag < len(frames)]
    total = np.zeros(frames.shape[1:])
    for t in starts:
        total += _spectrum_diff(frames[t], frames[t + lag])
    return total / len(starts)


def _exact_pairs(frame_index, lag):
    """(i, j) positions of the recorded frames exactly lag camera frames apart."""
    j = np.searchsorted(frame_index, frame_index + lag)
    ok = j < len(frame_index)
    ok[ok] = frame_index[j[ok]] == frame_index[ok] + lag
    return np.flatnonzero(ok), j[ok]


def _time_averaged_grid(frames, frame_index, lag, max_couples):
    """As _time_averaged for videos with dropped frames, using only pairs exactly lag camera frames apart."""
    i0, i1 = _exact_pairs(frame_index, lag)
    pick = np.unique(np.linspace(0, len(i0) - 1, min(max_couples, len(i0))).astype(int))
    total = np.zeros(frames.shape[1:])
    for a, b in zip(i0[pick], i1[pick]):
        total += _spectrum_diff(frames[a], frames[b])
    return total / len(pick)


# ---------------------------------------------------------------- averaging over q

class _RadialAverager:
    def __init__(self, shape):
        self.dists = np.sqrt(np.fft.fftfreq(shape[0])[:, None]**2 + np.fft.fftfreq(shape[1])[None, :]**2)
        self.dists[0] = 0
        self.dists[:, 0] = 0
        self.bins = np.arange(max(shape) / 2 + 1) / float(max(shape))
        self.hd = np.histogram(self.dists, self.bins)[0]

    def __call__(self, im):
        return np.histogram(self.dists, self.bins, weights=im)[0] / self.hd


class _SectorAverager(_RadialAverager):
    """Radial average over the q within half_width degrees of the line at centre_deg, measured from image +x towards
    +y (down the rows, clockwise on screen). q and -q count as the same sector. Bins with no pixels give NaN."""

    def __init__(self, shape, centre_deg, half_width):
        super().__init__(shape)
        qy, qx = np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]
        d = (np.degrees(np.arctan2(qy, qx)) - centre_deg) % 180
        self.mask = np.broadcast_to(np.minimum(d, 180 - d) <= half_width, shape)
        self.sector_dists = self.dists[self.mask]
        self.hd = np.histogram(self.sector_dists, self.bins)[0]

    def __call__(self, im):
        hw = np.histogram(self.sector_dists, self.bins, weights=im[self.mask])[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.hd > 0, hw / self.hd, np.nan)


def _log_spaced(n, points_per_decade):
    decades = np.log10(n)
    return np.unique(np.logspace(0, decades, num=int(decades * points_per_decade), base=10, endpoint=False).astype(int))


# ---------------------------------------------------------------- the ISF

def compute_isf(frames, fps, pixel_size, sector_half_width=20.0, drift_angle_deg=0.0, max_couples=300,
                points_per_decade=30, frame_index=None):
    """ISF of an (n, height, width) array of frames.

    fps                camera frame rate
    pixel_size         µm per pixel
    sector_half_width  degrees either side of each sector's centre line. None: radial ISF only
    drift_angle_deg    expected drift direction: the parallel sector is centred here, the perp sector 90° from it.
                       Measured from image +x towards +y (clockwise on screen); 0 is left to right
    max_couples        at most this many frame pairs are averaged per lag
    points_per_decade  density of the log-spaced lags
    frame_index        for videos with dropped frames: the camera-frame number of each frame in frames, increasing.
                       Only pairs an exact number of camera frames apart are used, and lags with no such pair are left out
    """
    frames = np.asarray(frames)
    if frame_index is None:
        lags = _log_spaced(len(frames), points_per_decade)
    else:
        frame_index = np.asarray(frame_index)
        if len(frame_index) != len(frames):
            raise ValueError(f"{len(frame_index)} frame indices for {len(frames)} frames")
        lags = _log_spaced(frame_index[-1] - frame_index[0] + 1, points_per_decade)
        lags = lags[[len(_exact_pairs(frame_index, lag)[0]) > 0 for lag in lags]]

    shape = frames.shape[1:]
    radial_avg = _RadialAverager(shape)
    radial = np.zeros((len(lags), len(radial_avg.hd)))
    sectors = {}
    if sector_half_width is not None:
        sectors = {"parallel": _SectorAverager(shape, drift_angle_deg, sector_half_width),
                   "perp": _SectorAverager(shape, drift_angle_deg + 90.0, sector_half_width)}
    sector_isf = {k: np.zeros_like(radial) for k in sectors}
    for i, lag in enumerate(lags):
        spectrum = (_time_averaged(frames, lag, max_couples) if frame_index is None
                    else _time_averaged_grid(frames, frame_index, lag, max_couples))
        radial[i] = radial_avg(spectrum)
        for k, average in sectors.items():
            sector_isf[k][i] = average(spectrum)

    q = 2 * np.pi / (2 * radial.shape[-1] * pixel_size) * np.arange(radial.shape[-1])
    return ISF(radial=radial, lags=lags / fps, q=q, half_width=sector_half_width, **sector_isf)
