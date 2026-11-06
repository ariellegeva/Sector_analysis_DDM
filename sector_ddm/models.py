"""The intermediate scattering functions f(q, tau) that can be fitted, as ISF = A (1 - f) + B at each q.

Drift at speed u at angle theta to the expected direction gives, for a q vector at angle phi,
    f = exp(-Gamma tau) cos(Omega u tau cos(phi - theta))
(cos, not J0: the ISF is even in q, so the sign of the drift is lost). Sector models average this over the sector's
angles instead of assuming f is constant across the sector. Speeds u = v_i / <v> are lognormal with mean 1 and
log-width sigma. Gamma = D q^2 and Omega = v q.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.special import j0

N_SPEEDS = 24                 # Gauss-Hermite nodes for the average over the (lognormal) speeds
N_ANGLES = 8                  # Gauss-Legendre nodes for the average over a sector's angles

_z, _wz = hermegauss(N_SPEEDS)
_wz = _wz / _wz.sum()
ANGLE_NODES, _wx = leggauss(N_ANGLES)
_wx = _wx / _wx.sum()


@dataclass(frozen=True)
class Model:
    name: str
    data: str                 # "radial", or "sector" for the parallel and perp ISFs
    k: int                    # free parameters per q
    has: tuple                # which of Gamma, Omega, sigma, theta it fits
    needs: tuple              # models whose fits give its starting points, fitted first
    description: str


MODELS = {m.name: m for m in (
    Model("exp", "radial", 3, ("Gamma",), (), "diffusion only"),
    Model("sinc", "radial", 3, ("Omega",), (), "drift in random 3D directions"),
    Model("J0", "radial", 3, ("Omega",), (), "drift at one speed in the plane"),
    Model("exp*sinc", "radial", 4, ("Gamma", "Omega"), (), "sinc + diffusion"),
    Model("exp*J0", "radial", 4, ("Gamma", "Omega"), (), "J0 + diffusion"),
    Model("exp*J0*u_spread", "radial", 5, ("Gamma", "Omega", "sigma"), ("exp*J0", "exp*sinc"),
          "exp*J0 + speed spread"),
    Model("sectors_single", "sector", 6, ("Gamma", "Omega"), (),
          "both sectors, one speed along the drift direction, shared Gamma and Omega"),
    Model("sectors_spread", "sector", 7, ("Gamma", "Omega", "sigma"), ("sectors_single",),
          "sectors_single + speed spread"),
    Model("sectors_angle", "sector", 8, ("Gamma", "Omega", "sigma", "theta"), ("sectors_spread",),
          "sectors_spread + free drift angle"),
    Model("sectors_separate", "sector", 10, ("Gamma", "Omega", "sigma"), ("sectors_spread",),
          "the spread model on each sector alone"),
)}


def sinc(x):
    """sin(x) / x"""
    return np.sinc(x / np.pi)


def spread_j0(t, Omega, sigma):
    """<J0(Omega u t)> over a lognormal u with mean 1 and log-width sigma."""
    u = np.exp(sigma * _z - sigma**2 / 2)
    return j0(Omega * np.outer(t, u)) @ _wz


def sector_f(t, Gamma, Omega, sigma, theta, phis):
    """exp(-Gamma t) <cos(Omega u t cos(phi - theta))> over lognormal u (mean 1) and the sector angles phis."""
    u = np.exp(sigma * _z - sigma**2 / 2)
    phase = Omega * t[:, None, None] * u[None, :, None] * np.cos(phis - theta)[None, None, :]
    return np.exp(-Gamma * t) * np.einsum("tuf,u,f->t", np.cos(phase), _wz, _wx)


RADIAL_F = {   # f(t, Gamma, Omega, sigma)
    "exp": lambda t, G, O, S: np.exp(-G * t),
    "sinc": lambda t, G, O, S: sinc(O * t),
    "exp*sinc": lambda t, G, O, S: np.exp(-G * t) * sinc(O * t),
    "J0": lambda t, G, O, S: j0(O * t),
    "exp*J0": lambda t, G, O, S: np.exp(-G * t) * j0(O * t),
    "exp*J0*u_spread": lambda t, G, O, S: np.exp(-G * t) * spread_j0(t, O, S),
}


def speed_cv(sigma):
    """Relative spread (std/mean) of a lognormal with log-width sigma."""
    return np.sqrt(np.expm1(np.asarray(sigma, dtype=float)**2))
