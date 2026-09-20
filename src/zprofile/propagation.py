"""
Propagation constant of the cable from its own transmission measurement.

For a cable of length l with (fixture-free) differential transmission
S_dd21(f),

    alpha(f) = -ln|S_dd21| / l              (Np/m)
    beta(f)  = -phi_abs(f) / l              (rad/m)

where phi_abs is the unwrapped phase with the integer number of turns at
the first point fixed from the low-frequency group delay.  The phase
velocity v_p = omega/beta at the top of the band is the velocity used to
turn round-trip time into distance; the attenuation is also fitted to the
two-term form  alpha = a sqrt(f) + b f  (conductor + dielectric loss) so
that a smooth, noise-free gamma is available for the peeling recursion, and
the fit residual is reported.  Multiple reflections inside the cable put a
ripple on |S21| that the fit averages out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transform import C0

__all__ = ["Propagation", "gamma_from_s21", "gamma_from_model"]


@dataclass
class Propagation:
    f: np.ndarray
    alpha: np.ndarray        # Np/m, measured
    beta: np.ndarray         # rad/m, measured (absolute)
    alpha_fit: np.ndarray    # c + a sqrt f + b f
    a: float
    b: float
    c: float                 # DC-resistance term (Np/m), >= 0
    velocity: float          # m/s (phase velocity near the top of the band)
    nvp: float
    fit_rms_np_per_m: float
    source: str

    def _beta(self, f: np.ndarray) -> np.ndarray:
        be = np.interp(f, self.f, self.beta)
        out = (f > self.f.max()) | (f < self.f.min())
        return np.where(out, 2 * np.pi * f / self.velocity, be)

    def alpha_parts(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(conductor-type attenuation c + a sqrt f, dielectric-type b f) in Np/m."""
        return self.c + self.a * np.sqrt(f), self.b * f

    def z0_factor(self, f: np.ndarray) -> np.ndarray:
        """Complex, dimensionless factor by which the high-frequency-limit impedance
        Zc is multiplied to obtain the lossy line's characteristic impedance:
            Z0(f) = Zc sqrt( (1 - j 2 alpha_c/beta) / (1 - j 2 alpha_d/beta) ),
        from R/(omega L) = 2 alpha_c/beta and G/(omega C) = 2 alpha_d/beta."""
        f = np.asarray(f, float)
        ac, ad = self.alpha_parts(f)
        be = np.maximum(self._beta(f), 1e-12)
        return np.sqrt((1 - 2j * ac / be) / (1 - 2j * ad / be))

    def gamma(self, f: np.ndarray | None = None, smooth: bool = True) -> np.ndarray:
        """Propagation constant.  With ``smooth`` the physically consistent form
        gamma = j beta sqrt((1 - j 2 alpha_c/beta)(1 - j 2 alpha_d/beta)) built from the
        fitted attenuation terms is returned (its real part equals c + a sqrt f + b f to
        first order); otherwise the raw measured alpha + j beta."""
        if f is None:
            f = self.f
        f = np.asarray(f, float)
        if not smooth:
            return np.interp(f, self.f, self.alpha) + 1j * self._beta(f)
        ac, ad = self.alpha_parts(f)
        be = self._beta(f)
        return 1j * be * np.sqrt((1 - 2j * ac / np.maximum(be, 1e-12)) * (1 - 2j * ad / np.maximum(be, 1e-12)))


def gamma_from_s21(f: np.ndarray, s21: np.ndarray, length_m: float) -> Propagation:
    f = np.asarray(f, float)
    s21 = np.asarray(s21, complex)
    alpha = -np.log(np.maximum(np.abs(s21), 1e-12)) / length_m
    ph = np.unwrap(np.angle(s21))
    if np.max(np.abs(np.diff(ph))) > 0.9 * np.pi:
        raise ValueError("frequency grid too coarse to unwrap S21 phase")
    n_low = max(3, f.size // 10)
    tau_g = -np.polyfit(2 * np.pi * f[:n_low], ph[:n_low], 1)[0]
    k = np.round((ph[0] + 2 * np.pi * f[0] * tau_g) / (2 * np.pi))
    ph_abs = ph - 2 * np.pi * k
    beta = -ph_abs / length_m
    n_top = max(5, f.size // 20)
    velocity = float(np.median(2 * np.pi * f[-n_top:] / beta[-n_top:]))
    from scipy.optimize import nnls
    A = np.column_stack([np.ones_like(f), np.sqrt(f), f])
    coef, _ = nnls(A, alpha)
    c, a, b = (float(coef[0]), float(coef[1]), float(coef[2]))
    alpha_fit = c + a * np.sqrt(f) + b * f
    rms = float(np.sqrt(np.mean((alpha - alpha_fit) ** 2)))
    return Propagation(f, alpha, beta, alpha_fit, a, b, c, velocity, velocity / C0, rms, "S21")


def gamma_from_model(f: np.ndarray, nvp: float = 0.68, a: float = 2.0e-6, b: float = 2.0e-11) -> Propagation:
    """Fallback when no transmission measurement exists: a = 2e-6 Np/m/sqrt(Hz), b = 2e-11 Np/m/Hz
    are typical of a 0.35 mm^2 automotive STP (about 0.6 dB/m at 600 MHz)."""
    f = np.asarray(f, float)
    v = nvp * C0
    alpha = a * np.sqrt(f) + b * f
    beta = 2 * np.pi * f / v
    return Propagation(f, alpha, beta, alpha.copy(), a, b, 0.0, v, nvp, 0.0, "model")
