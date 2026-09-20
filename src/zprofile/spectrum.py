"""
Spectrum preparation: uniform grid and extrapolation to zero frequency.

A VNA sweep starts at f_min > 0.  The step response needs S(0) and every
sample in the gap 0 < f < f_min, and the *entire* low-frequency part of the
step response (the level the trace settles to after each discontinuity)
depends on how that gap is filled.  Four estimators are provided:

``zero``
    S(f) = 0 below f_min.  Wrong for anything but a perfectly matched cable;
    kept as the baseline that many quick scripts silently use.
``linear``
    Re S extrapolated linearly from the first ``n_fit`` points, Im S taken
    linearly to 0 (S(0) of a passive reciprocal one-port is real).
``rational``
    Fit of the first ``n_fit`` points to the causal first-order model
    S(f) = (a + j b f) / (1 + j c f), which is what a resistive mismatch in
    series/parallel with one reactive element looks like; S(0) = a.
``physical``
    S(0) from the known DC picture: the far end terminated in Z_term through
    the loop resistance R_loop = R'_dc * length gives
    S(0) = (R_loop + Z_term - Z_ref) / (R_loop + Z_term + Z_ref); the gap is
    then bridged from that anchor to the first measured point with a
    sqrt(f)-shaped curve, which is the shape the skin-effect input impedance
    of a long line has at low frequency.

The choice matters: on the reference cables the four estimators move the
settled impedance level by up to a few ohms (quantified in the sensitivity
study), while the *peeled* profile, which only ever reads the early part of
each cell's response, is nearly insensitive to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PreparedSpectrum", "prepare", "DC_METHODS"]

DC_METHODS = ("zero", "linear", "rational", "physical")


@dataclass
class PreparedSpectrum:
    f: np.ndarray          # uniform grid from 0 Hz, step df
    s: np.ndarray          # complex spectrum on that grid (gap filled)
    df: float
    k_first: int           # index of the first *measured* point
    s_dc: complex
    dc_method: str


def _uniform(f, s, rel_tol=1e-6):
    d = np.diff(f)
    if np.allclose(d, d[0], rtol=rel_tol):
        return f, s, float(d[0])
    step = float(np.min(d))
    fu = np.arange(f[0], f[-1] + step / 2, step)
    return fu, np.interp(fu, f, s.real) + 1j * np.interp(fu, f, s.imag), step


def _regrid_from_dc(f, s):
    """Uniform grid that contains 0 Hz exactly; returns (f_full, s_full_masked, k_first)."""
    f, s, df = _uniform(f, s)
    k0 = int(round(f[0] / df))
    if abs(k0 * df - f[0]) > 1e-3 * df:  # f_min not a multiple of df: resample
        n = int(np.floor(f[-1] / df))
        fu = np.arange(0, n + 1) * df
        m = fu >= f[0]
        su = np.zeros(fu.size, complex)
        su[m] = np.interp(fu[m], f, s.real) + 1j * np.interp(fu[m], f, s.imag)
        return fu, su, int(np.argmax(m)), df
    fu = np.arange(0, k0 + f.size) * df
    su = np.zeros(fu.size, complex)
    su[k0:] = s
    return fu, su, k0, df


def _rational_fit(f, s):
    """Least squares for S = (a + j b f)/(1 + j c f)  <=>  S + j c f S = a + j b f."""
    A = np.column_stack([np.ones_like(f), 1j * f, -1j * f * s])
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    a, b, c = coef
    return a, b, c


def prepare(f: np.ndarray, s: np.ndarray, method: str = "linear", n_fit: int = 8,
            r_loop_ohm: float | None = None, z_term: float | None = None, z_ref: float = 100.0) -> PreparedSpectrum:
    f = np.asarray(f, float)
    s = np.asarray(s, complex)
    if method not in DC_METHODS:
        raise ValueError(f"dc method must be one of {DC_METHODS}")
    fu, su, k0, df = _regrid_from_dc(f, s)
    if k0 == 0:
        return PreparedSpectrum(fu, su, df, 0, su[0], "measured")
    fm, sm = fu[k0:], su[k0:]
    n_fit = max(3, min(n_fit, fm.size))
    fg = fu[:k0]
    if method == "zero":
        s_dc = 0.0
        gap = np.zeros(k0, complex)
    elif method == "linear":
        p = np.polyfit(fm[:n_fit], sm[:n_fit].real, 1)
        s_dc = float(np.clip(np.polyval(p, 0.0), -1, 1))
        gap = s_dc + (sm[0].real - s_dc) * fg / fm[0] + 1j * sm[0].imag * fg / fm[0]
    elif method == "rational":
        a, b, c = _rational_fit(fm[:n_fit], sm[:n_fit])
        s_dc = float(np.clip(a.real, -1, 1))
        gap = (a + 1j * b * fg) / (1 + 1j * c * fg)
        gap[0] = s_dc
        # make the join continuous
        gap = gap + (sm[0] - (a + 1j * b * fm[0]) / (1 + 1j * c * fm[0])) * fg / fm[0]
    else:  # physical
        if r_loop_ohm is None:
            raise ValueError("physical DC extrapolation needs r_loop_ohm (= R'_dc * length)")
        zt = z_ref if z_term is None else z_term
        s_dc = (r_loop_ohm + zt - z_ref) / (r_loop_ohm + zt + z_ref)
        # sqrt(f)-shaped bridge from the anchor to the first measured point
        u = np.sqrt(fg / fm[0])
        gap = s_dc + (sm[0].real - s_dc) * u + 1j * sm[0].imag * u
    su[:k0] = gap
    return PreparedSpectrum(fu, su, df, k0, complex(s_dc), method)
