"""
EXPERIMENTAL - model-based refinement of the peeled profile (regularised
Gauss-Newton).  Not used by the default pipeline; see the report, section
"What did not work", for the validation result that led to that decision:
the refinement reduces the *data* misfit by a factor of 3-20 but, because
the cell model reproduces the distributed loss reflection of a real cable
only to first order in alpha/beta, it converts that model error into
impedance and *increases* the error against the truth (uniform 15 m cable:
+2.4 ohm).  It is kept as a documented starting point for a version with an
exact lossy-line cell model.

Layer peeling is a one-pass, causal estimator: every cell is read once and
never revisited, so an under-resolved feature leaves an over/undershoot in
the cells that follow it.  The refinement closes the loop.  With the same
cell model as the peeling (impedances Z_k, per-cell propagation constants
from the measured attenuation under the chosen velocity hypothesis, the
far end terminated in Z_ref) the reflection coefficient of the whole chain
is computed *forward*,

    S11_model(f; Z) = reflection of  cell_1 . cell_2 . ... . cell_N . Z_ref ,

and the cell impedances are adjusted to minimise

    J(Z) = sum_f | W(f) (S11_model(f; Z) - S11_meas(f)) |^2  +  lambda sum_k (Z_{k+1} - 2 Z_k + Z_{k-1})^2 ,

where W is the window times the rise-time filter (the data are trusted only
inside the resolution bandwidth) and the second term is a smoothness prior
that keeps features finer than the resolution from being invented.  The
Jacobian is built by finite differences (one forward model per cell, a few
milliseconds each) and each Gauss-Newton step is clipped to a trust radius.
Typically three to five iterations reduce the misfit by a factor of three
and remove the post-feature overshoot; the smoothness weight is the knob
that trades ringing against feature depth and is part of the sensitivity
budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .peel import PeelResult
from .propagation import Propagation
from .spectrum import PreparedSpectrum
from .windows import gaussian_rise_filter, one_sided_window

__all__ = ["RefineResult", "forward_s11", "refine"]


@dataclass
class RefineResult:
    z: np.ndarray
    x: np.ndarray
    misfit_before: float
    misfit_after: float
    iterations: int
    lam: float


def _cell_params(prop: Propagation, f: np.ndarray, z_ref: float, velocity_model: str, lossy: bool, complex_z0: bool):
    beta_f = np.maximum(prop.gamma(f).imag, 1e-12)
    ac_f, ad_f = prop.alpha_parts(f)

    def cell(zk: float):
        r = zk / z_ref
        if velocity_model == "capacitance":
            vr, ac, ad = r, ac_f / r, ad_f * r
        elif velocity_model == "inductance":
            vr, ac, ad = 1.0 / r, ac_f / r, ad_f * r
        else:
            vr, ac, ad = 1.0, ac_f / r, ad_f * r
        be = beta_f / vr
        if not lossy:
            return 1j * be, 1.0, vr
        zf = np.sqrt((1 - 2j * ac / be) / (1 - 2j * ad / be)) if complex_z0 else 1.0
        g = 1j * be * np.sqrt((1 - 2j * ac / be) * (1 - 2j * ad / be))
        return g, zf, vr
    return cell


def forward_s11(z: np.ndarray, dx_ref: float, f: np.ndarray, z_ref: float, prop: Propagation,
                velocity_model: str = "capacitance", lossy: bool = True, complex_z0: bool = True) -> np.ndarray:
    """S11 (in the Z_ref system) of the chain of cells with impedances ``z``;
    every cell has the round-trip delay of ``dx_ref`` at the reference velocity."""
    cell = _cell_params(prop, f, z_ref, velocity_model, lossy, complex_z0)
    # chain matrices (ABCD) multiplied from the far end backwards is cheaper: Zin recursion
    # Zin_k = Zc (Zin_{k+1} + Zc tanh(g l)) / (Zc + Zin_{k+1} tanh(g l))
    zin = np.full(f.size, z_ref, complex)
    for zk in z[::-1]:
        g, zf, vr = cell(float(zk))
        zc = zk * zf
        th = np.tanh(g * dx_ref * vr)
        zin = zc * (zin + zc * th) / (zc + zin * th)
    return (zin - z_ref) / (zin + z_ref)


def refine(prep: PreparedSpectrum, prop: Propagation, pr: PeelResult, z_ref: float, window: str = "kaiser",
           beta: float = 6.0, atten_db: float = 60.0, t_rise: float | None = 500e-12, lam: float = 2e-4,
           iterations: int = 4, trust_ohm: float = 5.0, velocity_model: str = "capacitance", lossy: bool = True,
           complex_z0: bool = True, gap_weight: float = 0.1, dz_fd: float = 0.05,
           f_low: float | None = None, length_m: float | None = None) -> RefineResult:
    f = prep.f
    w = one_sided_window(f.size, window, beta, atten_db)
    if t_rise is not None:
        w = w * gaussian_rise_filter(f, t_rise)
    w[: prep.k_first] *= gap_weight
    # the level of the profile is set by the peeling (early-time reads); the
    # refinement is only allowed to move *features*, so frequencies below a few
    # round-trip periods of the whole cable (where only the level and the
    # distributed loss reflection live) get zero weight
    if f_low is None:
        f_low = 4.0 * prop.velocity / (2.0 * length_m) if length_m else 0.0
    w = w * (0.5 * (1 - np.cos(np.pi * np.clip(f / max(f_low, 1e-9), 0, 1))))
    z = pr.z.copy()
    n = z.size

    def resid(zz):
        r = (forward_s11(zz, pr.dx, f, z_ref, prop, velocity_model, lossy, complex_z0) - prep.s) * w
        return np.concatenate([r.real, r.imag])

    D2 = np.zeros((max(n - 2, 0), n))
    for i in range(n - 2):
        D2[i, i:i + 3] = [1.0, -2.0, 1.0]
    r0 = resid(z)
    m0 = float(np.linalg.norm(r0))
    it_done = 0
    for it in range(iterations):
        J = np.empty((r0.size, n))
        for k in range(n):
            zp = z.copy()
            zp[k] += dz_fd
            J[:, k] = (resid(zp) - r0) / dz_fd
        H = J.T @ J + lam * (D2.T @ D2) + 1e-9 * np.eye(n)
        g = J.T @ r0 + lam * (D2.T @ (D2 @ z))
        step = np.clip(-np.linalg.solve(H, g), -trust_ohm, trust_ohm)
        z_new = np.clip(z + step, 0.2 * z_ref, 5 * z_ref)
        r_new = resid(z_new)
        it_done = it + 1
        if np.linalg.norm(r_new) > np.linalg.norm(r0):   # no improvement: stop
            break
        z, r0 = z_new, r_new
        if np.abs(step).max() < 0.02:
            break
    return RefineResult(z, pr.x, m0, float(np.linalg.norm(r0)), it_done, lam)
