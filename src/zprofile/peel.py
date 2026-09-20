"""
Loss-aware layer peeling (dynamic deconvolution) in the frequency domain.

The cable is modelled as a chain of uniform cells of length dx, each a lossy
transmission line with its own (real, high-frequency-limit) characteristic
impedance Z_k and the *measured* propagation constant gamma(f) of the cable.
Let G_k(f) be the reflection coefficient looking into cell k, referenced to
the real impedance Z_ref.  Two steps per cell:

The cell impedance is the *high-frequency limit* Zc = sqrt(L/C), a real
number.  The distributed low-frequency reflection of a lossy line (the
sqrt(t) rise of a TDR trace, from Z0(f) = Zc sqrt((1 - j2 alpha_c/beta)/
(1 - j2 alpha_d/beta)) ) is deliberately *not* attributed to the cells:
because every cell is read at early time, that slow term never accumulates
along the recursion, and the residual bias it leaves is validated at
+0.3 ohm on a 15 m cable at 600 MHz (+0.1 ohm at 6 GHz).  A variant with the
complex Z0(f) in the cell model (``complex_z0=True``) is provided for
completeness; it needs a DC-consistent read and is not the default.

1. **Read.**  Z_k is read from the early part of the step response of G_k,
   r_k = step_k(t_read), Z_k = Z_ref (1 + r_k)/(1 - r_k), with t_read equal
   to the effective 10-90 % rise time of the measurement (window plus
   rise-time filter): early enough that the distributed sqrt(t) reflection
   of the lossy line has not built up, late enough that the edge has
   settled.

2. **Peel.**  With the cell's own two-port (a line of impedance Z_k, length
   dx and propagation constant gamma, in the Z_ref system:
   S11 = (Z_k^2 - Z_ref^2) sinh(gamma dx)/D,  S21 = 2 Z_k Z_ref / D,
   D = 2 Z_k Z_ref cosh(gamma dx) + (Z_k^2 + Z_ref^2) sinh(gamma dx))
   the reflection looking into the *next* cell follows from the exact
   load-extraction identity
       G_{k+1} = (G_k - S11) / ( S11 (G_k - S11) + S21^2 ).

Step 2 removes, exactly for the discrete model, the three effects the plain
transform gets wrong: transmission loss through earlier mismatches,
re-reflections between them, and the attenuation and dispersion of the
lossy medium in between.  The division by S21^2 ~ e^{-2 gamma dx} is an
inverse propagation; its gain e^{2 alpha(f) x} grows with distance and
frequency and would eventually amplify noise without bound, so a
Wiener-like low-pass  W(x_k, f) = 1 / (1 + (e^{2 alpha(f) x_k} / G_max)^2)
is imposed on the residual (applied incrementally so that the total filter
at distance x_k is exactly W(x_k, f)): frequencies whose round-trip
attenuation exceeds G_max (default 20 dB) are progressively discarded,
which makes the resolution degrade gracefully with distance instead of the
noise exploding.
Because the accumulated regularisation slows the edge, the read time is
recomputed from the 10-90 % rise time of the total filter whenever the
regularisation has bitten, so that each cell is still read on a settled
step rather than on a ringing edge.
Setting ``lossy=False`` uses gamma = j beta only (classic lossless peeling).

References: Bruckstein & Kailath 1987 (inverse scattering on discrete
transmission-line models); Hayden & Tripathi 1994 (impedance profiles from
TDR with multiple-reflection correction); Dunsmore 2020, ch. 7.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .propagation import Propagation
from .spectrum import PreparedSpectrum
from .transform import step_response
from .windows import effective_rise_time

__all__ = ["PeelResult", "peel"]


@dataclass
class PeelResult:
    x: np.ndarray            # cell centres, m
    z: np.ndarray            # impedance per cell, ohm
    rho: np.ndarray          # read reflection per cell
    residual: np.ndarray     # rms of the residual spectrum after each cell (diagnostic)
    dx: float
    t_read: float
    t_cell: float
    lossy: bool
    gain_max_db: float
    t_read_final: float = 0.0   # read time used at the far end (grows with the regularisation)


def _cell_two_port(zk, zref: float, gamma: np.ndarray, dx: float):
    """S11 and S21 (in the Z_ref system) of a line of characteristic impedance zk
    (scalar or per-frequency complex), length dx and propagation constant gamma."""
    sh, ch = np.sinh(gamma * dx), np.cosh(gamma * dx)
    den = 2 * zk * zref * ch + (zk ** 2 + zref ** 2) * sh
    s11 = (zk ** 2 - zref ** 2) * sh / den
    s21 = 2 * zk * zref / den
    return s11, s21


def peel(prep: PreparedSpectrum, prop: Propagation, z_ref: float, x_max: float,
         window: str = "kaiser", beta: float = 6.0, atten_db: float = 60.0, t_rise: float | None = 500e-12,
         t_cell: float | None = None, t_read: float | None = None, lossy: bool = True,
         gain_max_db: float = 20.0, rho_clip: float = 0.9, n_fft: int | None = None,
         complex_z0: bool = True, newton_steps: int = 2, velocity_model: str = "capacitance") -> PeelResult:
    f = prep.f
    f_max = float(f[-1])
    t_eff = effective_rise_time(window, f_max, t_rise, beta, atten_db)
    t_cell = t_eff if t_cell is None else t_cell
    t_read = t_eff if t_read is None else t_read
    v = prop.velocity
    dx = v * t_cell / 2.0
    n_cells = int(np.ceil(x_max / dx))
    alpha = prop.gamma(f).real
    beta_f = np.maximum(prop.gamma(f).imag, 1e-12)
    ac_f, ad_f = prop.alpha_parts(f)
    if velocity_model not in ("capacitance", "inductance", "constant"):
        raise ValueError("velocity_model must be capacitance, inductance or constant")

    def cell_gamma(zk: float):
        """Propagation constant, complex-impedance factor and velocity ratio of a
        cell of impedance zk, under the chosen hypothesis for *why* the impedance
        differs from the reference: a capacitance change (C' = C (Zr/zk)^2, so
        v ~ zk, conductor loss ~ 1/zk, dielectric loss ~ zk), an inductance
        change (the reverse) or no velocity change."""
        r = zk / z_ref
        if velocity_model == "capacitance":
            vr, ac, ad = r, ac_f / r, ad_f * r
        elif velocity_model == "inductance":
            vr, ac, ad = 1.0 / r, ac_f / r, ad_f * r
        else:
            vr, ac, ad = 1.0, ac_f / r, ad_f * r
        be = beta_f / vr
        if not lossy:
            return 1j * be, np.ones_like(f), vr
        zf = np.sqrt((1 - 2j * ac / be) / (1 - 2j * ad / be)) if complex_z0 else np.ones_like(f)
        g = 1j * be * np.sqrt((1 - 2j * ac / be) * (1 - 2j * ad / be))
        return g, zf, vr

    gamma, zfac, _ = cell_gamma(z_ref)
    g_max = 10 ** (gain_max_db / 20)
    G = prep.s.copy()
    x_centres = np.empty(n_cells)
    x_pos = 0.0
    z = np.empty(n_cells)
    rho = np.empty(n_cells)
    resid = np.empty(n_cells)
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(8 * f.size)))
    reg = np.ones(f.size)          # accumulated regularisation filter
    t_read_k = t_read

    def read(spec: np.ndarray) -> float:
        pk = PreparedSpectrum(f, spec, prep.df, prep.k_first, spec[0], prep.dc_method)
        sr = step_response(pk, window, beta, atten_db, t_rise=t_rise, n_fft=n_fft)
        return float(np.interp(t_read_k, sr.t, sr.step))

    def rise_of_filter(F: np.ndarray) -> float:
        """10-90 % rise time of the step response of the total filter F (window x rise x reg)."""
        h = np.fft.irfft(F, n=n_fft)
        s = np.cumsum(np.roll(h, n_fft // 2))
        s = s / s[-1]
        tt = (np.arange(n_fft) - n_fft // 2) / (n_fft * prep.df)
        return float(np.interp(0.9, s, tt) - np.interp(0.1, s, tt))

    base_filter = step_response(PreparedSpectrum(f, np.ones(f.size, complex), prep.df, prep.k_first, 1.0, prep.dc_method),
                                window, beta, atten_db, t_rise=t_rise, n_fft=n_fft).window
    from .windows import gaussian_rise_filter
    base_filter = base_filter * (gaussian_rise_filter(f, t_rise) if t_rise else 1.0)

    for k in range(n_cells):
        r = float(np.clip(read(G), -rho_clip, rho_clip))
        zk = z_ref * (1 + r) / (1 - r)
        if lossy and complex_z0:
            # the read contains the early part of the lossy cell's own distributed
            # reflection; solve  read(Gamma(Z_k zeta)) = r  for Z_k (Newton steps)
            # model = a uniform lossy remainder of impedance Z_k zeta(f), length
            # x_max - x_k, terminated in Z_ref (consistent at DC, unlike an
            # infinite line whose Z0(f) diverges)
            l_rem = max(x_max - x_pos, dx)
            for _ in range(newton_steps):
                g_k, zf_k, _ = cell_gamma(zk)
                r_model = read(_cell_two_port(zk * zf_k, z_ref, g_k, l_rem)[0])
                dr_dz = 2 * z_ref / (zk + z_ref) ** 2
                zk = float(np.clip(zk + (r - r_model) / dr_dz, 0.2 * z_ref, 5 * z_ref))
        z[k], rho[k] = zk, r
        g_k, zf_k, vr = cell_gamma(zk)
        dx_k = dx * vr                       # physical length of a cell of delay t_cell/2 at this impedance
        x_centres[k] = x_pos + dx_k / 2
        x_pos += dx_k
        s11, s21 = _cell_two_port(zk * zf_k, z_ref, g_k, dx_k)
        G = (G - s11) / (s11 * (G - s11) + s21 ** 2)
        gain = np.exp(2 * alpha * (k + 1) * dx)
        reg_new = 1.0 / (1.0 + (gain / g_max) ** 2)          # Wiener-like: total filter for distance x_k
        G = G * (reg_new / reg)                              # apply only the increment
        reg = reg_new
        # the regularisation slows the edge; read later so the step has settled
        if reg.min() < 0.5:
            t_read_k = max(t_read, rise_of_filter(base_filter * reg))
        resid[k] = float(np.sqrt(np.mean(np.abs(G) ** 2)))
    return PeelResult(x_centres, z, rho, resid, dx, t_read, t_cell, lossy, gain_max_db, t_read_final=t_read_k)
