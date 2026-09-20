"""
A simulated time-domain-reflectometry *instrument* and the ground truth.

No real TDR head is attached to this project, so the "hardware" reference
is simulated from the same multiconductor-transmission-line model that
generates the reference cables (cablecheck.synth), but in a way that
mimics what the instrument does rather than what the VNA method does:

* the cable model is evaluated on a very wide band (default 0.005-20 GHz,
  4000 points, i.e. 5 MHz step = 200 ns unambiguous range),
* the instrument's own step (Gaussian edge, 10-90 % rise time t_src,
  default 35 ps) is applied as the excitation,
* the reflected voltage is sampled at the instrument's time step,
  white noise of rms ``noise_rho`` is added, and the trace is *normalised*
  to the specification rise time with a digital Gaussian filter, exactly
  as the "normalize" function of a sampling oscilloscope does,
* the displayed impedance is Z_ref (1 + rho)/(1 - rho).

The DC level is *known* to the instrument (a real TDR sees DC), so this
path has no DC-extrapolation step - which is one of the points of the
comparison.

``true_profile`` returns the geometric odd-mode impedance of the model,
Z(z) = Z_d / sqrt(1 + delta(z)), which is what every method is judged
against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cablecheck.mixedmode import PortMap, to_mixed_mode
from cablecheck.synth import PairSpec, make_pair

from .transform import rho_to_z
from .windows import gaussian_rise_filter

__all__ = ["TDRTrace", "true_profile", "wideband_s11", "simulate_tdr_instrument"]


@dataclass
class TDRTrace:
    t: np.ndarray
    rho: np.ndarray
    z: np.ndarray
    x: np.ndarray
    t_src: float
    t_norm: float | None
    velocity: float
    f: np.ndarray            # spectrum the trace was built from (for reuse by the peeling)
    s11: np.ndarray


def true_profile(spec: PairSpec) -> tuple[np.ndarray, np.ndarray]:
    seg_len, delta = spec.segments()
    x = np.cumsum(seg_len) - seg_len / 2
    return x, spec.z_diff / np.sqrt(1 + delta)


def wideband_s11(spec: PairSpec, f_max: float = 20e9, n_points: int = 4000):
    f = np.linspace(f_max / n_points, f_max, n_points)
    net = make_pair(spec, f)
    mm = to_mixed_mode(net, PortMap.single_pair())
    return f, mm.param(("d", "A", "near"), ("d", "A", "near")), mm.param(("d", "A", "far"), ("d", "A", "near"))


def simulate_tdr_instrument(spec: PairSpec, t_src: float = 35e-12, t_norm: float | None = 500e-12,
                            f_max: float = 20e9, n_points: int = 4000, noise_rho: float = 1e-3,
                            seed: int = 7, z_ref: float = 100.0, velocity: float | None = None) -> TDRTrace:
    f, s11, s21 = wideband_s11(spec, f_max, n_points)
    df = f[1] - f[0]
    # exact DC value from the model: loop resistance + matched far end
    r_loop = 2 * spec.r_dc_ohm_per_m * spec.proximity * spec.length_m
    s_dc = (r_loop + z_ref - z_ref) / (r_loop + z_ref + z_ref)
    fu = np.concatenate([[0.0], f])
    su = np.concatenate([[s_dc], s11])
    # excitation: Gaussian edge with rise time t_src
    x = su * gaussian_rise_filter(fu, t_src)
    n_fft = 1 << int(np.ceil(np.log2(8 * fu.size)))
    h = np.fft.irfft(x, n=n_fft)
    t = np.arange(n_fft) / (n_fft * df)
    rho = np.cumsum(h)
    rng = np.random.default_rng(seed)
    rho = rho + noise_rho * rng.standard_normal(rho.size)
    if t_norm is not None and t_norm > t_src:
        # normalise the displayed trace to the specification edge
        t_add = np.sqrt(t_norm ** 2 - t_src ** 2)
        H = gaussian_rise_filter(np.fft.rfftfreq(n_fft, t[1] - t[0]), t_add)
        rho = np.fft.irfft(np.fft.rfft(rho) * H, n=n_fft)
    if velocity is None:
        # phase velocity from the model's own S21
        ph = np.unwrap(np.angle(s21))
        velocity = float(-2 * np.pi * (f[-1] - f[-200]) / (ph[-1] - ph[-200]) * spec.length_m)
    xx = velocity * t / 2
    return TDRTrace(t, rho, rho_to_z(rho, z_ref), xx, t_src, t_norm, velocity, fu, su)
