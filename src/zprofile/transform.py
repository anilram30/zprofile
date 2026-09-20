"""
Frequency -> time: the step response and the plain ("naive") impedance profile.

Given the prepared one-sided spectrum S_k = S(k df), k = 0..K-1, the window
w_k and the rise-time filter H_k, the impulse response is the inverse real
DFT with zero padding to N points,

    h[n] = (1/N) sum_k (S_k w_k H_k) e^{+j 2 pi k n / N}   (Hermitian completion),

sampled at t_n = n / (N df); the step response is the running sum
r[n] = sum_{m <= n} h[m] started a few resolution cells *before* t = 0 in
signed time (the half of an edge at the reference plane that falls at
negative time wraps to the end of the DFT array and must be counted), and
converges to S(0).  The reflection coefficient
maps to impedance as  Z = Z_ref (1 + r)/(1 - r)  and round-trip time to
one-way distance as  x = v t / 2.

This is what every VNA "time-domain option" does.  It is exact for a
single discontinuity in a lossless line and it is wrong, in a way that grows
with distance, for everything else: transmission through earlier
discontinuities (the incident step is smaller than one), re-reflections
between discontinuities, and the distributed reflection of a lossy line
(the sqrt(t) rise).  The layer-peeling algorithm in ``peel`` removes those
three effects; the plain transform is kept because it is the comparison
baseline and because it is what the hardware TDR instrument displays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectrum import PreparedSpectrum
from .windows import gaussian_rise_filter, one_sided_window

C0 = 299_792_458.0

__all__ = ["StepResponse", "step_response", "rho_to_z", "C0"]


@dataclass
class StepResponse:
    t: np.ndarray
    impulse: np.ndarray
    step: np.ndarray
    df: float
    n_fft: int
    window: np.ndarray

    def signed_time(self) -> np.ndarray:
        period = 1.0 / self.df
        return np.where(self.t > period / 2, self.t - period, self.t)

    def distance(self, velocity: float) -> np.ndarray:
        return velocity * self.t / 2.0


def step_response(prep: PreparedSpectrum, window: str = "kaiser", beta: float = 6.0, atten_db: float = 60.0,
                  t_rise: float | None = 500e-12, n_fft: int | None = None, pad_factor: int = 8,
                  t_pre: float | None = None) -> StepResponse:
    K = prep.f.size
    w = one_sided_window(K, window, beta, atten_db)
    x = prep.s * w
    if t_rise is not None:
        x = x * gaussian_rise_filter(prep.f, t_rise)
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(pad_factor * K)))
    h = np.fft.irfft(x, n=n_fft)
    t = np.arange(n_fft) / (n_fft * prep.df)
    # The windowed impulse response is symmetric about t = 0, so half of an
    # edge that sits at the reference plane lives at *negative* time, which the
    # DFT wraps to the end of the array.  The step at t >= 0 therefore starts
    # the running sum a few resolution cells before t = 0 (signed time), which
    # captures that half but not the aliased far tail of a slow response.
    if t_pre is None:
        t_pre = 4.0 / prep.f[-1]
    n_pre = min(int(round(t_pre * n_fft * prep.df)), n_fft // 4)
    pre = h[n_fft - n_pre:].sum() if n_pre > 0 else 0.0
    step = np.cumsum(h) + pre
    return StepResponse(t, h, step, prep.df, n_fft, w)


def rho_to_z(rho: np.ndarray, z_ref: float, rho_max: float = 0.999) -> np.ndarray:
    r = np.clip(rho, -rho_max, rho_max)
    return z_ref * (1 + r) / (1 - r)
