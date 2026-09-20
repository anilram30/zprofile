"""
Band-edge windows and their consequences.

A one-sided spectrum measured up to f_max is truncated; the inverse
transform of the truncation is a sinc-shaped impulse response whose side
lobes ring for many resolution cells after every discontinuity and whose
main lobe sets the resolution.  A window w(f) tapering to zero at f_max
trades main-lobe width (resolution, feature smearing) against side-lobe
level (ringing, false features).  The catalogue here covers the windows
actually offered by VNA time-domain options plus the Kaiser family, and
``metrics`` computes, numerically rather than from textbook tables, what
each one does to a step in *this* measurement's bandwidth:

* ``rise_10_90``     10-90 % rise time of the step response of the window alone
* ``sidelobe_db``    peak level of the impulse response outside the main lobe
* ``overshoot_pct``  overshoot of the step response (ringing seen on a TDR trace)
* ``enbw``           equivalent noise bandwidth relative to rectangular

The effective 10-90 % rise time of a step seen through a window *and* a
Gaussian rise-time filter of rise time t_r is computed exactly by
``effective_rise_time`` (from the step response of the total filter).  The
popular root-sum-square rule  sqrt(t_window^2 + t_r^2)  underestimates it
by up to 25 % for a Kaiser edge (whose step is not Gaussian), which is why
it is not used anywhere in the package.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import windows as sw

__all__ = ["one_sided_window", "WINDOWS", "WindowMetrics", "metrics", "effective_rise_time", "gaussian_rise_filter"]

WINDOWS = ("rectangular", "hann", "hamming", "blackman", "blackmanharris", "kaiser", "chebyshev")


def one_sided_window(K: int, kind: str = "kaiser", beta: float = 6.0, atten_db: float = 60.0) -> np.ndarray:
    """Right half (k = 0..K-1, w[0] = 1) of a symmetric window of length 2K-1."""
    kind = kind.lower()
    n = 2 * K - 1
    if kind == "rectangular":
        w = np.ones(n)
    elif kind == "hann":
        w = sw.hann(n, sym=True)
    elif kind == "hamming":
        w = sw.hamming(n, sym=True)
    elif kind == "blackman":
        w = sw.blackman(n, sym=True)
    elif kind == "blackmanharris":
        w = sw.blackmanharris(n, sym=True)
    elif kind == "kaiser":
        w = sw.kaiser(n, beta, sym=True)
    elif kind == "chebyshev":
        w = sw.chebwin(n, atten_db, sym=True)
    else:
        raise ValueError(f"unknown window {kind!r}; choose from {WINDOWS}")
    w = w[K - 1:]
    return w / w[0]


def gaussian_rise_filter(f: np.ndarray, t_rise: float) -> np.ndarray:
    """|H(f)| = exp(-ln2 (f/f3)^2), f3 = 0.339/t_rise -> 10-90 % step rise time t_rise."""
    return np.exp(-np.log(2.0) * (f / (0.339 / t_rise)) ** 2)


@dataclass
class WindowMetrics:
    kind: str
    param: float | None
    rise_10_90: float       # s
    sidelobe_db: float
    overshoot_pct: float
    enbw: float

    def row(self):
        return (self.kind, "" if self.param is None else f"{self.param:g}", f"{self.rise_10_90 * 1e12:.0f} ps",
                f"{self.sidelobe_db:.1f} dB", f"{self.overshoot_pct:.1f} %", f"{self.enbw:.2f}")


def _step_of(spectrum: np.ndarray, df: float, n_fft: int):
    h = np.fft.irfft(spectrum, n=n_fft)
    t = np.arange(n_fft) / (n_fft * df)
    return t, h, np.cumsum(h)


def metrics(kind: str, f_max: float, beta: float = 6.0, atten_db: float = 60.0, K: int = 2001,
            t_rise: float | None = None) -> WindowMetrics:
    """Numerical step/impulse metrics of a window at bandwidth f_max (Hz)."""
    df = f_max / (K - 1)
    f = np.arange(K) * df
    w = one_sided_window(K, kind, beta, atten_db)
    if t_rise is not None:
        w = w * gaussian_rise_filter(f, t_rise)
    n_fft = 1 << int(np.ceil(np.log2(64 * K)))
    t, h, s = _step_of(w, df, n_fft)
    s = s / s[n_fft // 2]  # final value (well before wrap-around)
    # rise time: step from -inf... impulse is centred at t=0 with a symmetric tail
    # wrapping to the end; use the signed time axis
    period = 1.0 / df
    ts = np.where(t > period / 2, t - period, t)
    order = np.argsort(ts)
    ts, ss, hh = ts[order], np.cumsum(h[order]), h[order]
    ss = ss / ss[-1]
    t10 = np.interp(0.1, ss, ts)
    t90 = np.interp(0.9, ss, ts)
    overshoot = 100 * (ss.max() - 1.0)
    # side lobes: impulse response beyond the first zero crossing after the peak
    ha = np.abs(hh) / np.abs(hh).max()
    ip = int(np.argmax(ha))
    k = ip
    while k + 1 < ha.size and ha[k + 1] <= ha[k]:
        k += 1
    sidelobe = 20 * np.log10(max(ha[k:].max(), 1e-12)) if k + 1 < ha.size else -np.inf
    enbw = float(np.sum(w ** 2) / (np.sum(w) ** 2) * (2 * K - 1))
    param = beta if kind == "kaiser" else (atten_db if kind == "chebyshev" else None)
    return WindowMetrics(kind, param, float(t90 - t10), float(sidelobe), float(overshoot), enbw)


def effective_rise_time(kind: str, f_max: float, t_rise: float | None, beta: float = 6.0, atten_db: float = 60.0) -> float:
    """Exact 10-90 % rise time of the window (and optional Gaussian filter) at this bandwidth."""
    return metrics(kind, f_max, beta, atten_db, t_rise=t_rise).rise_10_90
