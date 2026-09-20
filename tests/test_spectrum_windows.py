import numpy as np
import pytest

from zprofile.spectrum import prepare
from zprofile.windows import WINDOWS, effective_rise_time, metrics, one_sided_window


def test_prepare_grid_starts_at_dc_and_keeps_data():
    f = np.linspace(2e6, 100e6, 50)
    s = 0.1 * np.exp(-1j * 2 * np.pi * f * 5e-9)
    for m in ("zero", "linear", "rational"):
        p = prepare(f, s, m)
        assert p.f[0] == 0.0 and np.isclose(p.df, 2e6)
        assert np.allclose(p.s[p.k_first:], s)
        assert abs(p.s_dc.imag) < 1e-12 and abs(p.s_dc) <= 1
    p = prepare(f, s, "physical", r_loop_ohm=2.0, z_ref=100.0)
    assert np.isclose(p.s_dc.real, 2.0 / 202.0)
    with pytest.raises(ValueError):
        prepare(f, s, "physical")


def test_rational_fit_recovers_first_order_model():
    f = np.linspace(1e6, 50e6, 40)
    a, b, c = 0.05, 2e-9, 1e-8
    s = (a + 1j * b * f) / (1 + 1j * c * f)
    p = prepare(f, s, "rational", n_fit=40)
    assert abs(p.s_dc.real - a) < 1e-4


def test_window_metrics_ordering():
    m = {k: metrics(k, 600e6) for k in ("rectangular", "hann", "blackmanharris")}
    assert m["rectangular"].rise_10_90 < m["hann"].rise_10_90 < m["blackmanharris"].rise_10_90
    assert m["rectangular"].sidelobe_db > m["hann"].sidelobe_db > m["blackmanharris"].sidelobe_db
    assert m["rectangular"].overshoot_pct > 5 and m["blackmanharris"].overshoot_pct < 0.1
    for k in WINDOWS:
        w = one_sided_window(101, k)
        assert w[0] == 1.0 and w.size == 101 and np.all(w >= -1e-12)


def test_effective_rise_time_bounds():
    t_w = effective_rise_time("kaiser", 3e9, None)
    t_r = 500e-12
    t_eff = effective_rise_time("kaiser", 3e9, t_r)
    # the root-sum-square rule underestimates for a Kaiser edge; the exact value lies between
    assert max(t_w, t_r) < t_eff < t_w + t_r
    assert np.hypot(t_w, t_r) < t_eff
