import numpy as np
from cablecheck.synth import C0, PairSpec

from conftest import differential_2port
from zprofile.hwtdr import true_profile
from zprofile.peel import peel
from zprofile.profile import Settings, compute_profile
from zprofile.propagation import gamma_from_s21
from zprofile.spectrum import prepare
from zprofile.transform import rho_to_z, step_response


def test_propagation_from_s21(uniform_10m, f600):
    spec, sdd = uniform_10m
    prop = gamma_from_s21(f600, sdd[:, 1, 0], 10.0)
    assert abs(prop.nvp - spec.nvp) < 0.01
    assert prop.fit_rms_np_per_m < 1e-4 and prop.a > 0 and prop.b >= 0
    # DC term equals the loop resistance attenuation R'/(2 Zc)
    r_loop_per_m = 2 * spec.r_dc_ohm_per_m * spec.proximity
    assert abs(prop.c - r_loop_per_m / (2 * spec.z_diff)) < 2e-4


def test_naive_profile_of_matched_line_starts_at_nominal(uniform_10m, f600):
    spec, sdd = uniform_10m
    prep = prepare(f600, sdd[:, 0, 0], "linear")
    sr = step_response(prep, "kaiser", 6.0, t_rise=500e-12)
    z = rho_to_z(sr.step, 100.0)
    x = sr.distance(spec.nvp * C0)
    m = (x > 0.3) & (x < 1.0)
    assert abs(np.mean(z[m]) - 100.0) < 1.0
    # and the lossy line's trace rises with distance (the effect the peeling removes)
    assert np.mean(z[(x > 8) & (x < 9)]) > np.mean(z[m]) + 1.0


def test_peeling_removes_loss_rise(uniform_10m, f600):
    spec, sdd = uniform_10m
    p = compute_profile(f600, sdd[:, 0, 0], 100.0, 10.0, sdd[:, 1, 0], Settings(), s12=sdd[:, 0, 1], s22=sdd[:, 1, 1])
    assert p.mask.apply(p.x).sum() > 0.5 * p.x.size      # the connector mask keeps most of a 10 m cable
    assert abs(p.stats["mean"] - 100.0) < 1.5
    assert p.stats["max"] - p.stats["min"] < 3.0
    mn = p.mask.apply(p.x_naive)
    assert np.mean(p.z_naive[mn]) - 100.0 > 2.0  # naive is biased, peeled is not


def test_peeling_tracks_other_impedances(f600):
    for zd in (90.0, 112.0):
        spec = PairSpec(length_m=8.0, z_diff=zd, n_segments=8)
        sdd = differential_2port(spec, f600)
        p = compute_profile(f600, sdd[:, 0, 0], 100.0, 8.0, sdd[:, 1, 0], Settings(), s12=sdd[:, 0, 1], s22=sdd[:, 1, 1])
        assert abs(p.stats["mean"] - zd) < 1.5, zd
        assert abs(p.z_reference - zd) < 0.5


def test_section_is_found():
    from zprofile.validate import StepPairSpec
    spec = StepPairSpec(length_m=8.0, n_segments=160, step_from_m=3.0, step_to_m=5.0, step_amp=-0.15)
    f = np.linspace(1e6, 3e9, 1500)
    sdd = differential_2port(spec, f)
    p = compute_profile(f, sdd[:, 0, 0], 100.0, 8.0, sdd[:, 1, 0], Settings(), s12=sdd[:, 0, 1], s22=sdd[:, 1, 1])
    z_sec = 100 / np.sqrt(0.85)
    inside = (p.x > 3.4) & (p.x < 4.6)
    outside = ((p.x > 1.0) & (p.x < 2.6)) | ((p.x > 5.6) & (p.x < 7.4))
    assert abs(np.mean(p.z[inside]) - z_sec) < 3.0
    assert abs(np.mean(p.z[outside]) - 100.0) < 1.5


def test_defect_depth_and_position():
    spec = PairSpec(length_m=5.0, defect_pos_m=2.5, defect_amp=0.3, defect_width_m=0.06, n_segments=250)
    f = np.linspace(1e6, 6e9, 3000)
    sdd = differential_2port(spec, f)
    xt, zt = true_profile(spec)
    p = compute_profile(f, sdd[:, 0, 0], 100.0, 5.0, sdd[:, 1, 0], Settings(), s12=sdd[:, 0, 1], s22=sdd[:, 1, 1])
    d = p.features["defects"]
    assert d, "defect not detected"
    main = min(d, key=lambda q: q["z_ohm"])
    assert abs(main["x_m"] - 2.5) < 0.15
    assert abs(main["z_ohm"] - zt.min()) < 3.5  # peeling over-reads a resolved dip by up to ~3 ohm


def test_lossless_option_and_regularisation_run(uniform_10m, f600):
    spec, sdd = uniform_10m
    prep = prepare(f600, sdd[:, 0, 0], "linear")
    prop = gamma_from_s21(f600, sdd[:, 1, 0], 10.0)
    a = peel(prep, prop, 100.0, 11.0, lossy=False)
    b = peel(prep, prop, 100.0, 11.0, gain_max_db=6.0)
    assert a.z.size == b.z.size and np.all(np.isfinite(a.z)) and np.all(np.isfinite(b.z))
    assert b.t_read_final >= b.t_read
