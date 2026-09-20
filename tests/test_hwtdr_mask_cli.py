import json

import numpy as np
from cablecheck.synth import PairSpec

from zprofile.cli import main
from zprofile.hwtdr import simulate_tdr_instrument, true_profile
from zprofile.mask import find_mask


def test_tdr_instrument_matches_truth_level_and_rises():
    spec = PairSpec(length_m=6.0, n_segments=6)
    tr = simulate_tdr_instrument(spec, f_max=8e9, n_points=1600)
    xt, zt = true_profile(spec)
    near = (tr.x > 0.2) & (tr.x < 0.6)
    assert abs(np.mean(tr.z[near]) - 100.0) < 0.8
    far = (tr.x > 5.0) & (tr.x < 5.5)
    assert np.mean(tr.z[far]) > np.mean(tr.z[near]) + 0.5  # lossy-line rise is real on the instrument too


def test_mask_detects_ringing_launch():
    x = np.linspace(0, 10, 1001)
    z = 100 + 40 * np.exp(-x / 0.4) * np.cos(40 * x)
    m = find_mask(x, z, 10.0, resolution_m=0.1, x_min=0.2)
    assert 0.7 < m.x_start < 1.6 and m.x_end == 9.5 and "settling" in m.reason
    m2 = find_mask(x, z, 10.0, resolution_m=0.1, x_min=0.2, auto=False)
    assert m2.x_start == 0.2


def test_cli_profile_and_windows(tmp_path, capsys):
    ref = "/home/claude/cablecheck/tests/reference"
    import os
    if not os.path.exists(ref + "/good_15m_truth.s4p"):
        import pytest
        pytest.skip("cablecheck reference data not available")
    rc = main(["profile", ref + "/good_15m_truth.s4p", "--length", "15", "--out", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and abs(out["stats"]["mean"] - 100.0) < 2.0
    assert (tmp_path / "good_15m_truth_pairA.profile.csv").exists() and (tmp_path / "good_15m_truth_pairA.png").exists()
    rc = main(["windows", "--fmax", "600e6"])
    assert rc == 0 and "kaiser" in capsys.readouterr().out
