"""
Validation of the frequency-domain profile against the simulated TDR
instrument and against the model truth, on a set of reference cables.

For every cable three views are compared on the same distance axis:

* ``truth``      the geometric impedance of the model,
* ``tdr``        the simulated instrument trace (35 ps source edge),
                 normalised to a chosen rise time,
* ``fd-naive``   the plain transform of a VNA sweep,
* ``fd-peeled``  the loss-aware layer-peeled profile of the same sweep,
* ``tdr-peeled`` the peeling algorithm applied to the *instrument's* data
                 (shows the algorithm is instrument-agnostic).

Two questions are answered separately.  *Does the transform reproduce the
instrument?* - compare fd-naive with the instrument normalised to the same
effective rise time; only DC extrapolation and window shape differ.  *Does
the algorithm reproduce the cable?* - compare fd-peeled with truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from cablecheck.mixedmode import PortMap, to_mixed_mode
from cablecheck.synth import PairSpec, make_pair

from .hwtdr import simulate_tdr_instrument, true_profile
from .peel import peel
from .profile import Settings, compute_profile
from .propagation import gamma_from_s21
from .spectrum import prepare

__all__ = ["StepPairSpec", "REFERENCE_CABLES", "validate_cable", "run_validation", "ValidationRow"]


@dataclass
class StepPairSpec(PairSpec):
    """PairSpec with an impedance *section* (step up and back) of relative
    capacitance change ``step_amp`` between ``step_from_m`` and ``step_to_m``."""
    step_from_m: float = 4.0
    step_to_m: float = 7.0
    step_amp: float = -0.12

    def segments(self):
        seg_len, delta = super().segments()
        edges = np.cumsum(seg_len)
        z = edges - seg_len / 2
        delta = delta + np.where((z >= self.step_from_m) & (z <= self.step_to_m), self.step_amp, 0.0)
        return seg_len, delta


REFERENCE_CABLES = {
    "uniform_15m": PairSpec(length_m=15.0, d_wire_m=0.55e-3, tan_delta=0.0012, n_segments=30),
    "thin_lossy_15m": PairSpec(length_m=15.0, d_wire_m=0.28e-3, tan_delta=0.006, proximity=1.35, n_segments=30),
    "section_10m": StepPairSpec(length_m=10.0, d_wire_m=0.5e-3, tan_delta=0.0015, n_segments=200,
                                step_from_m=4.0, step_to_m=7.0, step_amp=-0.12),
    "ripple_defect_15m": PairSpec(length_m=15.0, d_wire_m=0.55e-3, tan_delta=0.0012, ripple_amp=0.025,
                                  ripple_period_m=0.26, defect_pos_m=6.0, defect_amp=0.25, defect_width_m=0.08,
                                  n_segments=240, seed=13),
    "short_defect_5m": PairSpec(length_m=5.0, d_wire_m=0.5e-3, tan_delta=0.0015, defect_pos_m=2.5,
                                defect_amp=0.35, defect_width_m=0.03, n_segments=250),
}


@dataclass
class ValidationRow:
    cable: str
    sweep: str
    method: str
    rms_vs_truth: float
    bias_vs_truth: float
    max_abs_vs_truth: float
    rms_vs_tdr: float | None
    feature_true: float | None
    feature_read: float | None
    resolution_m: float
    extra: dict = field(default_factory=dict)


def _feature(spec: PairSpec, x: np.ndarray, z: np.ndarray) -> float | None:
    """The cable's characteristic feature: minimum inside the defect/section, else None."""
    if isinstance(spec, StepPairSpec):
        m = (x > spec.step_from_m + 0.3) & (x < spec.step_to_m - 0.3)
        return float(np.mean(z[m])) if np.any(m) else None
    if spec.defect_pos_m is not None:
        m = (x > spec.defect_pos_m - 0.4) & (x < spec.defect_pos_m + 0.4)
        return float(np.min(z[m])) if np.any(m) else None
    return None


def _cmp(x, z, xt, zt, x0, x1):
    m = (x >= x0) & (x <= x1)
    zi = np.interp(x[m], xt, zt)
    e = z[m] - zi
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(e)), float(np.max(np.abs(e)))


def validate_cable(name: str, spec: PairSpec, sweeps=((600e6, 1200), (6e9, 3000)), settings: Settings | None = None,
                   tdr_kwargs: dict | None = None, f_peel_max: float = 6e9) -> tuple[list[ValidationRow], dict]:
    st = settings or Settings()
    xt, zt = true_profile(spec)
    x0, x1 = 0.5, spec.length_m - 0.5
    tdr = simulate_tdr_instrument(spec, t_norm=500e-12, **(tdr_kwargs or {}))
    rows = []
    curves = {"truth": (xt, zt), "tdr_500ps": (tdr.x, tdr.z)}
    ft = _feature(spec, xt, zt)
    for f_max, n in sweeps:
        f = np.linspace(1e6, f_max, n)
        net = make_pair(spec, f)
        mm = to_mixed_mode(net, PortMap.single_pair())
        s11 = mm.param(("d", "A", "near"), ("d", "A", "near"))
        s21 = mm.param(("d", "A", "far"), ("d", "A", "near"))
        s12 = mm.param(("d", "A", "near"), ("d", "A", "far"))
        s22 = mm.param(("d", "A", "far"), ("d", "A", "far"))
        prof = compute_profile(f, s11, 100.0, spec.length_m, s21, st, s12=s12, s22=s22)
        tag = f"{f_max / 1e9:g}GHz" if f_max >= 1e9 else f"{f_max / 1e6:g}MHz"
        # instrument normalised to the same effective rise time as this sweep
        tdr_eq = simulate_tdr_instrument(spec, t_norm=prof.t_eff, velocity=prof.velocity, **(tdr_kwargs or {}))
        for method, (x, z) in (("fd-naive", (prof.x_naive, prof.z_naive)), ("fd-peeled", (prof.x, prof.z))):
            rms, bias, mx = _cmp(x, z, xt, zt, x0, x1)
            rms_tdr = _cmp(x, z, tdr_eq.x, tdr_eq.z, x0, x1)[0] if method == "fd-naive" else None
            rows.append(ValidationRow(name, tag, method, rms, bias, mx, rms_tdr, ft, _feature(spec, x, z),
                                      prof.resolution_m, {"t_eff_ps": prof.t_eff * 1e12}))
            curves[f"{method}_{tag}"] = (x, z)
        curves[f"tdr_eq_{tag}"] = (tdr_eq.x, tdr_eq.z)
    # peeling applied to the instrument's own data.  The instrument spectrum is
    # band-limited to ``f_peel_max`` first: above a few GHz the round-trip
    # attenuation of a thin automotive pair exceeds any sensible gain cap within
    # the first metres, so those frequencies carry no usable information.
    from .hwtdr import wideband_s11
    _, _, s21w = wideband_s11(spec)
    keep = tdr.f <= f_peel_max
    prep = prepare(tdr.f[keep], tdr.s11[keep], "linear")
    f_i = tdr.f[1:]
    prop = gamma_from_s21(f_i[f_i <= f_peel_max], s21w[f_i <= f_peel_max], spec.length_m)
    pr = peel(prep, prop, 100.0, spec.length_m * 1.1 + 0.5, window="kaiser", beta=6.0, t_rise=500e-12)
    rms, bias, mx = _cmp(pr.x, pr.z, xt, zt, x0, x1)
    rows.append(ValidationRow(name, f"TDR (<= {f_peel_max / 1e9:g} GHz)", "tdr-peeled", rms, bias, mx, None, ft, _feature(spec, pr.x, pr.z), pr.dx,
                              {"t_eff_ps": pr.t_read * 1e12}))
    rms, bias, mx = _cmp(tdr.x, tdr.z, xt, zt, x0, x1)
    rows.append(ValidationRow(name, "TDR 20GHz", "tdr-naive-500ps", rms, bias, mx, None, ft, _feature(spec, tdr.x, tdr.z),
                              tdr.velocity * 500e-12 / 2))
    curves["tdr_peeled"] = (pr.x, pr.z)
    return rows, curves


def rows_to_markdown(rows: list[ValidationRow]) -> str:
    out = ["| cable | data | method | rms vs truth | bias | max abs | rms vs TDR (same t_r) | feature true / read | resolution |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ft = "–" if r.feature_true is None else f"{r.feature_true:.1f} / {r.feature_read:.1f} Ω"
        rt = "–" if r.rms_vs_tdr is None else f"{r.rms_vs_tdr:.2f} Ω"
        out.append(f"| {r.cable} | {r.sweep} | {r.method} | {r.rms_vs_truth:.2f} Ω | {r.bias_vs_truth:+.2f} Ω | "
                   f"{r.max_abs_vs_truth:.2f} Ω | {rt} | {ft} | {r.resolution_m * 100:.0f} cm |")
    return "\n".join(out)


def plot_curves(name: str, spec: PairSpec, curves: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = sorted({k[len("fd-naive_"):] for k in curves if k.startswith("fd-naive_")}, key=lambda s: float(s.replace("GHz", "e9").replace("MHz", "e6")))
    fig, axes = plt.subplots(len(tags) + 1, 1, figsize=(9, 3.0 * (len(tags) + 1)), sharex=True)
    cols = {"truth": "#0b0b0b", "tdr": "#eb6834", "naive": "#2a78d6", "peeled": "#008300"}
    xt, zt = curves["truth"]
    for ax, tag in zip(axes, tags):
        ax.plot(xt, zt, color=cols["truth"], lw=1.2, label="truth (model)")
        x, z = curves[f"tdr_eq_{tag}"]
        ax.plot(x, z, color=cols["tdr"], lw=1.0, label="TDR instrument, same rise time")
        x, z = curves[f"fd-naive_{tag}"]
        ax.plot(x, z, color=cols["naive"], lw=1.2, ls="--", label=f"FD naive, {tag} sweep")
        x, z = curves[f"fd-peeled_{tag}"]
        ax.step(x, z, color=cols["peeled"], lw=1.4, where="mid", label=f"FD peeled, {tag} sweep")
        ax.set_ylabel("ohm")
        ax.set_title(f"{name}: sweep to {tag}", loc="left", fontsize=10)
        ax.grid(True, color="#e6e6e3")
        ax.legend(fontsize=8, frameon=False, ncol=2)
    ax = axes[-1]
    ax.plot(xt, zt, color=cols["truth"], lw=1.2, label="truth (model)")
    x, z = curves["tdr_500ps"]
    ax.plot(x, z, color=cols["tdr"], lw=1.0, label="TDR instrument, 500 ps normalised")
    x, z = curves["tdr_peeled"]
    ax.step(x, z, color=cols["peeled"], lw=1.4, where="mid", label="TDR data, peeled")
    ax.set_title(f"{name}: instrument path", loc="left", fontsize=10)
    ax.set_xlabel("distance / m")
    ax.set_ylabel("ohm")
    ax.grid(True, color="#e6e6e3")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_xlim(0, spec.length_m + 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_validation(out_dir: str | Path, cables: dict | None = None, sweeps=((600e6, 1200), (6e9, 3000))):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cables = cables or REFERENCE_CABLES
    all_rows = []
    for name, spec in cables.items():
        rows, curves = validate_cable(name, spec, sweeps)
        all_rows.extend(rows)
        plot_curves(name, spec, curves, out / f"validation_{name}.png")
    md = rows_to_markdown(all_rows)
    (out / "validation_table.md").write_text(md)
    import csv
    with open(out / "validation_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cable", "data", "method", "rms_vs_truth", "bias", "max_abs", "rms_vs_tdr", "feature_true", "feature_read", "resolution_m"])
        for r in all_rows:
            w.writerow([r.cable, r.sweep, r.method, r.rms_vs_truth, r.bias_vs_truth, r.max_abs_vs_truth, r.rms_vs_tdr,
                        r.feature_true, r.feature_read, r.resolution_m])
    return all_rows, md
