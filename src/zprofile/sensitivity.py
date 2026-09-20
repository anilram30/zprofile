"""
Sensitivity budget: how much does each processing choice move the reading?

For a reference cable and a sweep, the profile is computed with the default
settings and with one choice changed at a time; the table reports the
shift of the window mean, the minimum, the maximum, the ripple amplitude and
the defect reading (the minimum inside the defect window) for the *peeled*
profile and, where it applies, the plain transform.  The point is not to
find the "best" setting but to give the number an operator needs when two
labs disagree by 1.5 ohm: "which of your settings differ, and how much can
that explain".
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from cablecheck.mixedmode import PortMap, to_mixed_mode
from cablecheck.synth import PairSpec, make_pair

from .hwtdr import true_profile
from .profile import Settings, compute_profile

__all__ = ["VARIATIONS", "run_sensitivity", "SensitivityRow"]

# (label, group, dict of Settings overrides)
VARIATIONS = [
    ("DC: zero", "dc", {"dc_method": "zero"}),
    ("DC: linear", "dc", {"dc_method": "linear"}),
    ("DC: rational", "dc", {"dc_method": "rational"}),
    ("window: rectangular", "window", {"window": "rectangular"}),
    ("window: Hann", "window", {"window": "hann"}),
    ("window: Kaiser b=3", "window", {"window": "kaiser", "beta": 3.0}),
    ("window: Kaiser b=9", "window", {"window": "kaiser", "beta": 9.0}),
    ("window: Blackman-Harris", "window", {"window": "blackmanharris"}),
    ("rise filter: none", "rise", {"t_rise": None}),
    ("rise filter: 200 ps", "rise", {"t_rise": 200e-12}),
    ("rise filter: 1000 ps", "rise", {"t_rise": 1000e-12}),
    ("mask: 0.3 m", "mask", {"x_min": 0.3, "auto_mask": False}),
    ("mask: 1.0 m", "mask", {"x_min": 1.0, "auto_mask": False}),
    ("mask: no settling detector", "mask", {"auto_mask": False}),
    ("peel: lossless", "peel", {"lossy": False}),
    ("peel: gain cap 10 dB", "peel", {"gain_max_db": 10.0}),
    ("peel: gain cap 30 dB", "peel", {"gain_max_db": 30.0}),
    ("peel: velocity model constant", "peel", {"velocity_model": "constant"}),
    ("peel: velocity model inductance", "peel", {"velocity_model": "inductance"}),
    ("reference: given (100 ohm)", "peel", {"reference": "given"}),
    ("no peeling (plain transform)", "peel", {"peel": False}),
]


@dataclass
class SensitivityRow:
    label: str
    group: str
    mean: float
    min: float
    max: float
    ripple_amp: float | None
    defect: float | None
    d_mean: float
    d_min: float
    d_max: float
    d_ripple: float | None
    d_defect: float | None
    resolution_cm: float


def _readings(prof, spec: PairSpec):
    m = prof.mask.apply(prof.x)
    z, x = prof.z[m], prof.x[m]
    rip = prof.features.get("ripple", {}).get("amplitude_ohm")
    d = None
    if spec.defect_pos_m is not None:
        md = (x > spec.defect_pos_m - 0.4) & (x < spec.defect_pos_m + 0.4)
        d = float(np.min(z[md])) if np.any(md) else None
    return float(np.mean(z)), float(np.min(z)), float(np.max(z)), rip, d


def run_sensitivity(spec: PairSpec, f_max: float = 600e6, n_points: int = 1200, base: Settings | None = None,
                    out_dir: str | Path | None = None, tag: str = "") -> tuple[list[SensitivityRow], str]:
    base = base or Settings()
    f = np.linspace(1e6, f_max, n_points)
    net = make_pair(spec, f)
    mm = to_mixed_mode(net, PortMap.single_pair())
    i, j = mm.index("d", "A", "near"), mm.index("d", "A", "far")
    sdd = mm.s[:, [i, j]][:, :, [i, j]]
    xt, zt = true_profile(spec)

    def run(st):
        return compute_profile(f, sdd[:, 0, 0], 100.0, spec.length_m, sdd[:, 1, 0], st, s12=sdd[:, 0, 1], s22=sdd[:, 1, 1])

    p0 = run(base)
    m0 = _readings(p0, spec)
    rows = [SensitivityRow("default", "-", *m0, 0.0, 0.0, 0.0, 0.0 if m0[3] is not None else None,
                           0.0 if m0[4] is not None else None, p0.resolution_m * 100)]
    # truth row
    mt = (xt > p0.mask.x_start) & (xt < p0.mask.x_end)
    dt = float(np.min(zt[(xt > spec.defect_pos_m - 0.4) & (xt < spec.defect_pos_m + 0.4)])) if spec.defect_pos_m else None
    rows.append(SensitivityRow("truth (model)", "-", float(np.mean(zt[mt])), float(np.min(zt[mt])), float(np.max(zt[mt])),
                               None, dt, float(np.mean(zt[mt])) - m0[0], float(np.min(zt[mt])) - m0[1],
                               float(np.max(zt[mt])) - m0[2], None, (dt - m0[4]) if (dt is not None and m0[4] is not None) else None,
                               0.0))
    for label, group, over in VARIATIONS:
        try:
            p = run(replace(base, **over))
        except Exception as e:  # pragma: no cover
            rows.append(SensitivityRow(label + f" (failed: {e})", group, *([np.nan] * 3), None, None, np.nan, np.nan, np.nan, None, None, 0.0))
            continue
        mv = _readings(p, spec)
        rows.append(SensitivityRow(label, group, *mv, mv[0] - m0[0], mv[1] - m0[1], mv[2] - m0[2],
                                   (mv[3] - m0[3]) if (mv[3] is not None and m0[3] is not None) else None,
                                   (mv[4] - m0[4]) if (mv[4] is not None and m0[4] is not None) else None,
                                   p.resolution_m * 100))
    md = rows_to_markdown(rows)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"sensitivity{tag}.md").write_text(md)
        with open(out / f"sensitivity{tag}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["label", "group", "mean", "min", "max", "ripple_amp", "defect", "d_mean", "d_min", "d_max", "d_ripple", "d_defect", "resolution_cm"])
            for r in rows:
                w.writerow([r.label, r.group, r.mean, r.min, r.max, r.ripple_amp, r.defect, r.d_mean, r.d_min, r.d_max, r.d_ripple, r.d_defect, r.resolution_cm])
        plot_sensitivity(rows, out / f"sensitivity{tag}.png")
    return rows, md


def _f(v, nd=2, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def rows_to_markdown(rows: list[SensitivityRow]) -> str:
    out = ["| choice | mean | min | max | ripple amp | defect | Δmean | Δmin | Δmax | Δripple | Δdefect | res. |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r.label} | {_f(r.mean)} | {_f(r.min)} | {_f(r.max)} | {_f(r.ripple_amp)} | {_f(r.defect, 1)} | "
                   f"{_f(r.d_mean, 2, True)} | {_f(r.d_min, 2, True)} | {_f(r.d_max, 2, True)} | {_f(r.d_ripple, 2, True)} | "
                   f"{_f(r.d_defect, 1, True)} | {r.resolution_cm:.0f} cm |")
    return "\n".join(out)


def plot_sensitivity(rows: list[SensitivityRow], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rr = [r for r in rows if r.label not in ("default",) and not r.label.startswith("truth")]
    labels = [r.label for r in rr]
    y = np.arange(len(rr))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(rr) + 1.5), sharey=True)
    for ax, key, title in zip(axes, ("d_mean", "d_defect"), ("shift of the window mean / ohm", "shift of the defect reading / ohm")):
        vals = np.array([getattr(r, key) if getattr(r, key) is not None else 0.0 for r in rr], float)
        cols = ["#2a78d6" if v >= 0 else "#eb6834" for v in vals]
        ax.barh(y, vals, color=cols, height=0.6)
        ax.axvline(0, color="#0b0b0b", lw=1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, axis="x", color="#e6e6e3")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
