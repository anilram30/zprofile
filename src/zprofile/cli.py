"""Command line: ``zprofile profile|validate|sensitivity|windows``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def cmd_profile(args):
    import numpy as np
    from cablecheck.deembed import deembed
    from cablecheck.io import read_touchstone
    from cablecheck.mixedmode import PortMap, to_mixed_mode

    from .profile import Settings, compute_profile

    net = read_touchstone(args.file)
    if args.fixture:
        fx = read_touchstone(args.fixture)
        net = deembed(net, fx, fx)
    pm = PortMap.parse(args.port_map) if args.port_map else (PortMap.single_pair() if net.nports == 4 else PortMap.two_pairs())
    mm = to_mixed_mode(net, pm)
    pair = args.pair
    s11 = mm.param(("d", pair, "near"), ("d", pair, "near"))
    z_ref = float(mm.z0[mm.index("d", pair, "near")])
    s21 = s12 = s22 = None
    if (pair, "far") in pm.groups():
        s21 = mm.param(("d", pair, "far"), ("d", pair, "near"))
        s12 = mm.param(("d", pair, "near"), ("d", pair, "far"))
        s22 = mm.param(("d", pair, "far"), ("d", pair, "far"))
    st = Settings(dc_method=args.dc, window=args.window, beta=args.beta, t_rise=None if args.t_rise_ps <= 0 else args.t_rise_ps * 1e-12,
                  peel=not args.no_peel, lossy=not args.lossless, gain_max_db=args.gain_cap, x_min=args.mask,
                  x_end_margin=args.end_margin, auto_mask=not args.no_auto_mask, nvp=args.nvp,
                  r_loop_ohm=args.r_loop, reference=args.reference, velocity_model=args.velocity_model)
    prof = compute_profile(net.f, s11, z_ref, args.length, s21, st, s12=s12, s22=s22)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / (Path(args.file).stem + f"_pair{pair}")
    np.savetxt(stem.with_suffix(".profile.csv"), np.column_stack([prof.x, prof.z]), delimiter=",", header="x_m,z_ohm", comments="")
    np.savetxt(stem.with_name(stem.name + "_naive").with_suffix(".csv"), np.column_stack([prof.x_naive, prof.z_naive]),
               delimiter=",", header="x_m,z_ohm", comments="")
    (stem.with_suffix(".json")).write_text(json.dumps(prof.to_dict(), indent=1, default=float))
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(prof.x_naive, prof.z_naive, color="#2a78d6", lw=1.0, ls="--", label="plain transform")
    if prof.peel_result is not None:
        ax.step(prof.x, prof.z, where="mid", color="#008300", lw=1.5, label="reconstructed (peeled)")
    ax.axvspan(0, prof.mask.x_start, color="#9a9a96", alpha=0.25, label="masked")
    if args.length:
        ax.axvspan(prof.mask.x_end, prof.x[-1], color="#9a9a96", alpha=0.25)
    ax.set_xlabel("distance / m")
    ax.set_ylabel("ohm")
    s = prof.stats
    ax.set_title(f"{Path(args.file).name} pair {pair}: mean {s.get('mean', float('nan')):.2f}, min {s.get('min', float('nan')):.2f}, "
                 f"max {s.get('max', float('nan')):.2f} ohm; resolution {prof.resolution_m * 100:.0f} cm", fontsize=9, loc="left")
    ax.grid(True, color="#e6e6e3")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=130)
    print(json.dumps({"stats": prof.stats, "features": prof.features, "mask": {"x_start": prof.mask.x_start, "x_end": prof.mask.x_end,
                      "reason": prof.mask.reason}, "resolution_m": prof.resolution_m, "z_reference": prof.z_reference,
                      "dc_method": prof.dc_method_used, "outputs": str(stem) + ".*"}, indent=1, default=float))
    return 0


def cmd_validate(args):
    from .validate import REFERENCE_CABLES, run_validation
    cables = REFERENCE_CABLES if not args.cable else {k: REFERENCE_CABLES[k] for k in args.cable}
    sweeps = tuple((float(s.split(":")[0]), int(s.split(":")[1])) for s in args.sweep) if args.sweep else ((600e6, 1200), (6e9, 3000))
    rows, md = run_validation(args.out, cables, sweeps)
    print(md)
    return 0


def cmd_sensitivity(args):
    from .sensitivity import run_sensitivity
    from .validate import REFERENCE_CABLES
    spec = REFERENCE_CABLES[args.cable]
    rows, md = run_sensitivity(spec, args.fmax, args.points, out_dir=args.out, tag=f"_{args.cable}_{args.fmax / 1e6:g}MHz")
    print(md)
    return 0


def cmd_windows(args):
    from .windows import WINDOWS, metrics
    print(f"window metrics at f_max = {args.fmax / 1e6:g} MHz" + (f", with {args.t_rise_ps:g} ps rise filter" if args.t_rise_ps > 0 else ""))
    print(f"{'window':16s} {'param':6s} {'10-90 rise':>11s} {'side lobe':>10s} {'overshoot':>10s} {'ENBW':>6s}")
    tr = None if args.t_rise_ps <= 0 else args.t_rise_ps * 1e-12
    for kind in WINDOWS:
        params = [3.0, 6.0, 9.0] if kind == "kaiser" else ([50.0, 70.0] if kind == "chebyshev" else [None])
        for p in params:
            m = metrics(kind, args.fmax, beta=p or 6.0, atten_db=p or 60.0, t_rise=tr)
            r = m.row()
            print(f"{r[0]:16s} {r[1]:6s} {r[2]:>11s} {r[3]:>10s} {r[4]:>10s} {r[5]:>6s}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="zprofile", description="Impedance profile reconstruction from S-parameters")
    p.add_argument("--version", action="version", version=f"zprofile {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("profile", help="profile of one Touchstone file")
    a.add_argument("file")
    a.add_argument("--length", type=float, help="cable length in m (enables S21-based loss model and end mask)")
    a.add_argument("--port-map"), a.add_argument("--pair", default="A")
    a.add_argument("--fixture", help="fixture Touchstone (VNA-side ports first) to de-embed at both ends")
    a.add_argument("--dc", default="auto", choices=["auto", "zero", "linear", "rational", "physical"])
    a.add_argument("--r-loop", type=float, help="DC loop resistance in ohm for --dc physical")
    a.add_argument("--window", default="kaiser"), a.add_argument("--beta", type=float, default=6.0)
    a.add_argument("--t-rise-ps", type=float, default=500.0, help="specification rise time; 0 = no filter")
    a.add_argument("--no-peel", action="store_true"), a.add_argument("--lossless", action="store_true")
    a.add_argument("--gain-cap", type=float, default=20.0)
    a.add_argument("--velocity-model", default="capacitance", choices=["capacitance", "inductance", "constant"])
    a.add_argument("--reference", default="fitted", choices=["fitted", "given"])
    a.add_argument("--mask", type=float, default=0.5), a.add_argument("--end-margin", type=float, default=0.5)
    a.add_argument("--no-auto-mask", action="store_true"), a.add_argument("--nvp", type=float)
    a.add_argument("--out", default="zprofile_out")
    a.set_defaults(func=cmd_profile)
    dm = sub.add_parser("demo", help="run the full validation against the simulated TDR instrument")
    dm.add_argument("out", nargs="?", default="demo_out")
    dm.add_argument("--cable", action="append"), dm.add_argument("--sweep", action="append")
    dm.set_defaults(func=cmd_validate)

    v = sub.add_parser("validate", help="run the validation against the simulated TDR instrument")
    v.add_argument("--out", default="validation"), v.add_argument("--cable", action="append")
    v.add_argument("--sweep", action="append", help="fmax:points, e.g. 600e6:1200 (repeatable)")
    v.set_defaults(func=cmd_validate)
    s = sub.add_parser("sensitivity", help="sensitivity budget of the processing choices")
    s.add_argument("--cable", default="ripple_defect_15m"), s.add_argument("--fmax", type=float, default=600e6)
    s.add_argument("--points", type=int, default=1200), s.add_argument("--out", default="sensitivity")
    s.set_defaults(func=cmd_sensitivity)
    w = sub.add_parser("windows", help="window metrics table")
    w.add_argument("--fmax", type=float, default=600e6), w.add_argument("--t-rise-ps", type=float, default=0.0)
    w.set_defaults(func=cmd_windows)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
