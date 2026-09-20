# zprofile — impedance profile reconstruction

**Reconstructs a cable's impedance profile along its physical length from frequency-domain measurements.**

[![CI](https://github.com/anilram30/zprofile/actions/workflows/ci.yml/badge.svg)](https://github.com/anilram30/zprofile/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen)](tests/)
[![Report](https://img.shields.io/badge/report-17%20pages-informational)](docs/report.pdf)

> **Part of the [HF cable toolchain](https://github.com/anilram30/hf-cable-toolchain)** — seven packages that take a high-frequency cable from a raw measurement to a predicted Ethernet link.
> 
> [A · cablecheck](https://github.com/anilram30/cablecheck)  ·  [B · labauto](https://github.com/anilram30/labauto)  ·  [C · shieldeval](https://github.com/anilram30/shieldeval)  ·  **D · zprofile**  ·  [E · cableanalytics](https://github.com/anilram30/cableanalytics)  ·  [F · labplatform](https://github.com/anilram30/labplatform)  ·  [G · linktwin](https://github.com/anilram30/linktwin)

---

## The problem it solves

A frequency-domain measurement tells you how a cable behaves; it does not tell you *where* along the cable something is wrong. `zprofile` inverts that: from the reflected signal it reconstructs the characteristic impedance metre by metre along the cable, so a manufacturing defect, a crushed section or a bad connector becomes a position in metres rather than a bump on a graph. It does this with layer peeling rather than a plain transform, which means later reflections are corrected for everything the wave passed through on the way.

## At a glance

|  |  |
|---|---|
| **Takes** | The differential reflection S_dd11(f) of a cable, and its transmission S_dd21(f) when available |
| **Produces** | Characteristic impedance against position along the cable, with ripple period, defect positions and deviation statistics |
| **Checked against** | Five synthetic reference cables with known profiles, and a simulated TDR instrument |
| **Technical report** | [`docs/report.pdf`](docs/report.pdf) — 17 pages, 16 references, every method stated with its mathematics and its limitations |
| **Tests** | 14, run against Python 3.11, 3.12 and 3.13 on every push |
| **Data** | Entirely synthetic. No proprietary or customer measurements are used anywhere in this toolchain. |

## Install

Python 3.11 or newer.

```sh
pip install "git+https://github.com/anilram30/cablecheck.git" \
            "git+https://github.com/anilram30/zprofile.git"
```

---

## What it does

**What it does.** Takes the differential reflection `S_dd11(f)` (and, when available,
`S_dd21`) of a cable and returns the impedance along its length, with every processing
choice explicit and its effect quantified:

| step | module | what is different from a VNA "time-domain option" |
|---|---|---|
| DC gap | `spectrum` | four estimators; default anchors `Γ(0)` to the loop resistance derived from the cable's own attenuation fit |
| band-edge window | `windows` | catalogue with *numerically computed* rise time, side lobe, overshoot at the actual bandwidth; exact effective rise time (no root-sum-square rule) |
| rise-time matching | `transform` | Gaussian filter giving the specification's 10–90 % rise time |
| loss model | `propagation` | `alpha = c + a√f + b f` from `S21` on the 2-port renormalised to the cable's fitted impedance; consistent complex `Z0(f)` and `γ(f)` |
| reconstruction | `peel` | loss-aware layer peeling: early-time read, Newton correction against a lossy remainder, exact load extraction per cell, Wiener-like regularisation of the inverse propagation with adaptive read time, velocity hypothesis for impedance changes |
| connector mask | `mask` | fixed masks + settling detector, fixture de-embedding upstream |
| features | `profile` | stats, ripple period/amplitude, defects (position, depth, extent) |
| validation | `hwtdr`, `validate` | simulated TDR instrument (35 ps source, known DC, normalised trace) and model truth on five reference cables |
| sensitivity | `sensitivity` | shift of mean/min/max/ripple/defect for each choice — the honest statement |

Result on the reference cables: the plain transform is biased by +4 to +9 Ω on a 15 m
automotive pair (the √t rise of a lossy line, which a real TDR instrument shows too);
the reconstruction brings that to within a few tenths of an ohm, and reads features
correctly once the bandwidth resolves them.

## Install and test

```bash
pip install -e ../cablecheck      # dependency (Project A)
pip install -e ".[dev]"
pytest                             # 14 tests, ~1 min
```

## Use

```bash
# profile of a measured cable (fixture removed first), 15 m, pair A
zprofile profile sample.s4p --length 15 --fixture fixture_launch.s4p --out out/
# → out/sample_pairA.profile.csv, _naive.csv, .json (stats, features, all settings), .png

# what does each window do at my bandwidth?
zprofile windows --fmax 600e6 --t-rise-ps 500

# reproduce the validation and the sensitivity budget
zprofile validate --out validation/
zprofile sensitivity --cable ripple_defect_15m --fmax 600e6 --out sens/
```

Python:

```python
from zprofile import compute_profile, Settings
prof = compute_profile(f, s11, z_ref=100.0, length_m=15.0, s21=s21, s12=s12, s22=s22, settings=Settings())
prof.x, prof.z            # reconstructed profile (cell centres, ohm)
prof.x_naive, prof.z_naive
prof.stats, prof.features, prof.mask, prof.resolution_m, prof.to_dict()
```

Key `Settings`: `dc_method` (auto/zero/linear/rational/physical), `window`, `beta`, `t_rise`,
`peel`, `lossy`, `gain_max_db`, `velocity_model` (capacitance/inductance/constant),
`reference` (fitted/given), `x_min`, `x_end_margin`, `auto_mask`.

## Documentation

`docs/report.md` / `docs/report.pdf`: the mathematics (why the plain transform is wrong,
the layer-peeling recursion with loss, regularisation, DC anchoring, windows), the
simulated instrument, the validation tables and figures, the sensitivity budget, and a
section on what did not work (including a step-response wrap bug that was found here and
fixed in Project A, and a model-based Gauss–Newton refinement that is kept as experimental).
---

## Contributing

Bug reports, questions about the methods, and pull requests are all welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). Numerical changes need a numerical test, and a change to a method
is also a change to `docs/report.md`.

## Licence and attribution

MIT — see [LICENSE](LICENSE). Author: Sreeram Anil.

Built with AI assistance; the commit history records it. The engineering decisions, the validation
strategy and the limitations stated in the report are the substance of the work.

Part of the **[HF cable toolchain](https://github.com/anilram30/hf-cable-toolchain)** · [Report an issue](https://github.com/anilram30/zprofile/issues) ·
[Changelog](CHANGELOG.md)
