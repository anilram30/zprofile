"""
Connector masking: decide which part of the profile is *cable*.

The launch (VNA cable, adapter, test-fixture connector, the first few
centimetres of un-twisted wire) produces the largest features of any
profile and none of them belong to the cable.  Three mechanisms, applied
in this order:

1. If fixture files exist, de-embed them first (cablecheck.deembed); the
   remaining launch feature is then only the connector itself.
2. A fixed minimum mask ``x_min`` (default 0.5 m) at the near end and
   ``x_end_margin`` (default 0.5 m) before the far end, which is located
   from the stated length or from the measured delay.
3. Automatic settling detection: starting at ``x_min``, the near-end mask
   is extended until the naive profile stays within ``tol`` ohm of its
   local median for a run of ``settle_cells`` resolution cells, so that a
   connector that rings longer than expected does not leak into the
   statistics.  The detected start is reported so the operator can see it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Mask", "find_mask"]


@dataclass
class Mask:
    x_start: float
    x_end: float
    x_start_auto: float      # what the settling detector found (>= x_min)
    reason: str

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x >= self.x_start) & (x <= self.x_end)


def find_mask(x: np.ndarray, z: np.ndarray, length_m: float | None, resolution_m: float,
              x_min: float = 0.5, x_end_margin: float = 0.5, tol: float = 3.0,
              settle_cells: float = 3.0, auto: bool = True) -> Mask:
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    x_end = (length_m - x_end_margin) if length_m else float(x[-1])
    start = x_min
    reason = "fixed minimum"
    if auto:
        run = settle_cells * resolution_m
        # local median over one run length, evaluated on the naive profile
        i0 = int(np.searchsorted(x, x_min))
        found = None
        for i in range(i0, x.size):
            m = (x >= x[i]) & (x <= x[i] + run)
            if not np.any(m) or x[i] + run > x_end:
                break
            seg = z[m]
            if np.all(np.abs(seg - np.median(seg)) <= tol):
                found = float(x[i])
                break
        if found is not None and found > x_min:
            start = found
            reason = f"settling detector ({tol:g} ohm over {run:.2f} m)"
        elif found is None:
            reason = "settling detector found no quiet run; fixed minimum used"
    x_start_auto = start
    return Mask(start, x_end, x_start_auto, reason)
