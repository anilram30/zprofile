import numpy as np
import pytest
from cablecheck.mixedmode import PortMap, to_mixed_mode
from cablecheck.synth import PairSpec, make_pair


def differential_2port(spec: PairSpec, f: np.ndarray):
    net = make_pair(spec, f)
    mm = to_mixed_mode(net, PortMap.single_pair())
    i, j = mm.index("d", "A", "near"), mm.index("d", "A", "far")
    return mm.s[:, [i, j]][:, :, [i, j]]


@pytest.fixture(scope="session")
def f600():
    return np.linspace(1e6, 600e6, 600)


@pytest.fixture(scope="session")
def uniform_10m(f600):
    spec = PairSpec(length_m=10.0, n_segments=10)
    return spec, differential_2port(spec, f600)
