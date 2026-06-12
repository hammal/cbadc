"""The numba-accelerated simulate_dt path must match the numpy reference exactly."""
import numpy as np
import pytest

import cbadc.analog_frontend as afmod
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import ZeroOrderHold


@pytest.mark.skipif(not afmod._HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize(
    "builder,N,E",
    [(AnalogFrontend.chain_of_integrators, 4, 12), (AnalogFrontend.leapfrog, 6, 14)],
)
def test_numba_matches_numpy(builder, N, E, monkeypatch):
    af, _ = builder(N=N, ENOB=E, BW=1e6)
    af = af if af.is_discrete_time else af.discretize(dt=af.digital_control.dt)
    size = 1 << 13
    ref = ZeroOrderHold.uniform_reference_signal(
        af.dt, -np.ones((1, 3)), np.ones((1, 3)), size=size, seed=2)
    af.analog_signal = ref

    fast = af.simulate_dt(size)
    # force the numpy reference path
    monkeypatch.setattr(afmod, "_HAS_NUMBA", False)
    slow = af.simulate_dt(size)

    for key in ("v", "x", "y"):
        assert np.array_equal(fast[key], slow[key]), f"{key} differs"
