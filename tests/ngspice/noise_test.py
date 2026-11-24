"""Check state and comparator noise in a clocked loop against ngspice.

Independent random draws cannot give matching trajectories. Compare state
spread and decision switching rate after the initial transient instead.
"""

from copy import deepcopy

import numpy as np
import pytest

from cbadc.circuit import GmC
from cbadc.frontend import Frontend

from . import spice


@pytest.mark.parametrize("kind", ["state", "comparator"])
def test_noise_through_feedback(kind):
    """One noisy channel must give similar closed-loop statistics in both solvers.

    State noise enters ``dx/dt``; comparator noise enters ``y = Cx + eta``.
    Compare state spread and the fraction of periods where ``v`` changes.
    A noise-free run checks that noise actually broadens the state.
    """
    dt, size = 1e-5, 4096
    af = Frontend([[-0.2 / dt]], [[0.0, -0.5 / dt]], [[1.0]], dt=dt)
    # One state, no external input: B[0, 1] couples the DAC decision to x.
    if kind == "state":
        af.state_covariance = np.array([[4000.0]])
    af = GmC(
        af,
        thermal_noise=False,
        comparator_input_noise_rms=0.25 if kind != "state" else None,
    )

    tran = spice.run(af.netlist(size, seed=11))
    # Discard settling before comparing independent noise realizations.
    x_spice = spice.states(tran, af.N, dt, size)[128:, 0]
    v_spice = spice.decisions(tran, af.N, af.M, dt, size)[128:, 0]
    result = af.simulate(size, seed=29, record=("x", "v"), verbose=False)
    x_cbadc = result["x"][128:, 0, 0]
    v_cbadc = result["v"][128:, 0, 0]
    quiet_af = deepcopy(af)
    quiet_af.state_covariance = quiet_af.output_covariance = None
    quiet = quiet_af.simulate(size, record="x", verbose=False)["x"][128:, 0, 0]

    assert np.std(x_cbadc) > 1.15 * np.std(quiet)
    np.testing.assert_allclose(np.std(x_spice), np.std(x_cbadc), rtol=0.15)
    np.testing.assert_allclose(
        np.mean(v_spice[1:] != v_spice[:-1]),
        np.mean(v_cbadc[1:] != v_cbadc[:-1]),
        atol=0.04,
    )
