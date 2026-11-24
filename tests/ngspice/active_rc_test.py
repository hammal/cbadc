"""Validate the two-pole active-RC macromodel against an ngspice transient."""

import numpy as np
import pytest

from cbadc.circuit import ActiveRC
from cbadc.frontend import DAC, Constant, Frontend

from . import spice


@pytest.mark.parametrize("Ro", [1e7, np.inf])
def test_active_rc_states_and_decisions_match_ngspice(Ro):
    """The Cint/Co capacitors and R/G currents reproduce augmented ABCD.

    Start both solvers at the same nonzero ``[x; p]`` state, then compare the
    state voltages at clock edges and the binary decisions between edges.
    """
    af = Frontend(
        [[-1e5]],
        [[1e5, -1e5]],
        [[1.0]],
        dt=1e-5,
        input=Constant(0.3),
        dac=DAC(Constant(1.0)),
    )
    rc = ActiveRC(
        af,
        Cint=1e-10,
        Cp=0.2e-10,
        C_design=1e-10,
        gm=1e-4,
        Ro=Ro,
        Co=1e-10,
        thermal_noise=False,
    )
    size, x0 = 48, np.array([0.2, 0.0])
    tran = spice.run(rc.netlist(size, x0))
    states = spice.states(tran, rc.N, rc.dt, size)
    decisions = spice.decisions(tran, rc.N, rc.M, rc.dt, size)
    sim = rc.simulate(size, x0=x0, record=("x", "v"), verbose=False)
    # Ngspice's first saved point follows t=0; compare actual clock edges.
    np.testing.assert_allclose(states[1:], sim["x"][1:, :, 0], atol=2e-3, rtol=0)
    np.testing.assert_array_equal(decisions, sim["v"][:, :, 0])
