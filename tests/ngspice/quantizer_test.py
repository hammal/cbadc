"""Check the comparator equation ``y = Cx + D[u; v] + offset`` in ngspice.

Both tests change one term in the same three-state leapfrog, then compare
clocked decisions ``v`` and states ``x`` with the generated circuit.
"""

import numpy as np

from cbadc.circuit import GmC
from cbadc.frontend import DAC, Constant, Frontend, Sinusoid

from . import bench

SIZE = 1 << 11
X0 = 0.01


def _leapfrog():
    """Use one immediate NRZ, sinusoid-driven loop for both comparisons."""
    af, _ = Frontend.leapfrog(N=3, ENOB=12, BW=1e3)
    af.dac = DAC(Constant(1.0))
    af.input = Sinusoid(0.5, af.fs / 512)
    return af


def test_comparator_offset():
    """A nonzero offset on each comparator must shift the circuit's decisions."""
    af = _leapfrog()
    # The offset adds to each row of y; A, B, C, and D stay unchanged.
    af.comparator_input_offset = np.array([0.1, -0.05, 0.02])
    cmp = bench.run_both(GmC(af, thermal_noise=False), SIZE, np.full(af.N, X0))
    bench.assert_agree(cmp, min_agree=SIZE // 2, margin=1e-3, state_tol=1e-3)


def test_input_feedthrough():
    """Direct input feedthrough into one comparator must match the circuit.

    ``D[-1, 0]`` adds ``0.2 u`` to the last comparator input without passing
    through a state. Compare its decisions and states with ngspice.
    """
    af = _leapfrog()
    D = np.zeros((af.M, af.L + af.M))
    D[-1, 0] = 0.2
    af.D = D  # Frontend adds the shared J=K=1 axes.
    cmp = bench.run_both(GmC(af, thermal_noise=False), SIZE, np.full(af.N, X0))
    bench.assert_agree(cmp, min_agree=SIZE // 2, margin=1e-3, state_tol=1e-3)
