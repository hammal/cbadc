"""Check chain synthesis dimensions, nearest-neighbor A, and leakage."""

import numpy as np

from cbadc import Frontend


def test_chain_of_integrators_synthesis():
    """A six-state chain has one comparator per state and five A couplings.

    Separately request four states with leakage ``rho`` and check the
    diagonal of A equals that continuous-time decay rate.
    """
    af, OSR = Frontend.chain_of_integrators(N=6, ENOB=14, BW=1e5, xi=1)
    assert af.N == af.M == 6 and af.L == 1
    assert np.count_nonzero(af.A[0, 0]) == 5  # the chain couplings only
    af, _ = Frontend.chain_of_integrators(N=4, OSR=16, BW=1e6, rho=-1e3)
    np.testing.assert_allclose(np.diag(af.A[0, 0]), -1e3)
