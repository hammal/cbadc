"""Check leapfrog synthesis dimensions, clock rate, and A structure."""

import numpy as np

from cbadc import Frontend


def test_leapfrog_synthesis_shapes_and_clock():
    """ENOB and OSR designs give the requested size and sample rate.

    In the continuous-time A matrix, upper and lower neighbor couplings
    have the expected opposite signs.
    """
    af, OSR = Frontend.leapfrog(N=6, ENOB=14, BW=1e5, xi=1)
    assert af.N == af.M == 6 and af.L == 1
    assert OSR == 1.0 / (2.0 * af.dt * 1e5)
    af, OSR = Frontend.leapfrog(N=4, OSR=32, BW=1e6)
    assert OSR == 32 and af.dt == 1.0 / (2 * 32 * 1e6)
    assert np.all(np.diag(af.A[0, 0], 1) > 0) and np.all(np.diag(af.A[0, 0], -1) < 0)
