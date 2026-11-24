"""Check that each batch member uses its own ABCD map and circuit."""

import numpy as np

from cbadc.circuit import GmC
from cbadc.frontend import Frontend, Sinusoid

from . import bench


def test_two_distinct_frontends_match_their_circuits():
    """Two batch members with different A must behave like separate circuits.

    Build ``J=2, K=1`` with shared B and C but distinct 24-state A matrices.
    Each step map has 24 x rows, two tone states, and one comparator row.
    These 27-row maps exercise Numba's packed path. For
    each member, compare its isolated cbadc run with ngspice, then compare
    batch ``v``, ``x``, and twin ``y_hat`` with that isolated run.
    """
    n, size, dt = 24, 128, 0.1
    f = 1.0 / (32 * dt)
    # J=2, K=1: A[1, 0] is 1.04 times A[0, 0]; both are 24 x 24.
    a = -np.diag(np.linspace(1.0, 2.0, n))
    a += 0.02 * np.tril(np.ones((n, n)), -1) / n
    matrices = np.stack((a, a * 1.04))[:, None]
    # B (N, L+M) couples u and v; C (M, N) averages x into y. D=0.
    b = np.column_stack((np.linspace(0.25, 0.4, n), -np.linspace(0.3, 0.2, n)))
    c = np.ones((1, n)) / n
    af = GmC(
        Frontend(matrices, b, c, dt=dt, input=Sinusoid(0.5, f)), thermal_noise=False
    )
    dz = af.discretize()
    assert af.J == dz.J_m == 2
    assert dz.N_x + af.M == 27
    x0 = np.full(n, 0.1)
    batch = af.simulate(size, x0=x0, record="all", verbose=False)

    for j in range(2):
        single = af[j]
        cmp = bench.run_both(single, size, x0)
        bench.assert_agree(cmp, min_agree=size, margin=1e-4, state_tol=1e-4)
        np.testing.assert_array_equal(batch["v"][:, :, j], cmp.r["v"][:, :, 0])
        np.testing.assert_allclose(batch["x"][:, :, j], cmp.r["x"][:, :, 0], atol=1e-12)
        np.testing.assert_allclose(
            batch["y_hat"][:, :, j], cmp.r["y_hat"][:, :, 0], atol=1e-12
        )
