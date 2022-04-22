import cbadc
import numpy as np
import pytest


@pytest.mark.parametrize(
    "N_gi",
    [
        pytest.param((2, 10.4), id="N=2"),
        pytest.param((4, 143.2), id="N=4"),
        pytest.param((6, 2.1e3), id="N=6"),
        pytest.param((8, 31.5e3), id="N=8"),
        pytest.param((12, 7.4e6), id="N=12"),
    ],
)
def test_g_i_leap_frog(N_gi):
    n = N_gi[0]
    value = N_gi[1]
    g_i = cbadc.synthesis.leap_frog.g_i(n)
    assert np.abs(value - g_i) / g_i < 1e-1


@pytest.mark.parametrize(
    "N",
    [
        pytest.param(2, id="N=2"),
        pytest.param(4, id="N=4"),
        pytest.param(6, id="N=6"),
        pytest.param(8, id="N=8"),
        pytest.param(12, id="N=12"),
    ],
)
def test_g_i_chain_of_integrators(N):
    n = N
    value = 2 * n + 1
    g_i = cbadc.synthesis.chain_of_integrators.g_i(n)
    assert np.abs(value - g_i) / g_i < 1e-1
