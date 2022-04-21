import cbadc
import numpy as np


def test_g_i():
    N = [2, 4, 6, 8, 12]
    g_ref = [10.4, 143.2, 2.1e3, 31.5e3, 7.4e6]
    for index, n in enumerate(N):
        g_i = cbadc.synthesis.leap_frog.g_i(n)
        assert np.abs(g_ref[index] - g_i) / g_i < 1e-1
