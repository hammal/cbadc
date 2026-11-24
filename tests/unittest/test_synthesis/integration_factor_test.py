"""Check tabulated leapfrog design factors used by synthesis."""

import numpy as np
import pytest

from cbadc.frontend import _g_i_leapfrog


@pytest.mark.parametrize(
    "N, value", [(2, 10.4), (4, 143.2), (6, 2.1e3), (8, 31.5e3), (12, 7.4e6)]
)
def test_g_i_leap_frog(N, value):
    """Computed factor for each order stays within 10% of its reference."""
    assert np.abs(value - _g_i_leapfrog(N)) / value < 1e-1
