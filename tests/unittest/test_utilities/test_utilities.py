import cbadc
import pytest


@pytest.mark.parametrize("M", [pytest.param(M, id=f"M={M}") for M in range(8, 129, 8)])
def test_bytes_selector(M):
    res_obj = cbadc.utilities.number_of_bytes_selector(M)
    # assert res_obj['number_of_bytes'] == M // 8
