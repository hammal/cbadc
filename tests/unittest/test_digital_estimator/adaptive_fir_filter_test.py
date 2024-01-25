from scipy.signal import firwin
import pytest
import numpy as np


@pytest.fixture()
def setup():
    size = 1 << 12
    R = 2
    M = 5
    K = 1 << 10
    h = np.array([firwin(K, 0.1), firwin(K, 0.2)]).reshape((R, K))
    s_control = np.random.randint(2, size=(size, M + R))
    return R, M, K, h, s_control


def test_init_filter(setup):
    R, _, K, h, _ = setup
    assert h.shape[0] == R
    assert h.shape[1] == K
