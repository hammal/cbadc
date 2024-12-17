import cbadc
import numpy as np


def test_shuffler():
    N = 6
    _indices = np.arange(N)
    shuffler = np.zeros(N, dtype=int)

    N_half = N // 2

    shuffler[:N:2] = _indices[:N_half]
    shuffler[1::2] = _indices[N_half:]
    de_shuffler = np.zeros_like(shuffler)
    de_shuffler[shuffler] = _indices

    if not np.allclose(_indices[shuffler], np.array([0, 3, 1, 4, 2, 5])):
        print(_indices[shuffler])
        raise ValueError
    if not np.allclose(_indices[shuffler][de_shuffler], _indices):
        print(_indices[shuffler][de_shuffler])
        raise ValueError
