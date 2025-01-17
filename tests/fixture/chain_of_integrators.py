from typing import Union
import numpy as np
import pytest
from cbadc.analog_filter import AnalogSystem
import cbadc


@pytest.fixture
def chain_of_integrators() -> dict[str, Union[np.ndarray, int, float]]:
    beta = 6250.0
    rho = -62.5
    N = 5
    A = np.eye(N) * rho + np.eye(N, k=-1) * beta
    B = np.zeros((N, N + 1))
    B[0, 0] = beta
    B[:, 1:] = -beta * np.eye(N)
    C = np.eye(N)
    D = np.zeros((N, N + 1))

    dt = 1.0 / np.abs(2.0 * beta)
    return {
        "N": N,
        "M": N,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "beta": beta,
        "rho": rho,
        "dt": dt,
    }
