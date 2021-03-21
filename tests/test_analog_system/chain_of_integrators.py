import numpy as np
import pytest
import os
import sys

# this is python madness
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")
if myPath:
    from cbadc.analog_system import AnalogSystem


@pytest.fixture
def chain_of_integrators():
    beta = 6250.0
    rho = -62.5
    N = 5
    A = np.eye(N) * rho + np.eye(N, k=-1) * beta
    B = np.zeros((N, 1))
    B[0] = beta
    C = np.zeros((N, 1))
    C[-1] = 1.0
    Gamma_tilde = np.eye(N)
    Gamma = Gamma_tilde * (-beta)
    return {
        "N": N,
        "A": A,
        "B": B,
        "C": C,
        "M": N,
        "Gamma": Gamma,
        "Gamma_tilde": Gamma_tilde,
        "system": AnalogSystem(A, B, C, Gamma, Gamma_tilde),
        "beta": beta,
        "rho": rho,
    }
