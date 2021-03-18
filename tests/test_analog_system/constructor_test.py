from tests.test_analog_system.chain_of_integrators import chain_of_integrators
import pytest
import os
import sys
import numpy as np

# this is python madness
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")
if myPath:
    from cbc.analog_system import AnalogSystem, InvalidAnalogSystemError


beta = 6250.0
rho = -62.5
N = 5
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0] = beta
C = np.zeros((N,1))
C[-1] = 1.0
Gamma_tilde = np.eye(N)
Gamma = Gamma_tilde * (-beta)


def test_correct_initialization():
    analog_system = AnalogSystem(A, B, C, Gamma, Gamma_tilde)


def test_chain_of_integrators_fixture(chain_of_integrators):
    print(chain_of_integrators["system"])
    np.testing.assert_allclose(
        chain_of_integrators["A"], chain_of_integrators["system"].A()
    )
    np.testing.assert_allclose(
        chain_of_integrators["B"], chain_of_integrators["system"].B()
    )
    np.testing.assert_allclose(
        chain_of_integrators["C"], chain_of_integrators["system"].C()
    )
    np.testing.assert_allclose(
        chain_of_integrators["Gamma"], chain_of_integrators["system"].Gamma(
        )
    )
    np.testing.assert_allclose(
        chain_of_integrators["Gamma_tilde"],
        chain_of_integrators["system"].Gamma_tilde(),
    )


def test_non_square_system_matrix():
    Atemp = np.ones((1, N))
    with pytest.raises(InvalidAnalogSystemError):
        AnalogSystem(Atemp, B, C, Gamma, Gamma_tilde)


def test_wrong_n_B():
    B = np.ones((N - 1, 2))
    with pytest.raises(InvalidAnalogSystemError):
        AnalogSystem(A, B, C, Gamma, Gamma_tilde)


def test_wrong_n_C():
    C = np.ones((1, N - 1))
    with pytest.raises(InvalidAnalogSystemError):
        AnalogSystem(A, B, C, Gamma, Gamma_tilde)


def test_wrong_n_Gamma():
    Gamma_temp = Gamma[1:, :]
    with pytest.raises(InvalidAnalogSystemError):
        AnalogSystem(A, B, C, Gamma_temp, Gamma_tilde)


def test_wrong_n_Gamma_tilde():
    Gamma_tilde_temp = Gamma_tilde[:, 1:]
    with pytest.raises(InvalidAnalogSystemError):
        AnalogSystem(A, B, C, Gamma, Gamma_tilde_temp)
