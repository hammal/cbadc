from tests.test_analog_system.chain_of_integrators import chain_of_integrators
import pytest
import os
import sys
import numpy as np

# this is python madness
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")
if myPath:
    from cbadc.analog_system import AnalogSystem, InvalidAnalogSystemError


beta = 6250.0
rho = -62.5
N = 5
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0] = beta
CT = np.zeros((N, 1)).transpose()
CT[-1] = 1.0
Gamma_tildeT = np.eye(N)
Gamma = Gamma_tildeT * (-beta)


def test_correct_initialization():
    AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)


def test_chain_of_integrators_fixture(chain_of_integrators):
    print(chain_of_integrators["system"])
    np.testing.assert_allclose(
        chain_of_integrators["A"], chain_of_integrators["system"].A
    )
    np.testing.assert_allclose(
        chain_of_integrators["B"], chain_of_integrators["system"].B
    )
    np.testing.assert_allclose(
        chain_of_integrators["CT"], chain_of_integrators["system"].CT
    )
    np.testing.assert_allclose(
        chain_of_integrators["Gamma"], chain_of_integrators["system"].Gamma
    )
    np.testing.assert_allclose(
        chain_of_integrators["Gamma_tildeT"],
        chain_of_integrators["system"].Gamma_tildeT,
    )


def test_non_square_system_matrix():
    Atemp = np.ones((1, N))
    with pytest.raises(InvalidAnalogSystemError):
        AnalogSystem(Atemp, B, CT, Gamma, Gamma_tildeT)


def test_wrong_n_B():
    B = np.ones((N - 1, 2))
    with pytest.raises(InvalidAnalogSystemError):
        AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)


def test_wrong_n_C():
    CT = np.ones((1, N - 1)).transpose()
    with pytest.raises(InvalidAnalogSystemError):
        AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)


def test_wrong_n_Gamma():
    Gamma_temp = Gamma[1:, :]
    with pytest.raises(InvalidAnalogSystemError):
        AnalogSystem(A, B, CT, Gamma_temp, Gamma_tildeT)


def test_wrong_n_Gamma_tilde():
    Gamma_tilde_temp = Gamma_tildeT[:, 1:]
    with pytest.raises(InvalidAnalogSystemError):
        AnalogSystem(A, B, CT, Gamma, Gamma_tilde_temp)
