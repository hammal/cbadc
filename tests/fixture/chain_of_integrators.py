import numpy as np
import pytest
from cbadc.analog_system import AnalogSystem
import cbadc


@pytest.fixture
def chain_of_integrators():
    beta = 6250.0
    rho = -62.5
    N = 5
    A = np.eye(N) * rho + np.eye(N, k=-1) * beta
    B = np.zeros((N, 1))
    B[0] = beta
    CT = np.zeros((N, 1)).transpose()
    CT[0, -1] = 1.0
    Gamma_tildeT = np.eye(N)
    Gamma = Gamma_tildeT * (-beta)
    return {
        "N": N,
        "A": A,
        "B": B,
        "CT": CT,
        "M": N,
        "Gamma": Gamma,
        "Gamma_tildeT": Gamma_tildeT,
        "system": AnalogSystem(A, B, CT, Gamma, Gamma_tildeT),
        "beta": beta,
        "rho": rho,
    }


@pytest.fixture
def chain_of_integrators_op_amp():
    beta = 6250.0
    rho = 0.0
    N = 5
    A = np.eye(N) * rho - np.eye(N, k=-1) * beta
    B = np.zeros((N, 1))
    B[0] = -beta
    CT = np.zeros((N, 1)).transpose()
    CT[0, -1] = 1.0
    Gamma_tildeT = np.eye(N)
    Gamma = Gamma_tildeT * (-beta)
    return {
        "N": N,
        "A": A,
        "B": B,
        "CT": CT,
        "M": N,
        "Gamma": Gamma,
        "Gamma_tildeT": Gamma_tildeT,
        "system": AnalogSystem(A, B, CT, Gamma, Gamma_tildeT),
        "beta": beta,
        "rho": rho,
    }


@pytest.fixture
def chain_of_integrators_op_amp_small():
    beta = 6250.0
    rho = -0.1
    N = 2
    A = np.eye(N) * rho - np.eye(N, k=-1) * beta
    B = np.zeros((N, 1))
    B[0] = -beta
    CT = np.zeros((N, 1)).transpose()
    CT[0, -1] = 1.0
    Gamma_tildeT = np.eye(N)
    Gamma = Gamma_tildeT * (-beta)
    return {
        "N": N,
        "A": A,
        "B": B,
        "CT": CT,
        "M": N,
        "Gamma": Gamma,
        "Gamma_tildeT": Gamma_tildeT,
        "system": AnalogSystem(A, B, CT, Gamma, Gamma_tildeT),
        "beta": beta,
        "rho": rho,
    }


@pytest.fixture()
def get_simulator(chain_of_integrators):
    Ts = 1 / (2 * chain_of_integrators['beta'])
    M = chain_of_integrators['M']
    analogSignals = [cbadc.analog_signal.ConstantSignal(0.1)]
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    simulator = cbadc.simulator.get_simulator(
        chain_of_integrators["system"], digitalControl, analogSignals, clock
    )
    return simulator
