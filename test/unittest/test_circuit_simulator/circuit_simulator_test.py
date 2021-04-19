from cbadc.simulator import StateSpaceSimulator
from cbadc.analog_signal import ConstantSignal
from cbadc.analog_system import AnalogSystem
from cbadc.digital_control import DigitalControl
import numpy as np
from test.fixture.chain_of_integrators import chain_of_integrators

beta = 6250.0
rho = -62.5
N = 5
M = 5
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0] = beta
CT = np.zeros((1, N)).transpose()
CT[-1] = 1.0
Gamma_tildeT = np.eye(M)
Gamma = Gamma_tildeT * (-beta)
Ts = 1/(2 * beta)


def test_initialization(chain_of_integrators):
    analogSignals = [ConstantSignal(0.1)]
    digitalControl = DigitalControl(Ts, M)
    StateSpaceSimulator(
        chain_of_integrators["system"], digitalControl, analogSignals)


def test_iterator(chain_of_integrators):
    analogSignals = [ConstantSignal(0.1)]
    digitalControl = DigitalControl(Ts, M)
    statespacesimulator = StateSpaceSimulator(
        chain_of_integrators["system"], digitalControl, analogSignals,
        t_stop=Ts * 1000)
    for control_signal in statespacesimulator:
        pass


def test_large_integrator():
    N = 20
    M = N
    A = np.eye(N) * rho + np.eye(N, k=-1) * beta
    B = np.zeros((N, 1))
    B[0] = beta
    CT = np.zeros((N, 1)).transpose()
    CT[-1] = 1.0
    Gamma_tildeT = np.eye(N)
    Gamma = Gamma_tildeT * (-beta)
    analogSystem = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    analogSignals = [ConstantSignal(0.1)]
    digitalControl = DigitalControl(Ts, M)
    statespacesimulator = StateSpaceSimulator(
        analogSystem, digitalControl, analogSignals, t_stop=Ts * 1000)
    for control_signal in statespacesimulator:
        pass
