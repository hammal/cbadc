from cbadc import StateSpaceSimulator
from cbadc import ConstantSignal, Sinusodial
from cbadc import AnalogSystem
from cbadc import DigitalControl
import numpy as np
from tests.test_analog_system.chain_of_integrators import chain_of_integrators
beta = 6250.0
rho = -62.5
N = 5
M = 5
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0] = beta
C = np.zeros((1, N))
C[-1] = 1.0
Gamma_tilde = np.eye(M)
Gamma = Gamma_tilde * (-beta)
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
        chain_of_integrators["system"], digitalControl, analogSignals, t_stop=Ts * 1000)
    for control_signal in statespacesimulator:
        pass


def test_large_integrator():
    N = 100
    M = N
    A = np.eye(N) * rho + np.eye(N, k=-1) * beta
    B = np.zeros((N, 1))
    B[0] = beta
    C = np.zeros((N, 1))
    C[-1] = 1.0
    Gamma_tilde = np.eye(N)
    Gamma = Gamma_tilde * (-beta)
    analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
    analogSignals = [ConstantSignal(0.1)]
    digitalControl = DigitalControl(Ts, M)
    statespacesimulator = StateSpaceSimulator(
        analogSystem, digitalControl, analogSignals, t_stop=Ts * 1000)
    for control_signal in statespacesimulator:
        pass
