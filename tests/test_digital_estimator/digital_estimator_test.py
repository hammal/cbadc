from cbc.digital_estimator.digital_estimator import DigitalEstimator
from cbc.digital_estimator.filter import Filter
from cbc.analog_signal import AnalogSignal, Sinusodial
from cbc.analog_system import AnalogSystem
from cbc.digital_control import DigitalControl
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
    digitalControl = DigitalControl(Ts, M)
    eta2 = 1.0
    K1 = 100
    K2 = 10

    def controlSequence():
        yield np.ones(M)
    print(digitalControl.Ts())
    filter = Filter(
        chain_of_integrators['system'], digitalControl, eta2, K1, K2)
    DigitalEstimator(filter, controlSequence)


# def test_iterator(chain_of_integrators):
#     analogSignals = [AnalogSignal(0.1)]
#     digitalControl = DigitalControl(Ts, M)
#     circuitSimulator = CircuitSimulator(
#         chain_of_integrators["system"], digitalControl, analogSignals, t_stop=1.)
#     for control_signal in circuitSimulator:
#         pass


# def test_large_integrator():
#     N = 100
#     M = N
#     A = np.eye(N) * rho + np.eye(N, k=-1) * beta
#     B = np.zeros((N, 1))
#     B[0] = beta
#     C = np.zeros((1, N))
#     C[-1] = 1.0
#     Gamma_tilde = np.eye(N)
#     Gamma = Gamma_tilde * (-beta)
#     analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
#     analogSignals = [AnalogSignal(0.1)]
#     digitalControl = DigitalControl(Ts, M)
#     circuitSimulator = CircuitSimulator(
#         analogSystem, digitalControl, analogSignals, t_stop=Ts * 10000)
#     for control_signal in circuitSimulator:
#         pass
