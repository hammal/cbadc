from cbadc.simulator import StateSpaceSimulator
from cbadc.digital_estimator import DigitalEstimator
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
B[0, 0] = beta
# B[0, 1] = -beta
CT = np.eye(N)
Gamma_tildeT = np.eye(M)
Gamma = Gamma_tildeT * (-beta)
Ts = 1/(2 * beta)


def controlSequence():
    while True:
        yield np.ones(M, dtype=np.uint8)


def test_initialization(chain_of_integrators):
    digitalControl = DigitalControl(Ts, M)
    eta2 = 1.0
    K1 = 100
    K2 = 0
    DigitalEstimator(controlSequence(
    ), chain_of_integrators['system'], digitalControl, eta2, K1, K2)


def test_estimation():
    digitalControl = DigitalControl(Ts, M)
    eta2 = 100.0
    K1 = 100
    K2 = 10

    analogSystem = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)

    estimator = DigitalEstimator(
        controlSequence(), analogSystem, digitalControl, eta2, K1, K2,
        stop_after_number_of_iterations=25)
    for est in estimator:
        print(np.array(est))


def test_batch_iterations():
    digitalControl = DigitalControl(Ts, M)
    eta2 = 100.0
    K1 = 25
    K2 = 1000

    analogSystem = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    estimator = DigitalEstimator(controlSequence(
    ), analogSystem, digitalControl, eta2, K1, K2=K2,
    stop_after_number_of_iterations=200)
    for est in estimator:
        print(np.array(est))
    # raise "temp"


def test_estimation_with_circuit_simulator():
    eta2 = 1e12
    K1 = 1000
    K2 = 0

    analogSystem = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    # analogSignals = [Sinusodial(0.5, 10)]
    analogSignals = [ConstantSignal(0.25)]
    digitalControl = DigitalControl(Ts, M)
    circuitSimulator = StateSpaceSimulator(
        analogSystem, digitalControl, analogSignals, t_stop=Ts * 1000)
    estimator = DigitalEstimator(
        circuitSimulator, analogSystem, digitalControl, eta2, K1, K2)
    for est in estimator:
        print(est)
    # raise "Temp"
