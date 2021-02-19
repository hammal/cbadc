from cbc.circuit_simulator import CircuitSimulator
from cbc.digital_estimator.digital_estimator import DigitalEstimator
from cbc.parallel_digital_estimator.digital_estimator import DigitalEstimator as ParallelDigitalEstimator
from cbc.analog_signal import AnalogSignal, Sinusodial
from cbc.analog_system import AnalogSystem
from cbc.digital_control import DigitalControl
import numpy as np

beta = 6250.0
rho = -62.5
N = 5
M = 5
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0, 0] = beta
# B[0, 1] = -beta
C = np.eye(N)
Gamma_tilde = np.eye(M)
Gamma = Gamma_tilde * (-beta)
Ts = 1/(2 * beta)


def test_estimation_with_circuit_simulator():
    eta2 = 1e12
    K1 = 100
    K2 = 100
    size = 1000

    analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
    analogSignals = [Sinusodial(0.5, 1)]
    # analogSignals = [AnalogSignal(0.25)]
    digitalControl1 = DigitalControl(Ts, M)

    circuitSimulator1 = CircuitSimulator(
        analogSystem, digitalControl1, analogSignals)
    estimator1 = DigitalEstimator(
        circuitSimulator1, analogSystem, digitalControl1, eta2, K1, K2)
    digitalControl2 = DigitalControl(Ts, M)
    circuitSimulator2 = CircuitSimulator(
        analogSystem, digitalControl2, analogSignals)
    estimator2 = ParallelDigitalEstimator(
        circuitSimulator2, analogSystem, digitalControl1, eta2, K1, K2)
    model_error = 0
    e1_error = 0
    e2_error = 0
    for index in range(size):
        e1 = estimator1.__next__()
        e2 = estimator2.__next__()
        t = index * Ts
        u = analogSignals[0].evaluate(t)
        print("Input Signal       : ", u, " t: ", t)
        print("Estimator Quadratic: ", e1)
        print("Estimator Linear   : ", e2)
        model_error += np.abs(e1 - e2)**2
        e1_error += np.abs(e1 - u)**2
        e2_error += np.abs(e2 - u)**2
    model_error /= size
    e1_error /= size
    e2_error /= size
    print("Model Error:                 ", model_error)
    print("Quadratic estimator error:   ", e1_error)
    print("Linear estimator error:      ", e2_error)
    # assert(np.allclose(e1-e2))
    raise "Temp"
