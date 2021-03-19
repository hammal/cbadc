from cbc import StateSpaceSimulator
from cbc.digital_estimator import DigitalEstimator
from cbc.parallel_digital_estimator.digital_estimator import DigitalEstimator as ParallelDigitalEstimator
from cbc import AnalogSignal, Sinusodial
from cbc import AnalogSystem
from cbc import DigitalControl
import numpy as np
import time

beta = 6250.0
rho = -62.5
N = 10
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0, 0] = beta
# B[0, 1] = -beta
C = np.eye(N)
Gamma_tilde = np.eye(N)
Gamma = Gamma_tilde * (-beta)
Ts = 1/(2 * beta)


eta2 = 1e12
K1 = 10
K2 = 1000
size = 10000
analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
digitalControl = DigitalControl(Ts, N)
analogSignals = [Sinusodial(0.5, 1)]


def controlSequence():
    while True:
        yield np.ones(N, dtype=np.uint8)


# def test_initialization_parallel_linear_algorithm(benchmark):
#     def setup():
#         ParallelDigitalEstimator(
#             controlSequence(), analogSystem, digitalControl, eta2, K1, K2)
#     benchmark(setup)


# def test_initialization_quadratic_algorithm(benchmark):
#     def setup():
#         DigitalEstimator(
#             controlSequence(), analogSystem, digitalControl, eta2, K1, K2)
#     benchmark(setup)


# def test_initialization_circuit_simulation_algorithm(benchmark):
#     def setup():
#         StateSpaceSimulator(
#             analogSystem, digitalControl, analogSignals)
#     benchmark(setup)
