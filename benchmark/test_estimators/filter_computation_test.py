import numpy as np
from cbadc.digital_control import DigitalControl
from cbadc.analog_system import AnalogSystem
from cbadc.analog_signal import Sinusoidal, Clock
from cbadc.digital_estimator import (
    DigitalEstimator,
    ParallelEstimator,
    IIRFilter,
    FIRFilter,
)

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
Ts = 1 / (2 * beta)


eta2 = 1e6
K1 = 1 << 8
K2 = 1 << 8
size = K2 << 2
analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
clock = Clock(Ts)
digitalControl = DigitalControl(clock, N)
analogSignals = [Sinusoidal(0.5, 1)]


def controlSequence():
    while True:
        yield np.ones(N, dtype=np.uint8)


def test_filter_computation_parallel_estimator_algorithm(benchmark):
    def setup():
        ParallelEstimator(analogSystem, digitalControl, eta2, K1, K2)

    benchmark(setup)


def test_filter_computation_digital_estimator_algorithm(benchmark):
    def setup():
        DigitalEstimator(analogSystem, digitalControl, eta2, K1, K2)

    benchmark(setup)


def test_filter_computation_IIR_filter_algorithm(benchmark):
    def setup():
        IIRFilter(analogSystem, digitalControl, eta2, K2)

    benchmark(setup)


def test_filter_computation_FIR_filter_algorithm(benchmark):
    def setup():
        FIRFilter(analogSystem, digitalControl, eta2, K1, K2)

    benchmark(setup)
