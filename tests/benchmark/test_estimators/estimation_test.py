import numpy as np
from cbadc.digital_control import DigitalControl
from cbadc.analog_system import AnalogSystem
from cbadc.analog_signal import Sinusoidal
from cbadc.digital_estimator import DigitalEstimator, ParallelEstimator, \
    FIRFilter, IIRFilter

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

amplitude = 1.0
frequency = 10.
phase = 0.

eta2 = 1e6
K1 = 1 << 8
K2 = 1 << 8
size = K2 << 2
analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
digitalControl = DigitalControl(Ts, N)
analogSignals = [Sinusoidal(amplitude, frequency, phase)]

t = np.linspace(0, Ts * (size - 1), size)


def controlSequence():
    while True:
        yield np.ones(N, dtype=np.uint8)


def iterate_through(iterator):
    count = 0
    for _ in range(size):
        iterator.__next__()
        count = count + 1
    return count


def test_benchmark_parallel_estimator_algorithm(benchmark):
    est = ParallelEstimator(analogSystem, digitalControl, eta2, K1, K2)
    est(controlSequence())
    result = benchmark(iterate_through, est)
    assert(result == size)


def test_benchmark_digital_estimator_algorithm(benchmark):
    est = DigitalEstimator(
        analogSystem, digitalControl, eta2, K1, K2)
    est(controlSequence())
    result = benchmark(iterate_through, est)
    assert(result == size)


def test_benchmark_fir_filter_algorithm(benchmark):
    est = FIRFilter(
        analogSystem, digitalControl, eta2, K1, K2)
    est(controlSequence())
    result = benchmark(iterate_through, est)
    assert(result == size)


def test_benchmark_iir_estimator_algorithm(benchmark):
    est = IIRFilter(
        analogSystem, digitalControl, eta2, K2)
    est(controlSequence())
    result = benchmark(iterate_through, est)
    assert(result == size)
