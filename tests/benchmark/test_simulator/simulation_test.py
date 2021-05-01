import numpy as np
from cbadc.digital_control import DigitalControl
from cbadc.analog_system import AnalogSystem
from cbadc.analog_signal import Sinusodial
from cbadc.simulator import StateSpaceSimulator

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
K1 = 1 << 12
K2 = 1 << 12
size = K2 << 4
analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
digitalControl = DigitalControl(Ts, N)
analogSignals = [Sinusodial(amplitude, frequency, phase)]


def iterate_through(iterator):
    count = 0
    for _ in range(size):
        next(iterator)
        count = count + 1
    return count


def test_benchmark_state_space_simulation_algorithm(benchmark):
    est = StateSpaceSimulator(
        analogSystem, digitalControl, analogSignals)
    result = benchmark(iterate_through, est)
    assert(result == size)
