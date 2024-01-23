import numpy as np
from cbadc.digital_control import DigitalControl
from cbadc.analog_system import AnalogSystem
from cbadc.analog_signal import Sinusoidal, Clock
from cbadc.simulator.numerical_simulator import (
    FullSimulator,
    PreComputedControlSignalsSimulator,
)

beta = 6250.0
rho = -62.5
N = 5
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


def test_pre_full_solver_algorithm(benchmark):
    def setup():
        FullSimulator(analogSystem, digitalControl, analogSignals)

    benchmark(setup)


def test_pre_pre_computed_control_algorithm(benchmark):
    def setup():
        PreComputedControlSignalsSimulator(analogSystem, digitalControl, analogSignals)

    benchmark(setup)
