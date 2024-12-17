import numpy as np
import sympy as sp
from cbadc.analog_filter import AnalogSystem
from cbadc.analog_signal import Sinusoidal


beta = 6250.0
rho = -62.5
N = 5
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0] = beta
CT = np.zeros((N, 1)).transpose()
CT[-1] = 1.0
Gamma_tildeT = np.eye(N)
Gamma = Gamma_tildeT * (-beta)


def test_homogenous_equations():
    analog_filter = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    analog_signal = Sinusoidal(1.0, 2 * np.pi)
    sp.pprint(analog_filter._A_s)
    eqs, funcs = analog_filter.symbolic_differential_equations(
        analog_signal.symbolic(), 0
    )
    sp.pprint(eqs)
    sp.pprint(funcs)
    sp.pprint(analog_filter.homogenius_solution())
    # assert False


# Next add symbolic for digital control
#
# Move solver into simulator
