from numpy.core.numeric import allclose
from cbadc.analog_signal.clock import Clock
import sympy as sp
import numpy as np


def test_clock_numeric():
    T = 1e-3
    tt = 1e-6
    td = 1e-12
    duty_cycle = 0.4
    analog_signal = Clock(T, tt, td, duty_cycle)
    for k in range(-100, 100):
        assert np.allclose(analog_signal(k * T + td + tt / 2), 1)
        assert np.allclose(analog_signal(k * T + td + duty_cycle * T + tt), -1)
