from cbadc import ZeroOrderHold
import numpy as np


def test_binary_reference_signal():
    dt = 1.0
    amplitude = np.array([[1.0], [0.5]])
    offset = np.zeros_like(amplitude)
    ZeroOrderHold.binary_reference_signal(dt, amplitude, offset=offset, size=1 << 5)


def test_ternary_reference_signal():
    dt = 1.0
    amplitude = np.array([1.0])
    offset = np.zeros_like(amplitude)
    ZeroOrderHold.ternary_reference_signal(dt, amplitude, offset=offset, size=1 << 5)


def test_gaussian_reference_signal():
    dt = 1.0
    mean = np.array([0.0])
    std = np.ones_like(mean)
    ZeroOrderHold.gaussian_reference_signal(dt, mean, std, size=1 << 5)


def test_uniform_reference_signal():
    dt = 1.0
    low = np.array([0.0])
    high = np.array([1.0])
    ZeroOrderHold.uniform_reference_signal(dt, low, high, size=1 << 5)
