"""Signals honour a configurable floating-point ``dtype`` (e.g. float32)."""

import numpy as np

from cbadc.analog_signal import Sinusoidal, ZeroOrderHold


def test_sinusoidal_dtype_defaults_to_float64():
    s = Sinusoidal(np.array([1.0]), np.array([1e3]))
    out = s.evaluate(np.linspace(0, 1e-3, 16))
    assert out.dtype == np.dtype(float)


def test_sinusoidal_float32_propagates():
    s = Sinusoidal(np.array([1.0]), np.array([1e3]), dtype=np.float32)
    assert s.dtype == np.float32
    out = s.evaluate(np.linspace(0, 1e-3, 16))
    assert out.dtype == np.float32


def test_zero_order_hold_float32_propagates():
    values = np.ones((4, 1, 1))
    z = ZeroOrderHold(1e-4, values, dtype=np.float32)
    out = z.evaluate(np.linspace(0, 3e-4, 8))
    assert out.dtype == np.float32
