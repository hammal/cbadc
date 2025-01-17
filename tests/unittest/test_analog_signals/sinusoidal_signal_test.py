from cbadc.analog_signal import Sinusoidal
import numpy as np


def test_initialization():
    amplitude = np.array([1.0])
    frequency = np.array([42.0])
    Sinusoidal(amplitude, frequency)


def test_evaluate():
    amplitude = np.array([1.2])
    frequency = np.array([42.0])
    t = np.array([3.0])
    sinusoidal = Sinusoidal(amplitude, frequency)
    assert sinusoidal.evaluate(t) == (amplitude * np.sin(2 * np.pi * frequency * t))


def test_evaluate_with_offset_and_phase():
    amplitude = np.array([1.2])
    frequency = np.array([42.0])
    phase = np.array([7.5 * np.pi])
    offset = np.array([4.5321])
    t = np.array([3.0])
    sinusoidal = Sinusoidal(amplitude, frequency, phase, offset)
    assert np.isclose(
        sinusoidal.evaluate(t),
        np.array([amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset]),
    )


def test_properties():
    amplitude = np.array([1.2])
    frequency = np.array([42.0])
    phase = np.array([7.5 * np.pi])
    offset = np.array([4.5321])
    sinusoidal = Sinusoidal(amplitude, frequency, phase, offset)
    assert sinusoidal.amplitude == amplitude
    assert sinusoidal.frequency == frequency
    assert sinusoidal.phase == phase
    assert sinusoidal.offset == offset
