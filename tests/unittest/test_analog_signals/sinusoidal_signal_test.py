from cbadc.analog_signal import Sinusodial
import math


def test_initialization():
    amplitude = 1.0
    frequency = 42.
    Sinusodial(amplitude, frequency)


def test_evaluate():
    amplitude = 1.2
    frequency = 42.
    t = 3.
    sinusoidal = Sinusodial(amplitude, frequency)
    assert sinusoidal.evaluate(t) == (amplitude *
                                      math.sin(2 * math.pi * frequency * t))


def test_evaluate_with_offset_and_phase():
    amplitude = 1.2
    frequency = 42.
    phase = 7.5 * math.pi
    offset = 4.5321
    t = 3.
    sinusoidal = Sinusodial(amplitude, frequency, phase, offset)
    assert sinusoidal.evaluate(t) == (amplitude *
                                      math.sin(2 * math.pi * frequency * t + phase) + offset)


def test_properties():
    amplitude = 1.2
    frequency = 42.
    phase = 7.5 * math.pi
    offset = 4.5321
    sinusoidal = Sinusodial(amplitude, frequency, phase, offset)
    assert sinusoidal.amplitude == amplitude
    assert sinusoidal.frequency == frequency
    assert sinusoidal.phase == phase
    assert sinusoidal.offset == offset
