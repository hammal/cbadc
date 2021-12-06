from cbadc.analog_signal import Sinusoidal
import math
from sympy import sin, Eq, Symbol, pprint


def test_initialization():
    amplitude = 1.0
    frequency = 42.0
    Sinusoidal(amplitude, frequency)


def test_evaluate():
    amplitude = 1.2
    frequency = 42.0
    t = 3.0
    sinusoidal = Sinusoidal(amplitude, frequency)
    assert sinusoidal.evaluate(t) == (amplitude * math.sin(2 * math.pi * frequency * t))


def test_evaluate_with_offset_and_phase():
    amplitude = 1.2
    frequency = 42.0
    phase = 7.5 * math.pi
    offset = 4.5321
    t = 3.0
    sinusoidal = Sinusoidal(amplitude, frequency, phase, offset)
    assert sinusoidal.evaluate(t) == (
        amplitude * math.sin(2 * math.pi * frequency * t + phase) + offset
    )


def test_properties():
    amplitude = 1.2
    frequency = 42.0
    phase = 7.5 * math.pi
    offset = 4.5321
    sinusoidal = Sinusoidal(amplitude, frequency, phase, offset)
    assert sinusoidal.amplitude == amplitude
    assert sinusoidal.frequency == frequency
    assert sinusoidal.phase == phase
    assert sinusoidal.offset == offset


def test_symbolic():
    amplitude = 1.2
    frequency = 42.0
    phase = 7.5 * math.pi
    offset = 4.5321
    sinusoidal = Sinusoidal(amplitude, frequency, phase, offset)
    t = Symbol("t", real=True)
    sin_ref_symbolic = (
        amplitude * sin(2 * frequency * math.pi * t + phase + sinusoidal.sym_phase)
        + offset
    )
    signal = sinusoidal.symbolic()
    pprint(signal)
    assert Eq(signal - sin_ref_symbolic, 0)
