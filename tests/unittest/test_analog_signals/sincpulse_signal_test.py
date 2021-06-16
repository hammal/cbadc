from cbadc.analog_signal import SincPulse
import math


def test_initialization():
    amplitude = 1.0
    bandwidth = 42.
    delay = 11
    SincPulse(amplitude, bandwidth, delay)


def test_evaluate():
    amplitude = 1.2
    bandwidth = 42.
    delay = 11
    t = 1
    sinc = SincPulse(amplitude, bandwidth, delay)
    assert sinc.evaluate(t) == (amplitude *
                                    math.sin(2 * math.pi * bandwidth * (t-delay)) /
                                    (2 * math.pi * bandwidth * (t-delay))
                                ) 


def test_evaluate_with_offset_and_phase():
    amplitude = 1.2
    bandwidth = 42.
    delay = 20
    offset = 4.5321
    t = 3.7
    sinc = SincPulse(amplitude, bandwidth, delay)
    assert sinc.evaluate(t) == (amplitude *
                                    math.sin(2 * math.pi * bandwidth * (t-delay)) /
                                    (math.pi * (t-delay)) + offset
                                )


def test_properties():
    amplitude = 1.2
    bandwidth = 42.
    delay = 7.5 * math.pi
    offset = 4.5321
    sinc = SincPulse(amplitude, bandwidth, delay, offset)
    assert sinc.amplitude == amplitude
    assert sinc.bandwidth == bandwidth
    assert sinc.delay == delay
    assert sinc.offset == offset
