from numpy.testing import assert_almost_equal
from numpy import linspace 
from cbadc.analog_signal import SincPulse
import math


def test_initialization():
    amplitude = 1.0
    bandwidth = 42.
    delay = 11
    SincPulse(amplitude, bandwidth, delay)


def test_evaluate_multiple():
    amplitude = 1.2
    bandwidth = 42.
    delay = math.pi
    sinc = SincPulse(amplitude, bandwidth, delay)

    eval_times = linspace(-9.9, 9,8, 50)      # test at 50 different pointss
    for t in eval_times:
        if (abs(t) < 1e-15):
            assert_almost_equal(sinc.evaluate(t), amplitude, decimal=10)
        else: 
            assert_almost_equal(sinc.evaluate(t),
                                        ( amplitude * 
                                        math.sin(2 * math.pi * bandwidth * (t-delay)) / 
                                        (2 * math.pi * bandwidth * (t-delay)) ),    
                                    decimal=10)

def test_evaluate_with_offset():
    amplitude = 1.2
    bandwidth = 42.
    delay = 20
    offset = 4.5321
    t = 3.7
    sinc = SincPulse(amplitude, bandwidth, delay, offset)
    assert_almost_equal(sinc.evaluate(t),
                            (amplitude *
                             math.sin(2 * math.pi * bandwidth * (t-delay)) /
                            (2 * math.pi * bandwidth * (t-delay)) + offset ),
                        decimal=10)


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

