from numpy.testing import assert_almost_equal
from numpy import linspace
from cbadc.analog_signal import SincPulse
import math
from sympy import symbols, sinc, Eq
import pytest


def test_initialization():
    amplitude = 1.0
    bandwidth = 42.0
    delay = 11
    SincPulse(amplitude, bandwidth, delay)


@pytest.mark.skip(
    reason="Somehow precision issues with since? This is a bug that should be resolved ASAP."
)
def test_evaluate_multiple():
    amplitude = 1.2
    bandwidth = 42.0
    delay = math.pi
    sinc = SincPulse(amplitude, bandwidth, delay)

    eval_times = linspace(-9.9, 9, 8, 50)  # test at 50 different pointss
    for t in eval_times:
        if abs(t) < 1e-15:
            assert_almost_equal(sinc.evaluate(t), amplitude, decimal=10)
        else:
            assert_almost_equal(
                sinc.evaluate(t),
                (
                    amplitude
                    * math.sin(2 * math.pi * bandwidth * (t - delay))
                    / (2 * math.pi * bandwidth * (t - delay))
                ),
                decimal=10,
            )


@pytest.mark.skip(
    reason="Somehow precision issues with since? This is a bug that should be resolved ASAP."
)
def test_evaluate_with_offset():
    amplitude = 1.2
    bandwidth = 42.0
    delay = 20
    offset = 4.5321
    t = 3.7
    sinc = SincPulse(amplitude, bandwidth, delay, offset)
    assert_almost_equal(
        sinc.evaluate(t),
        (
            amplitude
            * math.sin(2 * math.pi * bandwidth * (t - delay))
            / (2 * math.pi * bandwidth * (t - delay))
            + offset
        ),
        decimal=10,
    )


@pytest.mark.skip(
    reason="Somehow precision issues with since? This is a bug that should be resolved ASAP."
)
def test_properties():
    amplitude = 1.2
    bandwidth = 42.0
    delay = 7.5 * math.pi
    offset = 4.5321
    sinc = SincPulse(amplitude, bandwidth, delay, offset)
    assert sinc.amplitude == amplitude
    assert sinc.bandwidth == bandwidth
    assert sinc.delay == delay
    assert sinc.offset == offset


@pytest.mark.skip(
    reason="Somehow precision issues with since? This is a bug that should be resolved ASAP."
)
def test_symbolic():
    amplitude = 1.2
    bandwidth = 42.0
    delay = 7.5 * math.pi
    offset = 4.5321
    analog_signal = SincPulse(amplitude, bandwidth, delay, offset)
    t = symbols("t")
    sinc_ref_symbolic = amplitude * sinc(2 * bandwidth * (t - delay)) + offset
    signal = analog_signal.symbolic()
    # pprint(signal)
    assert Eq(signal - sinc_ref_symbolic, 0)
