from cbc import AnalogSignal


def test_initialization():
    analog_signal_without_offset = AnalogSignal()


def test_evaluate():
    analog_signal = AnalogSignal()
    assert analog_signal.evaluate(3.) == 0.0


def test_evaluate_offset():
    analog_signal = AnalogSignal(0.3)
    assert analog_signal.evaluate(10) == 0.3


def test_properties():
    offset = 2131.213
    analog_signal = AnalogSignal(offset)
    assert analog_signal.offset == offset
