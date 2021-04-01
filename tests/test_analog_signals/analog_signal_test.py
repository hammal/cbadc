from cbadc.analog_signal import ConstantSignal


def test_initialization():
    ConstantSignal()


def test_evaluate():
    analog_signal = ConstantSignal()
    assert analog_signal.evaluate(3.) == 0.0


def test_evaluate_offset():
    analog_signal = ConstantSignal(0.3)
    assert analog_signal.evaluate(10) == 0.3


def test_properties():
    offset = 2131.213
    analog_signal = ConstantSignal(offset)
    assert analog_signal.offset == offset
