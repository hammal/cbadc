from cbadc.analog_signal import ConstantSignal
from sympy import Eq, Symbol, symbols, Float


def test_initialization():
    ConstantSignal()


def test_evaluate():
    analog_signal = ConstantSignal()
    assert analog_signal.evaluate(3.0) == 0.0


def test_evaluate_offset():
    analog_signal = ConstantSignal(0.3)
    assert analog_signal.evaluate(10) == 0.3


def test_properties():
    offset = 2131.213
    analog_signal = ConstantSignal(offset)
    assert analog_signal.offset == offset


def test_symbolic():
    analog_signal = ConstantSignal()
    c = Float(0)
    signal = analog_signal.symbolic()
    assert Eq(signal - c, 0)


def test_addition():
    analog_signal_1 = ConstantSignal(1)
    analog_signal_2 = ConstantSignal(2)
    analog_signal_3 = ConstantSignal(3)
    superposition = analog_signal_1 + analog_signal_2 + analog_signal_3
    superposition.evaluate(1.0)
    print(superposition)


def test_multiplication():
    analog_signal_1 = ConstantSignal(1)
    analog_signal_2 = ConstantSignal(2)
    analog_signal_3 = ConstantSignal(3)
    modulation_signal = analog_signal_1 * analog_signal_2 * analog_signal_3
    modulation_signal.evaluate(1.0)
    print(modulation_signal)
