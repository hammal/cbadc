from cbadc import AnalogSignal
import numpy as np


def test_initialization():
    AnalogSignal()


def test_evaluate():
    analog_signal = AnalogSignal()
    assert analog_signal.evaluate(np.array([3.0])) == 0.0


def test_evaluate_offset():
    analog_signal = AnalogSignal(np.array([0.3]))
    assert analog_signal.evaluate(np.array([10])) == 0.3


def test_properties():
    offset = np.array([[2131.213]])
    analog_signal = AnalogSignal(offset)
    assert analog_signal.offset.shape == offset.shape
    assert analog_signal.offset == offset


def test_addition():
    analog_signal_1 = AnalogSignal(np.array([1]))
    analog_signal_2 = AnalogSignal(np.array([2]))
    analog_signal_3 = AnalogSignal(np.array([3]))
    superposition = analog_signal_1 + analog_signal_2 + analog_signal_3
    superposition.evaluate(np.array([1.0]))
    print(superposition)


def test_multiplication():
    analog_signal_1 = AnalogSignal(np.array([1]))
    analog_signal_2 = AnalogSignal(np.array([2]))
    analog_signal_3 = AnalogSignal(np.array([3]))
    modulation_signal = analog_signal_1 * analog_signal_2 * analog_signal_3
    modulation_signal.evaluate(np.array([1.0]))
    print(modulation_signal)
