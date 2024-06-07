from cbadc.analog_signal import (
    BinaryReferenceSignal,
    GaussianReferenceSignal,
    TernaryReferenceSignal,
)


def test_binary_reference_signal():
    T = 1.0
    amplitude = 1.0
    offset = 0.0
    BinaryReferenceSignal(T, amplitude, offset)


def test_ternary_reference_signal():
    T = 1.0
    amplitude = 1.0
    offset = 0.0
    TernaryReferenceSignal(T, amplitude, offset)


def test_gaussian_reference_signal():
    T = 1.0
    mean = 0.0
    GaussianReferenceSignal(T, mean)
