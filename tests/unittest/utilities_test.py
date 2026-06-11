"""Tests for pure helpers in :mod:`cbadc.utilities`.

Plotting and file/URL IO helpers are intentionally not covered here; these
tests target the deterministic numeric/encoding helpers.
"""

import numpy as np
import pytest

from cbadc.utilities import (
    FixedPoint,
    byte_stream_2_control_signal,
    compute_power_spectral_density,
    control_signal_2_byte_stream,
    number_of_bytes_selector,
    snr_spectrum_computation,
)


@pytest.mark.parametrize(
    "M, n_bytes, marker",
    [(4, 1, "B"), (8, 1, "B"), (10, 2, "h"), (64, 4, "i"), (128, 8, "q")],
)
def test_number_of_bytes_selector(M, n_bytes, marker):
    fmt = number_of_bytes_selector(M)
    assert fmt["number_of_bytes"] == n_bytes
    assert fmt["format_marker"] == marker


def test_number_of_bytes_selector_too_large_raises():
    with pytest.raises(Exception):
        number_of_bytes_selector(200)


def test_control_signal_byte_stream_round_trip():
    M = 3
    control_signal = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 1]])
    recovered = list(
        byte_stream_2_control_signal(
            control_signal_2_byte_stream(control_signal, M), M
        )
    )
    assert len(recovered) == control_signal.shape[0]
    for original, got in zip(control_signal, recovered):
        np.testing.assert_array_equal(got, original)


def test_fixed_point_round_trip_within_quantisation():
    fp = FixedPoint(number_of_bits=16, max=1.0)
    quantum = fp.fixed_to_float(1)
    for value in (0.0, 0.25, -0.5, 0.99):
        recovered = fp.fixed_to_float(fp.float_to_fixed(value))
        assert recovered == pytest.approx(value, abs=quantum)


def test_fixed_point_overflow_raises():
    fp = FixedPoint(number_of_bits=8, max=1.0)
    with pytest.raises(ArithmeticError):
        fp.float_to_fixed(2.0)


def test_psd_peaks_at_signal_frequency():
    fs = 1e4
    f_sig = 500.0
    n = 1 << 14
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * f_sig * t)
    freq, psd = compute_power_spectral_density(x, nperseg=1 << 12, fs=fs)
    peak_freq = freq[np.argmax(psd)]
    # peak should land within one FFT bin of the true frequency
    assert peak_freq == pytest.approx(f_sig, abs=freq[1] - freq[0])


def test_snr_spectrum_computation():
    spectrum = np.array([10.0, 0.0, 1.0, 1.0])
    signal_mask = np.array([0])
    noise_mask = np.array([2, 3])
    assert snr_spectrum_computation(spectrum, signal_mask, noise_mask) == pytest.approx(
        5.0
    )
    # zero noise -> infinite SNR
    assert np.isinf(
        snr_spectrum_computation(spectrum, signal_mask, np.array([1]))
    )
