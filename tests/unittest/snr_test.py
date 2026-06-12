"""Tests for :mod:`cbadc.snr`."""

import numpy as np
import pytest

from cbadc import snr


def _tone(fs, f, n, noise_rms=0.0, seed=0):
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    return np.cos(2 * np.pi * f * t) + noise_rms * rng.standard_normal(n)


def test_snr_tone_clean_tone_is_high():
    # a coherent (integer-cycle) clean tone projects to ~machine precision
    fs, n = 1e3, 1 << 14
    f = 800 * fs / n  # integer number of cycles in the record
    y = _tone(fs, f, n, noise_rms=0.0)
    assert snr.snr_tone(y, f, fs) > 200  # essentially noiseless


def test_snr_tone_matches_known_noise_floor():
    # signal power 0.5 (unit amplitude cos), noise variance noise_rms**2
    # -> SNR ≈ 10log10(0.5 / noise_rms**2)
    noise_rms = 1e-3
    y = _tone(1e3, 50.0, 1 << 15, noise_rms=noise_rms)
    expected = 10 * np.log10(0.5 / noise_rms**2)
    assert snr.snr_tone(y, 50.0, 1e3) == pytest.approx(expected, abs=1.0)


def test_snr_tone_leakage_free_off_grid():
    # a non-integer number of cycles still reads correctly (no window leakage)
    fs, n = 1e3, 1 << 14
    f = 50.0 * (1 + 1e-3)  # deliberately off the FFT grid
    y = _tone(fs, f, n, noise_rms=1e-4)
    assert snr.snr_tone(y, f, fs) > 70


def test_snr_residual_recovers_known_ratio():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(1 << 15)
    err_rms = 1e-2
    u_hat = ref + err_rms * rng.standard_normal(ref.size)
    expected = 10 * np.log10(np.var(ref) / err_rms**2)
    assert snr.snr_residual(u_hat, ref) == pytest.approx(expected, abs=0.5)


def test_snr_vs_frequency_flat_for_white_error():
    rng = np.random.default_rng(2)
    fs = 1e3
    ref = rng.standard_normal(1 << 16)
    u_hat = ref + 1e-2 * rng.standard_normal(ref.size)
    f, snr_f = snr.snr_vs_frequency(u_hat, ref, fs)
    assert f[0] == 0.0 and f[-1] == pytest.approx(fs / 2)
    # white reference + white error -> roughly flat SNR(f)
    assert np.std(snr_f) < 3.0


def test_coherent_frequency_lands_on_fft_bin():
    fs, n = 1e3, 1 << 12
    f = snr.coherent_frequency(50.0, fs, n)
    cycles = f * n / fs
    assert cycles == pytest.approx(round(cycles))  # integer cycles
    assert abs(f - 50.0) < fs / n  # within one bin of the target
    # a coherent tone is a single FFT bin -> windowed and bin SNR now agree
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    y = np.cos(2 * np.pi * f * t) + 1e-3 * rng.standard_normal(n)
    assert snr.snr_tone(y, f, fs) == pytest.approx(
        10 * np.log10(0.5 / 1e-3**2), abs=1.5
    )


def test_snr_tone_is_trim_robust():
    # with a long enough record, the in-band SNR must not depend on trim
    fs, n = 1e3, 1 << 15
    f = snr.coherent_frequency(50.0, fs, n)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    y = np.cos(2 * np.pi * f * t) + 1e-3 * rng.standard_normal(n)
    vals = [snr.snr_tone(y, f, fs, trim=tr, band=fs / 4) for tr in (0, 500, 2000, 5000)]
    assert max(vals) - min(vals) < 1.0  # flat to within 1 dB


def test_snr_tone_handles_parallel_columns():
    fs, n = 1e3, 1 << 15
    f = snr.coherent_frequency(50.0, fs, n)
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    cols = np.stack(
        [np.cos(2 * np.pi * f * t) + 1e-3 * rng.standard_normal(n) for _ in range(4)],
        axis=-1,
    )[:, None, :]  # (size, L=1, J=4)
    s = snr.snr_tone(cols, f, fs, band=fs / 4)
    # close to the single-column value (~57 dB), not a flattened-garbage number
    assert 50 < s < 65


def test_snr_residual_handles_parallel_columns():
    rng = np.random.default_rng(2)
    ref = rng.standard_normal((1 << 14, 1, 3))
    u_hat = ref + 1e-2 * rng.standard_normal(ref.shape)
    expected = 10 * np.log10(np.var(ref) / 1e-2**2)
    assert snr.snr_residual(u_hat, ref) == pytest.approx(expected, abs=0.5)


def test_decimate_reduces_length():
    x = np.ones((1 << 12, 1, 1))
    y = snr.decimate(x, 4)
    assert y.shape[0] == pytest.approx(x.shape[0] // 4, abs=2)
