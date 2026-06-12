"""Sub-band SNR (snr_residual band= / measure_snr) behaviour."""
import numpy as np
import pytest

from cbadc.snr import measure_snr, snr_residual, snr_vs_frequency


def _ref(n=1 << 15, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def test_full_band_matches_time_domain():
    # band covering the whole spectrum (PSD path) ~= band=None (time-domain).
    rng = np.random.default_rng(1)
    ref = _ref()
    err = 1e-3 * rng.standard_normal(ref.size)
    uhat = ref + err
    full = snr_residual(uhat, ref)  # time-domain var ratio
    psd = snr_residual(uhat, ref, band=(0.0, 1.0))  # full spectrum via PSD
    assert psd == pytest.approx(full, abs=0.5)


def test_subband_excludes_edge_concentrated_noise():
    # error power confined to the top of the band (fraction > 0.7 of Nyquist)
    # -> the inner band [0, 0.6] must read much higher SNR than the full band.
    rng = np.random.default_rng(2)
    n = 1 << 15
    ref = _ref(n, seed=3)
    E = np.fft.rfft(rng.standard_normal(n))
    fr = np.fft.rfftfreq(n) / 0.5  # fraction of Nyquist, 0..1
    E[fr < 0.7] = 0.0  # keep only the top band
    edge = np.fft.irfft(E, n)
    uhat = ref + 1e-4 * rng.standard_normal(n) + 0.02 * edge
    full = snr_residual(uhat, ref)
    inner = snr_residual(uhat, ref, band=(0.0, 0.6))
    assert inner > full + 5.0


def test_flat_error_subband_close_to_full():
    rng = np.random.default_rng(4)
    ref = _ref(seed=5)
    uhat = ref + 1e-3 * rng.standard_normal(ref.size)
    full = snr_residual(uhat, ref)
    inner = snr_residual(uhat, ref, band=(0.0, 0.7))
    assert inner == pytest.approx(full, abs=2.0)


def test_hz_and_fraction_agree():
    rng = np.random.default_rng(6)
    fs = 1000.0
    ref = _ref(seed=7)
    uhat = ref + 1e-3 * rng.standard_normal(ref.size)
    frac = snr_residual(uhat, ref, band=(0.0, 0.7))  # fraction of Nyquist
    hz = snr_residual(uhat, ref, band=(0.0, 0.7 * fs / 2), fs=fs)  # Hz
    assert hz == pytest.approx(frac, abs=0.3)


def test_measure_snr_dispatch():
    rng = np.random.default_rng(8)
    ref = _ref(seed=9)
    uhat = ref + 1e-3 * rng.standard_normal(ref.size)
    assert measure_snr(uhat, ref, method="residual") == pytest.approx(
        snr_residual(uhat, ref), abs=1e-9
    )
    f, s = measure_snr(uhat, ref, method="snr_f", fs=1.0)
    assert f.shape == s.shape
    with pytest.raises(ValueError):
        measure_snr(uhat, ref, method="tone")  # missing f_sig/fs
