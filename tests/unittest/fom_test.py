"""Tests for the figure-of-merit conversions in :mod:`cbadc.fom`.

The ``MurmannSurvey`` class downloads a spreadsheet from Stanford and is
intentionally left untested (network / IO). These tests cover the pure
conversion and figure-of-merit formulas.
"""

import numpy as np
import pytest

from cbadc import fom


def test_enob_snr_round_trip():
    """ENOB to SNR and back must recover each input bit count."""
    for enob in (0.0, 1.0, 8.0, 13.5):
        assert fom.snr_to_enob(fom.enob_to_snr(enob)) == pytest.approx(enob)


def test_enob_to_snr_known_slope_and_offset():
    """One extra effective bit adds 20 log10(2) dB above the 1.76 dB offset."""
    # SNR ≈ 6.02 * ENOB + 1.76 dB
    assert fom.enob_to_snr(0.0) == pytest.approx(1.7609, abs=1e-3)
    assert fom.enob_to_snr(1.0) - fom.enob_to_snr(0.0) == pytest.approx(
        20.0 * np.log10(2.0)
    )


def test_snr_db_round_trip():
    """Linear power SNR survives conversion to dB and back."""
    for snr in (1.0, 10.0, 1234.0):
        assert fom.snr_from_dB(fom.snr_to_dB(snr)) == pytest.approx(snr)


def test_snr_to_dB_known_value():
    """Power ratios 10 and 100 correspond to 10 and 20 dB."""
    assert fom.snr_to_dB(10.0) == pytest.approx(10.0)
    assert fom.snr_to_dB(100.0) == pytest.approx(20.0)


def test_nyquist_and_osr():
    """Nyquist is fs/2, and OSR divides that by signal bandwidth."""
    assert fom.nyquist_frequency(1e6) == pytest.approx(5e5)
    assert fom.OSR(1e6, 1e3) == pytest.approx(500.0)


def test_walden_fom():
    """Walden energy per conversion step is P/(fs 2**ENOB)."""
    P, fs, enob = 1e-3, 1e6, 10.0
    assert fom.FoM_W(P, fs, enob) == pytest.approx(P / (fs * 2**enob))


def test_schreier_fom():
    """Schreier FoM combines SNR, Nyquist bandwidth, and power in dB."""
    P, fs, snr = 1e-3, 1e6, 80.0
    assert fom.FoM_S(P, fs, snr) == pytest.approx(snr + 10.0 * np.log10(fs / 2.0 / P))
