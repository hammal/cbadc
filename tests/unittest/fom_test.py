"""Tests for the figure-of-merit conversions in :mod:`cbadc.fom`.

The ``MurmannSurvey`` class downloads a spreadsheet from Stanford and is
intentionally left untested (network / IO). These tests cover the pure
conversion and figure-of-merit formulas.
"""

import numpy as np
import pytest

from cbadc import fom


def test_enob_snr_round_trip():
    for enob in (0.0, 1.0, 8.0, 13.5):
        assert fom.snr_to_enob(fom.enob_to_snr(enob)) == pytest.approx(enob)


def test_enob_to_snr_known_slope_and_offset():
    # SNR ≈ 6.02 * ENOB + 1.76 dB
    assert fom.enob_to_snr(0.0) == pytest.approx(1.7609, abs=1e-3)
    assert fom.enob_to_snr(1.0) - fom.enob_to_snr(0.0) == pytest.approx(
        20.0 * np.log10(2.0)
    )


def test_snr_db_round_trip():
    for snr in (1.0, 10.0, 1234.0):
        assert fom.snr_from_dB(fom.snr_to_dB(snr)) == pytest.approx(snr)


def test_snr_to_dB_known_value():
    assert fom.snr_to_dB(10.0) == pytest.approx(10.0)
    assert fom.snr_to_dB(100.0) == pytest.approx(20.0)


def test_nyquist_and_osr():
    assert fom.nyquist_frequency(1e6) == pytest.approx(5e5)
    assert fom.OSR(1e6, 1e3) == pytest.approx(500.0)


def test_walden_fom():
    P, fs, enob = 1e-3, 1e6, 10.0
    assert fom.FoM_W(P, fs, enob) == pytest.approx(P / (fs * 2**enob))


def test_schreier_fom():
    P, fs, snr = 1e-3, 1e6, 80.0
    assert fom.FoM_S(P, fs, snr) == pytest.approx(
        snr + 10.0 * np.log10(fs / 2.0 / P)
    )
