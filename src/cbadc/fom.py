"""Figures of merit: SNR and ENOB conversions, the Walden and Schreier FoMs."""

import numpy as np

_dB_per_bit = 20.0 * np.log10(2.0)  # 6.02
_quantization_offset_dB = 20.0 * np.log10(np.sqrt(6.0) / 2.0)  # 1.76


def snr_to_enob(SNR: float) -> float:
    """``(SNR - 1.76) / 6.02`` bits from an SNR in dB.

    Parameters
    ----------
    SNR : float or ndarray of float
        Signal-to-noise ratio in dB.

    >>> round(float(snr_to_enob(74.0)), 2)
    12.0

    Returns
    -------
    float or ndarray of float
        Effective number of bits, matching the input shape.
    """
    return (SNR - _quantization_offset_dB) / _dB_per_bit


def enob_to_snr(ENOB: float) -> float:
    """``6.02 ENOB + 1.76`` dB.

    Parameters
    ----------
    ENOB : float or ndarray of float
        Effective number of bits.

    >>> round(float(enob_to_snr(12)), 1)
    74.0

    Returns
    -------
    float or ndarray of float
        Signal-to-noise ratio in dB, matching the input shape.
    """
    return ENOB * _dB_per_bit + _quantization_offset_dB


def snr_to_dB(snr: float) -> float:
    """``10 log10(snr)``.

    Parameters
    ----------
    snr : float or ndarray of float
        Positive linear power ratio.

    >>> float(snr_to_dB(1000.0))
    30.0

    Returns
    -------
    float or ndarray of float
        Power ratio in dB, matching the input shape.
    """
    return 10.0 * np.log10(snr)


def snr_from_dB(snr: float) -> float:
    """``10 ** (snr / 10)``.

    Parameters
    ----------
    snr : float or ndarray of float
        Power ratio in dB.

    >>> snr_from_dB(30.0)
    1000.0

    Returns
    -------
    float or ndarray of float
        Linear power ratio, matching the input shape.
    """
    return 10 ** (snr / 10.0)


def nyquist_frequency(fs: float) -> float:
    """``fs / 2``.

    Parameters
    ----------
    fs : float
        Positive sampling frequency in Hz.

    Returns
    -------
    float
        Nyquist frequency in Hz.
    """
    return fs / 2.0


def OSR(fs: float, f_sig: float) -> float:
    """Oversampling ratio ``fs / (2 f_sig)``.

    Parameters
    ----------
    fs : float
        Positive sampling frequency in Hz.
    f_sig : float
        Positive signal bandwidth in Hz.

    >>> OSR(64e6, 1e6)
    32.0

    Returns
    -------
    float
        Dimensionless oversampling ratio.
    """
    return nyquist_frequency(fs) / f_sig


def FoM_W(P: float, fs: float, ENOB: float) -> float:
    """Walden's ``P / (fs 2**ENOB)`` [J/conversion-step] (Walden 1999).

    Parameters
    ----------
    P : float
        Positive power consumption in watts.
    fs : float
        Positive sampling frequency in Hz.
    ENOB : float
        Effective number of bits.

    >>> f"{FoM_W(1e-3, 1e8, 12):.2e}"
    '2.44e-15'

    Returns
    -------
    float
        Walden figure of merit in joules per conversion step.
    """
    return P / (fs * 2**ENOB)


def FoM_S(P: float, fs: float, SNR: float) -> float:
    """Schreier's ``SNR + 10 log10(fs / (2 P))`` [dB] (Schreier & Temes).

    Parameters
    ----------
    P : float
        Positive power consumption in watts.
    fs : float
        Positive sampling frequency in Hz.
    SNR : float
        Signal-to-noise ratio in dB.

    >>> float(FoM_S(1e-3, 2e6, 90.0))
    180.0

    Returns
    -------
    float
        Schreier figure of merit in dB.
    """
    return SNR + 10.0 * np.log10(nyquist_frequency(fs) / P)
