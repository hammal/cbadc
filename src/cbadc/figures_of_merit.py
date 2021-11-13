"""Figures of merit

This module provides tools to evaluate standard figures of merit as well as
provides an interface to Prof. Boris Murmann's famous `ADC Survey <https://web.stanford.edu/~murmann/adcsurvey.html>`_.
"""
import numpy as np
import pandas as pd

__twenty_log_2 = 20.0 * np.log10(2.0)
__quantization_noise_offset = 20.0 * np.log10(np.sqrt(6.0)/2.0)


def ENOB(SNR: float):
    """Compute the effective number of bits (ENOB).

    Parameters
    ----------
    SNR: `float`
        the SNR expressed in dB.

    Returns
    : `float`
        the effective number of bits.
    """
    return (SNR - __quantization_noise_offset) / __twenty_log_2


def SNR(ENOB: float):
    """Convert effective number of bits into SNR

    Parameters
    ----------
    ENOB: `float`
        effective number of bits

    Returns
    -------
    : `float`
        The corresponding SNR expressed in dB.
    """
    return ENOB * __twenty_log_2 + __quantization_noise_offset


def nyquist_frequency(fs: float):
    """The Nyquist frequency or bandwidth of a sampled signal.

    Parameters
    ----------
    fs: `float`
        the sampling frequency

    Returns
    -------
    : `float`
        the Nyquist frequency.
    """
    return fs / 2.0


def OSR(fs, f_sig):
    """The oversampling ratio (OSR)

    Parameters
    ----------
    fs: `float`
        the sampling frequency
    f_sig: `float`
        the signal bandwidth.

    Returns
    -------
    : `float`
        the oversampling ratio.
    """
    return nyquist_frequency(fs) / f_sig


def FoM_W(P, fs, ENOB):
    """The Walden figure of merit (FoM)

    See  `Walden 1999<https://ieeexplore.ieee.org/document/761034>`_

    Parameters
    ----------
    P: `float`
        the power consumption.
    fs: `float`
        the sampling frequency.
    ENOB: `float`
        effective number of bits.

    Returns
    -------
    The Walden figure of merit.

    """
    return P / (fs * 2 ** ENOB)


def FoM_S(P, fs, SNR):
    """The Schreier figure fo merit.

    From the book `Understanding Delta-Sigma Data Converters`. 

    Parameters
    ----------
    P: `float`
        power consumption.
    fs: `float`
        the sampling frequency.
    SNR: `float`
        the signal-to-noise (SNR) ratio. 

    Returns
    -------
    : `float`
     Schreier's figure of merit.
    """
    return SNR + 10.0 * np.log10(nyquist_frequency(fs) / P)


class Murmann_Survey():

    def __init__(self):
        self._db = pd.concat(
            pd.read_excel(
                "https://web.stanford.edu/~murmann/publications/ADCsurvey_rev20210628.xls", sheet_name=['ISSCC', 'VLSI']
            )
        )
