"""Figures of merit

This module provides tools to evaluate standard figures of merit as well as
provides an interface to Prof. Boris Murmann's famous `ADC Survey <https://web.stanford.edu/~murmann/adcsurvey.html>`_.
"""

import logging
import os
from typing import List, Tuple
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors

logger = logging.getLogger(__name__)

__twenty_log_2 = 20.0 * np.log10(2.0)
__quantization_noise_offset = 20.0 * np.log10(np.sqrt(6.0) / 2.0)


def snr_to_enob(SNR: float):
    """Compute the effective number of bits (ENOB).

    :math:`f(\\text{SNR}) \\approx \\frac{\\text{SNR} - 1.76}{6.02}`

    Parameters
    ----------
    SNR: `float`
        the SNR expressed in dB.

    Returns
    : `float`
        the effective number of bits.
    """
    return (SNR - __quantization_noise_offset) / __twenty_log_2


def snr_to_dB(snr: float):
    """Convert snr to dB

    :math:`f(\\text{snr}) = 10 \log(\\text{snr})`

    Parameters
    ----------
    snr: `float`
        the snr not expressed in dB
    Returns
    -------
    : `float`
        the SNR expressed in dB
    """
    return 10.0 * np.log10(snr)


def snr_from_dB(snr: float):
    """Convert SNR from dB

    :math:`f(\\text{snr}) = 10^{ \\frac{\\text{snr}}{10}}`

    Parameters
    ----------
    snr: `float`
        the snr expressed in dB
    Returns
    -------
    : `float`
        the SNR not expressed in dB
    """
    return 10 ** (snr / 10.0)


def enob_to_snr(ENOB: float):
    """Convert effective number of bits into SNR

    :math:`f(\\text{ENOB}) \\approx 6.02 \\cdot \\text{ENOB} + 1.76`

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

    :math:`f(f_s)=f_s / 2`

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

    :math:`f(f_s, f_{\\text{sig}}) = \\frac{f_s}{2 \\cdot f_{\\text{sig}}}`

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

    :math:`f(P, f_s, \\text{ENOB}) = \\frac{P}{f_s \\cdot 2 ^{ \\text{ENOB}}}`

    See  `Walden 1999 <https://ieeexplore.ieee.org/document/761034>`_

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
    return P / (fs * 2**ENOB)


def FoM_S(P, fs, SNR):
    """The Schreier figure fo merit.

    :math:`f(P, f_s, \\text{SNR}) = \\text{SNR} + 10 \log\\left(\\frac{f_s}{2 \cdot P}\\right)`

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


class MurmannSurvey:
    """Data container for Murmann's ADC survey

    A convenience class for downloading and parsing
    the `Murmann ADC survey <https://web.stanford.edu/~murmann/adcsurvey.html>`_
    into a pandas dataframe instance.

    Note that if the ADC survey excel file is not present in
    the working directory upon instantiating this class
    it is automatically downloaded.

    Attributes
    ----------
    db: :py:class:`pandas.DataFrame`
        a pandas data frame containing the full Murmann survey.
    """

    def __init__(self):
        self._version = [
            # "https://web.stanford.edu/~murmann/publications/ADCsurvey_rev20210628.xls"
            "https://github.com/bmurmann/ADC-survey/blob/main/xls/ADCsurvey_latest.xlsx"
        ]

        self._current_year = 2021

        self._FoMW_hf_corner = 4.42e8
        self._FoMW_hf_overall = 0.67
        self._FoMS_hf_corner = 3.19e7
        self._FoMS_hf_overall = 184.76

        filename = os.path.basename(self._version[-1])

        # Download from Standford if not present.
        if not os.path.isfile(filename):
            logging.info("Downloading Murmann Survey.")
            req = requests.get(self._version[-1], allow_redirects=True)
            with open(filename, "wb") as f:
                f.write(req.content)
        else:
            logging.info(f"Found local version of {filename}")

        _temp = pd.read_excel(filename, sheet_name=["ISSCC", "VLSI"])
        _temp["ISSCC"]["CONFERENCE"] = "ISSCC"
        _temp["VLSI"]["CONFERENCE"] = "VLSI"
        self.db = pd.concat(_temp, ignore_index=True)
        self._architecture = pd.unique(self.db["ARCHITECTURE"])
        self._color_map = matplotlib.cm.viridis
        self._color_list = [
            matplotlib.colors.rgb2hex(self._color_map(i))
            for i in np.linspace(0, 0.9, len(self._architecture))
        ]

        # Fix some data problems
        self.db["AREA [mm^2]"] = pd.to_numeric(self.db["AREA [mm^2]"], errors="coerce")

    def columns(self) -> List[str]:
        """Returns the columns of the dataframe

        Returns
        -------
        :[str]
            all valid columns of the parsed survey.
        """
        return self.db.columns

    def architectures(self) -> List[str]:
        """Returns the present architectures of the
        survey.

        Returns
        -------
        :[str]
            a list containing the architectures.
        """
        return self._architecture

    def energy(self):
        """Create the Murmann energy plot

        Creates a matplotlib scatter plot
        corresponding to the one found in the ADC survey.

        Returns
        -------
        : :py:class:`matplotlib.axes.Axes`
            the figure axis.
        """
        plt.figure()
        ax = plt.gca()

        # Plot from Murmann survey
        self._Murmann_style_data_and_legends("SNDR_hf [dB]", "P/fsnyq [pJ]", ax)

        # Plot FoM lines
        _x = [40, 120]
        _y = [1e12 * self._FoMW_SNDR_to_p_fs(1e-15, x) for x in _x]
        ax.plot(_x, _y, "--", color="green", label="FoMW 1fJ/conv-step")
        _x = [40, 120]
        _y = [1e12 * self._FoMS_SNDR_to_p_fs(185, x) for x in _x]
        ax.plot(_x, _y, color="green", label="FoMS=185dB")

        # Estetics
        _ = ax.legend()
        ax.set_yscale("log")
        ax.set_title("Energy")
        ax.grid(True, which="both")
        ax.set_xlim((10, 120))
        ax.set_ylim((1e-1, 1e7))
        return ax

    def aperture(self):
        """Create the Murmann aperture plot

        Creates a matplotlib scatter plot
        corresponding to the one found in the ADC survey.

        Returns
        -------
        : :py:class:`matplotlib.axes.Axes`
            the figure axis.
        """
        plt.figure()
        ax = plt.gca()

        # Plot from Murmann survey.
        self._Murmann_style_data_and_legends("SNDR_hf [dB]", "fin_hf [Hz]", ax)

        # Jitter lines
        _y = [1e6, 1e11]
        _x = [self._f_sigma_to_jitter_sndr(1e-12, y) for y in _y]
        ax.plot(_x, _y, color="red", label="Jitter=1psrms")
        _y = [1e6, 1e11]
        _x = [self._f_sigma_to_jitter_sndr(1e-13, y) for y in _y]
        ax.plot(_x, _y, "--", color="red", label="Jitter=0.1psrms")

        # Estetics
        _ = ax.legend()
        ax.set_yscale("log")
        ax.set_title("Aperture")
        ax.grid(True, which="both")
        ax.set_xlim((10, 120))
        ax.set_ylim((1e6, 1e11))
        return ax

    def walden_vs_speed(self):
        """Create the Murmann walden FoM vs speed plot

        Creates a matplotlib scatter plot
        corresponding to the one found in the ADC survey.

        Returns
        -------
        : :py:class:`matplotlib.axes.Axes`
            the figure axis.
        """
        plt.figure()
        ax = plt.gca()

        # Plot from Murmann Survey
        self._Murmann_style_data_and_legends("fsnyq [Hz]", "FOMW_hf [fJ/conv-step]", ax)

        # Envelope
        _x = np.logspace(3, 12, 100)
        _y = [self._FoMW_envelope(x) for x in _x]
        ax.plot(_x, _y, "--", color="black", label="Envelope")

        # Estetics
        _ = ax.legend()
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title("Walden's FoM vs Speed")
        ax.grid(True, which="both")
        ax.set_xlim((1e4, 5e11))
        ax.set_ylim((2e-1, 2e4))
        return ax

    def schreier_vs_speed(self):
        """Create the Murmann Schreier FoM vs speed plot

        Creates a matplotlib scatter plot
        corresponding to the one found in the ADC survey.

        Returns
        -------
        : :py:class:`matplotlib.axes.Axes`
            the figure axis.
        """
        plt.figure()
        ax = plt.gca()

        # Plot from Murmann Survey
        self._Murmann_style_data_and_legends("fsnyq [Hz]", "FOMS_hf [dB]", ax)

        # Envelope
        _x = np.logspace(2, 12, 100)
        _y = [self._FoMS_envelope(x) for x in _x]
        ax.plot(_x, _y, "--", color="black", label="Envelope")

        # Estetics
        _ = ax.legend()
        ax.set_xscale("log")
        ax.set_title("Schreier's FoM vs Speed")
        ax.grid(True, which="both")
        ax.set_xlim((1e2, 1e12))
        ax.set_ylim((120, 190))
        return ax

    def select_bw_and_enob(self, BW: Tuple[float, float], ENOB: Tuple[float, float]):
        """Select publications with a specific bandwidth
        and ENOB range.

        Specifically, return all publications where
        BW[0] <= nyquist frequency < BW[1],
        ENOB[0] <= effective number of bits < ENOB[1].

        Parameters
        ----------
        BW: [float, float]
            a lower and upper bandwidth range.
        ENOB: [float, float]
            a lower and upper effective number of bits (ENOB) range.

        Returns
        : :py:class:`pandas.DataFrame`
            a new pandas dataframe with the selected subset.
        """

        if BW[0] > BW[1]:
            raise Exception("BW must be a tuple with accsending values like (1e6, 1e8)")
        if ENOB[0] > ENOB[1]:
            raise Exception("ENOB must be a tuple with accsending values like (8, 10)")

        return self.db[
            (self.db["fsnyq [Hz]"] >= BW[0])
            & (self.db["fsnyq [Hz]"] < BW[1])
            & (self.db["SNR [dB]"] >= enob_to_snr(ENOB[0]))
            & (self.db["SNR [dB]"] < enob_to_snr(ENOB[1]))
        ]

    def _Murmann_style_data_and_legends(self, x, y, ax):
        self.db[
            (self.db["CONFERENCE"] == "ISSCC") & (self.db["YEAR"] == self._current_year)
        ].plot.scatter(
            x, y, label=f"ISSCC {self._current_year}", color="red", marker="s", ax=ax
        )
        self.db[
            (self.db["CONFERENCE"] == "VLSI") & (self.db["YEAR"] == self._current_year)
        ].plot.scatter(
            x, y, label=f"VLSI {self._current_year}", color="blue", marker="D", ax=ax
        )
        self.db[
            (self.db["CONFERENCE"] == "ISSCC") & (self.db["YEAR"] < self._current_year)
        ].plot.scatter(
            x,
            y,
            label=f"ISSCC 1997-{self._current_year - 1}",
            color="black",
            marker="o",
            ax=ax,
        )
        self.db[
            (self.db["CONFERENCE"] == "VLSI") & (self.db["YEAR"] < self._current_year)
        ].plot.scatter(
            x,
            y,
            label=f"VLSI 1997-{self._current_year - 1}",
            color="black",
            marker="x",
            ax=ax,
        )

    def _f_sigma_to_jitter_sndr(self, f, sigma):
        return -20.0 * np.log10(2 * np.pi * sigma * f)

    def _FoMW_SNDR_to_p_fs(self, FoMW, SNDR):
        ENOB = snr_to_enob(SNDR)
        return FoMW * 2**ENOB

    def _FoMS_SNDR_to_p_fs(self, FoMS, SNDR):
        return (10 ** (-(FoMS - SNDR) / 10.0)) / 2.0

    def _FoMW_envelope(self, f):
        return self._FoMW_hf_overall * np.sqrt(1.0 + (f / self._FoMW_hf_corner) ** 2)

    def _FoMS_envelope(self, f):
        return self._FoMS_hf_overall - 10.0 * np.log10(
            np.sqrt(1.0 + (f / self._FoMS_hf_corner) ** 2)
        )
