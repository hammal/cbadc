"""Submodule containing functions for decimation and demodulation of signals"""
import scipy.signal
import numpy as np


def decimate(signal: np.ndarray, DSR: int, axis: int = 0) -> np.ndarray:
    """
    Decimate the signal by a factor of DSR

    Parameters
    ----------
    signal : array_like
        The signal to decimate
    DSR : int
        The decimation factor
    axis : int, optional
        The axis along which to decimate the signal, by default 0
    """
    return scipy.signal.resample(signal, signal.shape[axis] // DSR, axis=axis)


def demodulate(signal: np.ndarray, Omega: float, axis: int = 0) -> np.ndarray:
    """
    Demodulate the signal

    Parameters
    ----------
    signal : array_like
        The signal to demodulate
    Omega : float
        The modulation frequency
    axis : int, optional
        The axis along which to demodulate the signal, by default 0
    """
    modulation_sequence = np.exp(-1j * Omega * np.arange(signal.shape[axis]))
    # We rely on broadcasting here. This is clearly problematic if:
    # - signal has multiple dimensions of same shape and axis is not the right most one.
    # TODO can we make this more robust?
    return signal * modulation_sequence
