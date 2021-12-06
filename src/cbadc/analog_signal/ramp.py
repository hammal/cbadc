from typing import Union
import numpy as np
from ._analog_signal import _AnalogSignal
from sympy import Function
from mpmath import mp


class Ramp(_AnalogSignal):
    """An analog continuous-time ramp signal.
    Parameters
    ----------
    amplitude : `float`
        The amplitude of the sinusoidal.
    period : `float`
        one over the period length of the ramp, specified in [Hz].
    phase : `float`, optional
        The phase offset in [s], defaults to 0.
    offset : `float`
        The offset value.
    Attributes
    ----------
    amplitude : `float`
        The amplitude.
    period : `float`
        The ramp's time period in [s].
    phase : `float`
        The phase offset in [s].
    offset : `float`, `optional`
        The offset
    See also
    --------
    cbadc.analog_signal.AnalogSignal
    """

    def __init__(
        self, amplitude: float, period: float, phase: float = 0.0, offset: float = 0.0
    ):
        self.amplitude: float = amplitude
        self.period: float = period
        self.phase: float = phase
        self.offset: float = offset - self.amplitude / 2.0
        self._mpmath_dic = {
            'amplitude': mp.mpmathify(amplitude),
            'period': mp.mpmathify(period),
            'phase': mp.mpmathify(phase),
            'offset': mp.mpmathify(offset),
        }
        super().__init__()

    def __str__(self):
        return f"""Sinusoidal parameterized as: \namplitude = {self.amplitude}, \n
        frequency = {self.frequency}, \nphase = {self.phase},
        and\noffset = {self.offset}"""

    def evaluate(self, t: float) -> float:
        """Evaluate the signal at time :math:`t`.
        Parameters
        ----------
        t : `float`
            the time instance for evaluation.
        Returns
        -------
        float
            The analog signal value
        """
        return self.amplitude * ((t + self.phase) % self.period) + self.offset

    def _mpmath(self, t: Union[mp.mpf, float]):
        t = mp.mpmathify(t)
        return super()._mpmath(t)

    def symbolic(self) -> Function:
        """Returns as symbolic exression

        Returns
        -------
        : :py:class:`sympy.Symbol`
            a ramp symbolic function
        """
        return self.amplitude * ((t + self.phase) % self.period) + self.offset
