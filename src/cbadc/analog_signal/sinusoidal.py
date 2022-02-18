"""Sinusoidal signals."""
from typing import Union
import numpy as np
from ._analog_signal import _AnalogSignal
from sympy import sin
from mpmath import mp


class Sinusoidal(_AnalogSignal):
    """An analog continuous-time sinusoidal signal.

    Parameters
    ----------
    amplitude : `float`
        The amplitude of the sinusoidal.
    frequency : `float`
        Frequency of the oscillation in [Hz].
    phase : `float`, optional
        The phase, defaults to 0.
    offset : `float`
        The offset value.

    Attributes
    ----------
    amplitude : `float`
        The amplitude.
    frequency : `float`
        The frequency in [Hz].
    angularFrequency : `float`
        The frequency in [radians/second].
    phase : `float`
        The phase.
    offset : `float`, `optional`
        The offset

    See also
    --------
    cbadc.analog_signal.AnalogSignal

    Example
    -------
    >>> from cbadc.analog_signal import Sinusoidal
    >>> import numpy as np
    >>> u = Sinusoidal(1, 123, np.pi/2, 0)
    >>> print(u.evaluate(0))
    1.0

    """

    amplitude: float
    angularFrequency: float
    phase: float
    offset: float

    def __init__(
        self,
        amplitude: float,
        frequency: float,
        phase: float = 0.0,
        offset: float = 0.0,
    ):
        super().__init__()
        self.amplitude: float = amplitude
        self.frequency: float = frequency
        self.angularFrequency: float = 2 * np.pi * self.frequency
        self.phase: float = phase
        self.offset: float = offset
        self._mpmath_dic = {
            'amplitude': mp.mpmathify(amplitude),
            'frequency': mp.mpmathify(frequency),
            'angularFrequency': mp.mpmathify('2') * mp.pi * mp.mpmathify(frequency),
            'phase': mp.mpmathify(phase),
            'offset': mp.mpmathify(offset),
        }

    def __str__(self):
        return f"""
Sinusoidal parameterized as:
amplitude = {self.amplitude},
frequency = {self.frequency},
phase = {self.phase},
and
offset = {self.offset}
        """

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
        return (
            self.amplitude * np.sin(self.angularFrequency * t + self.phase)
            + self.offset
        )

    def _mpmath(self, t: Union[mp.mpf, float]):
        t = mp.mpmathify(t)
        return (
            self._mpmath_dic['amplitude']
            * mp.sin(
                self._mpmath_dic['angularFrequency'] * t + self._mpmath_dic['phase']
            )
            + self._mpmath_dic['offset']
        )

    def symbolic(self) -> sin:
        """Returns as symbolic exression

        Returns
        -------
        : :py:class:`sympy.Symbol`
            a sinusodial function
        """
        return (
            self.amplitude
            * sin(self.angularFrequency * self.t + self.phase + self.sym_phase)
            + self.offset
        )
