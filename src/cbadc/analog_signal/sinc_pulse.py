"""The sinc signal."""
from typing import Union
import numpy as np
from ._analog_signal import _AnalogSignal
from sympy import sinc
from mpmath import mp


class SincPulse(_AnalogSignal):

    """An analog continuous-time sinc pulse.

    Parameters
    ----------
    amplitude : `float`
        The peak amplitude of the pulse.
    bandwidth : `float`
        The bandwidth in [Hz].
    t0 : `float`
        The time delay (instance of the peak) in [s].
    offset : `float`
        The offset value, defaults to 0.

    Attributes
    ----------
    amplitude : `float`
        The amplitude.
    bandwidth : `float`
        The bandwidth in [Hz].
    t0 : `float`
        The the time delay in [s].
    offset : `float`, `optional`
        The offset

    See also
    --------
    cbadc.analog_signal.AnalogSignal

    Example
    -------
    >>> from cbadc.analog_signal import SincPulse
    >>> import numpy as np
    >>> u = SincPulse(3, 1, 5)
    >>> print(u.evaluate(5))
    3.0

    """

    def __init__(
        self, amplitude: float, bandwidth: float, t0: float, offset: float = 0.0
    ):
        super().__init__()
        self.amplitude: float = amplitude
        self.bandwidth: float = bandwidth
        self.t0: float = t0
        self.offset: float = offset
        self._mpmath_dic = {
            'amplitude': mp.mpmathify(amplitude),
            'bandwidth': mp.mpmathify(bandwidth),
            't0': mp.mpmathify(t0),
            'offset': mp.mpmathify(offset),
        }

    def __str__(self):
        return f"""Sinc pulse parameterized as: t0 = {self.t0}, \n
        bandwidth = {self.bandwidth}, peak amplitude = {self.amplitude},
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
        return (
            self.amplitude * np.sinc(2.0 * self.bandwidth * (t - self.t0)) + self.offset
        )

    def _mpmath(self, t: Union[mp.mpf, float]):
        t = mp.mpmathify(t)
        x = (
            mp.mpf('2')
            * mp.pi
            * self._mpmath_dic['bandwidth']
            * (t - self._mpmath_dic['t0'])
        )
        return self._mpmath_dic['amplitude'] * mp.sin(x) / (x)

    def symbolic(self) -> sinc:
        """Returns as symbolic exression


        Returns
        -------
        : :py:class:`sympy.Symbol`
            a symbolic sinc_pulse function
        """
        return (
            self.amplitude * sinc(2 * self.bandwidth * (self.t - self.t0)) + self.offset
        )
