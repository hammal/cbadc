"""Sinusoidal signals."""
from typing import Iterable
from ._analog_signal import _AnalogSignal
from sympy import sin
from mpmath import mp
import numpy as np


class ZeroOrderHold(_AnalogSignal):
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

    _iterator: Iterable[float]
    T: float
    t0: float
    _value: float

    def __init__(self, sequence: Iterable[float], T: float):
        super().__init__()
        self._iterator = sequence
        self.T = T
        self.t0 = 0.0
        self._value = 0.0

    def __str__(self):
        return f"""Zero-order-hold signal with period {self.T}"""

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
        if t - self.t0 > self.T:
            self.t0 += self.T
            self._value = next(self._iterator)

        return self._value

    def __call__(self, sequence: Iterable):
        self._iterator = sequence
        self._value = next(self._iterator)
        return self
