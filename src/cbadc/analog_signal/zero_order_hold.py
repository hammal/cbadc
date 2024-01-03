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

    _data: np.ndarray
    T: float
    t0: float
    _value: float

    def __init__(self, data: np.ndarray, T: float):
        super().__init__()
        self._data = data
        self.T = T

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
        return self._data[int(t / self.T) % self._data.shape[0]]

    def __call__(self, data: np.ndarray):
        self._data = data
        return self


class BandLimitedSignal(_AnalogSignal):
    _data: np.ndarray
    T: float
    t0: float
    _value: float
    W: float

    def __init__(
        self, data: np.ndarray, T: float, alpha: float, number_of_samples: int
    ):
        super().__init__()
        self._data = data
        self._number_of_samples = number_of_samples
        self.T = T
        self.BW = 2 / T
        self.alpha = alpha

    def __str__(self):
        return f"""Zero-order-hold signal with period {self.T}"""

    def _impulse_response(self, t: float) -> float:
        return np.sinc(2 * self.BW * t)

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
        index = int(t / self.T) % self._data.shape[0]
        res = self._data[index]
        for i in range(1, self._number_of_samples // 2):
            if index - i > 0:
                res += self._impulse_response(t - i * self.T) * self._data[index - i]
            if index + i < self._data.shape[0]:
                res += self._impulse_response(t + i * self.T) * self._data[index + i]
        return res

    def __call__(self, data: np.ndarray):
        self._data = data
        return self


class RaisedCosine(_AnalogSignal):
    _data: np.ndarray
    T: float
    t0: float
    _value: float
    W: float

    def __init__(
        self, data: np.ndarray, T: float, alpha: float, number_of_samples: int
    ):
        super().__init__()
        self._data = data
        self._number_of_samples = number_of_samples
        self.T = T
        self.BW = 2 / T
        self.alpha = alpha

    def __str__(self):
        return f"""Zero-order-hold signal with period {self.T}"""

    def _impulse_response(self, t: float) -> float:
        return np.sinc(2 * self.BW * t) * (
            np.cos(2 * np.pi * self.alpha * self.BW * t)
            / (1.0 - (4 * self.alpha * self.BW * t) ** 2)
        )

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
        index = int(t / self.T) % self._data.shape[0]
        res = self._data[index]
        for i in range(1, self._number_of_samples // 2):
            if index - i > 0:
                res += self._impulse_response(t - i * self.T) * self._data[index - i]
            if index + i < self._data.shape[0]:
                res += self._impulse_response(t + i * self.T) * self._data[index + i]
        return res

    def __call__(self, data: np.ndarray):
        self._data = data
        return self
