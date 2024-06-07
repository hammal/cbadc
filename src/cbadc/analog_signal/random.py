"""Random signals."""

from ._analog_signal import _AnalogSignal
import numpy as np


class _FiniteRandomSignal(_AnalogSignal):
    def __init__(self, output_set: tuple, T: float):
        super().__init__()
        self.set = output_set
        self._buffer_size = 1 << 10
        self._buffer_index = 0
        self._values = np.random.choice(self.set, self._buffer_size)
        self._t = 0.0
        self.T = T

    def __str__(self):
        return f"""Random signal with the possible values: {self.set} and period {self.T}"""

    def evaluate(self, t: float) -> float:
        rel_t = t - self._t
        if rel_t >= self.T:
            self._t = t
            self._buffer_index += 1
            if self._buffer_index == self._buffer_size:
                self._values = np.random.choice(self.set, self._buffer_size)
                self._buffer_index = 0
        return self._values[self._buffer_index]


class BinaryReferenceSignal(_FiniteRandomSignal):
    """
    A binary reference signal.

    Parameters
    ----------
    T : `float`
        The period of the signal.
    set : {`float, ...}
        The set of possible values of the signal.

    Example
    -------
    >>> from cbadc.analog_signal import BinaryReferenceSignal
    >>> u = BinaryReferenceSignal(1, 1, 0)

    """

    def __init__(self, T: float, amplitude: float = 1.0, offset: float = 0.0):
        output_set = (-amplitude + offset, amplitude + offset)
        super().__init__(output_set, T)


class TernaryReferenceSignal(_FiniteRandomSignal):
    """
    A ternary reference signal.

    Parameters
    ----------
    T : `float`
        The period of the signal.
    amplitude : `float`, `optional`
        The amplitude of the signal, default is 1.0.
    offset : `float`, `optional`
        The offset of the signal, default is 0.0.

    Example
    -------
    >>> from cbadc.analog_signal import TernaryReferenceSignal
    >>> u = TernaryReferenceSignal(1, 1, 0)

    """

    def __init__(self, T: float, amplitude: float = 1.0, offset: float = 0.0):
        output_set = (-amplitude + offset, offset, amplitude + offset)
        super().__init__(output_set, T)


class GaussianReferenceSignal(_AnalogSignal):
    """
    A Gaussian reference signal.

    Parameters
    ----------
    T : `float`
        The period of the signal.
    mean : `float`, `optional`
        The mean of the signal, default is 0.0.
    std : `float`, `optional`
        The standard deviation of the signal, default is 1.0.

    Example
    -------
    >>> from cbadc.analog_signal import GaussianReferenceSignal
    >>> u = GaussianReferenceSignal(1, 0, 1)

    """

    def __init__(self, T: float, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
        self._t = 0.0
        self.T = T
        self._buffer_size = 1 << 10
        self._buffer_index = 0
        self._values = np.random.randn(self._buffer_size)

    def __str__(self):
        return f"""Gaussian random signal with mean: {self.mean}, standard deviation: {self.std}, and period {self.T}"""

    def evaluate(self, t: float) -> float:
        """
        Evaluate the signal at time :math:`t`.

        Parameters
        ----------
        t : `float`
            the time instance for evaluation.

        Returns
        -------
        float
            The analog signal value
        """
        rel_t = t - self._t
        if rel_t >= self.T:
            self._t = t
            self._buffer_index += 1
            if self._buffer_index == self._buffer_size:
                self._values = np.random.randn(self._buffer_size)
                self._buffer_index = 0
        return self._values[self._buffer_index]
