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
        self.piecewise_constant = True

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
        self._values = np.random.randn(self._buffer_size) * self.std + self.mean

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
                self._values = np.random.randn(self._buffer_size) * self.std + self.mean
                self._buffer_index = 0
        return self._values[self._buffer_index]


class UniformReferenceSignal(_AnalogSignal):
    """
    A Gaussian reference signal.

    Parameters
    ----------
    T : `float`
        The period of the signal.
    low : `float`
        the lower bound of the signal.
    high : `float`
        the upper bound (non inclusive) of the signal.
    offset : `float`, `optional`
        The offset of the signal, default is 0.0.
    seed : `int`, `optional`
        The seed for the random number generator.
    Example
    -------
    >>> from cbadc.analog_signal import UniformReferenceSignal
    >>> u = UniformReferenceSignal(1, 0, 1)

    """

    def __init__(
        self, T: float, low: float, high: float, offset: float = 0.0, seed: int = 42
    ):
        self.low = low
        self.high = high
        self.offset = offset
        self._t = 0.0
        self.T = T
        self._buffer_size = 1 << 10
        self._buffer_index = 0
        self.rng = np.random.default_rng(seed)
        self._values = self.rng.uniform(
            self.low + self.offset,
            self.high + self.offset,
            self._buffer_size,
        )
        self.piecewise_constant = True

    def __str__(self):
        return f"""Uniform random signal with values between: [{self.low+self.offset}, {self.high + self.offset}), and period {self.T}"""

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
        if rel_t > self.T:
            # if np.isclose(rel_t, self.T, atol=self.T * 1e-5, rtol=1e-10):
            self._t += self.T
            self._buffer_index += 1
            if self._buffer_index == self._buffer_size:
                self._values = self.rng.uniform(
                    self.low + self.offset, self.high + self.offset, self._buffer_size
                )
                self._buffer_index = 0
        return self._values[self._buffer_index]

    def tick(self):
        self.evaluate(self.T + self._t + 1e-1)
