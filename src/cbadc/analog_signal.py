"""Generic analog signals."""

from typing import Optional

import numpy as np
from scipy.signal import resample as _resample


class AnalogSignal:
    """A default continuous-time analog signal.

    Parameters
    ----------
    offset : `numpy.ndarray`, shape=(L, J), optional
        The offset value, defaults to numpy.zeros((1, 1), dtype=float).

    Attributes
    ----------
    offset : `numpy.ndarray`, shape=(L, J)
        The offset vector.
    L : `int`
        The dimension of the input.
    J: `int`
        The number of inputs, used for parallel simulation.
    piecewise_constant : `bool`
        A flag indicating if the signal is piecewise constant.

    """

    def __init__(self, offset: np.ndarray = None, dtype=float):
        self._dtype = dtype
        if offset is None:
            offset = np.zeros((1, 1), dtype=float)
        if not isinstance(offset, np.ndarray):
            raise TypeError("offset must be a numpy array")
        if offset.ndim > 2:
            raise ValueError("offset must be a 1 or 2-dimensional numpy array")
        elif offset.ndim == 1:
            offset = offset.reshape((-1, 1))
        self.L = offset.shape[0]
        self.J = offset.shape[1]
        self.offset = offset
        self.piecewise_constant = True

    @property
    def dtype(self):
        """The floating-point data type used by the signal.

        Returns
        -------
        dtype
            The numpy dtype (e.g. ``float``, ``np.float32``).
        """
        return self._dtype

    @property
    def L(self) -> int:
        """The dimension of the input.

        Returns
        -------
        int
            The dimension of the input.
        """
        return self._L

    @L.setter
    def L(self, value: int):
        if not isinstance(value, int):
            raise TypeError("L must be an integer")
        if value < 1:
            raise ValueError("L must be a positive integer")
        self._L = value

    @property
    def J(self) -> int:
        """The number of inputs.

        Used for parallel simulation of multiple signals.

        Returns
        -------
        int
            The number of inputs.
        """
        return self._J

    @J.setter
    def J(self, value: int):
        if not isinstance(value, int):
            raise TypeError("J must be an integer")
        if value < 1:
            raise ValueError("J must be a positive integer")
        self._J = value

    @property
    def offset(self) -> np.ndarray:
        """The offset value.

        Returns
        -------
        numpy.ndarray, shape=(L, J)
            The offset vector.
        """
        return self._offset

    @offset.setter
    def offset(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("offset must be a numpy array")
        if value.shape != (self.L, self.J):
            raise ValueError("offset must have shape (L,J)")
        self._offset = value.reshape((self.L, self.J))

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the signal at time t.

        Parameters
        ----------
        t : `numpy.ndarray`, shape=(size,)
            the time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            The analog signal values
        """
        return np.broadcast_to(
            self.offset[np.newaxis], (t.size, self.L, self.J)
        ).astype(self.dtype)

    def impulse_response(self, t: np.ndarray) -> np.ndarray:
        """Impulse response of the signal.

        Used for discretization and is not well defined for
        non LTI systems.

        Parameters
        ----------
        t : `numpy.ndarray`, shape=(size,)
            the time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            The impulse response of the signal
        """
        return np.ones((t.size, self.L, self.J), dtype=self.dtype)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return self.evaluate(t)

    def __add__(self, other):
        return SuperpositionSignal(self, other)

    def __sub__(self, other):
        return SuperpositionSignal(self, other, sub=[False, True])

    def __mul__(self, other):
        return ModulatedSignal(self, other)

    # def __div__(self, other):
    #     raise NotImplementedError

    def concatenate(self, *other):
        """Concatenate signals.

        Parameters
        ----------
        other : `AnalogSignal`
            The signals to be concatenated.

        Returns
        -------
        :py:class:cbadc.analog_signal.ConcatenatedSignals
            The concatenated signals.
        """
        return ConcatenatedSignals(self, *other)

    def __str__(self):
        return f"AnalogSignal(offset={self.offset}, L={self.L}, piecewise_constant={self.piecewise_constant})"


class Sinusoidal(AnalogSignal):
    """An analog continuous-time sinusoidal signal.

    Parameters
    ----------
    amplitude : `numpy.ndarray`, shape=(L, J)
        The amplitude vector of the sinusoidal.
    frequency : `numpy.ndarray`, shape=(L, J)
        Frequency of the oscillation in [Hz].
    phase : `numpy.ndarray`, optional, shape=(L, J)
        The phase, defaults to numpy.zeros((L, 1), dtype=float).
    offset : `numpy.ndarray`, optional, shape=(L, J)
        The offset value, defaults to numpy.zeros((L, 1), dtype=float).

    Attributes
    ----------
    amplitude : :py:class:`numpy.ndarray`, shape=(L, J)
        The amplitude vector.
    frequency : :py:class:`numpy.ndarray`, shape=(L, J)
        The frequency in [Hz].
    phase : :py:class:`numpy.ndarray`, shape=(L, J)
        The phase.
    offset : :py:class:`numpy.ndarray`, optional, shape=(L, J)
        The offset
    L : `int`
        The dimension of the input.
    J: `int`
        The number of inputs, used for parallel simulation.

    """

    def __init__(
        self,
        amplitude: np.ndarray,
        frequency: np.ndarray,
        phase: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        dtype=float,
    ):
        if not isinstance(amplitude, np.ndarray):
            raise TypeError("amplitude must be a numpy array")
        if amplitude.ndim == 1:
            amplitude = amplitude.reshape((-1, 1))
        if offset is None:
            offset = np.zeros_like(amplitude, dtype=dtype)
        super().__init__(offset=offset, dtype=dtype)
        self.amplitude = amplitude
        # angular frequency implicitly calculated in setter.
        self.frequency = frequency
        if phase is None:
            phase = np.zeros((self.L, self.J), dtype=float)
        self.phase = phase
        self.piecewise_constant = False

    @property
    def amplitude(self) -> np.ndarray:
        """The amplitude of the sinusoidal signal.

        Returns
        -------
        `numpy.ndarray`, shape=(L, J)
            The amplitude.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("amplitude must be a numpy array")
        if value.ndim == 1:
            value = value.reshape((-1, 1))
        if value.shape != (self.L, self.J):
            raise ValueError("amplitude must have shape (L, J)")
        self._amplitude = value.reshape((self.L, self.J))

    @property
    def frequency(self) -> np.ndarray:
        """The frequency of the sinusoidal signal.

        Returns
        -------
        `numpy.ndarray`, shape=(L, J)
            The frequency in Hz.
        """
        return self._frequency

    @frequency.setter
    def frequency(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("frequency must be a numpy array")
        if value.ndim == 1:
            value = value.reshape((-1, 1))
        if value.shape != (self.L, self.J):
            raise ValueError("frequency must have shape (L, J)")
        self._frequency = value.reshape((self.L, self.J))
        self._angular_frequency = 2.0 * np.pi * self._frequency

    @property
    def phase(self) -> np.ndarray:
        """The phase of the sinusoidal signal.

        Returns
        -------
        `numpy.ndarray`, shape=(L, J)
            The phase in radians.
        """
        return self._phase

    @phase.setter
    def phase(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("phase must be a numpy array")
        if value.ndim == 1:
            value = value.reshape((-1, 1))
        if value.shape != (self.L, self.J):
            raise ValueError("phase must have shape (L, J)")
        self._phase = value.reshape((self.L, self.J))

    def __str__(self):
        return (
            f"Sinusoidal parameterized as:\n"
            f"amplitude = {self.amplitude},\n"
            f"frequency = {self.frequency},\n"
            f"phase = {self.phase},\n"
            f"offset = {self.offset}"
        )

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the signal at time t.

        Parameters
        ----------
        t : `numpy.ndarray`, shape=(size,)
            the time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            The analog signal values
        """
        result = (
            self.amplitude[np.newaxis, :, :]
            * np.sin(
                self._angular_frequency[np.newaxis, :, :] * t.reshape((-1, 1, 1))
                + self.phase[np.newaxis, :, :]
            )
            + self.offset[np.newaxis, :, :]
        )
        return (
            result
            if result.dtype == np.dtype(self.dtype)
            else result.astype(self.dtype)
        )


class ZeroOrderHold(AnalogSignal):
    """A zero order hold signals.

    Parameters
    ----------
    dt : `float`,
        The sampling period.
    values : `numpy.ndarray`, shape=(size, L, J)
        The corresponding values of the zero-order hold signal where
        the value[i] is valid for the interval [time[i], time[i+1]).
    t0 : `float`, optional
        The initial time, defaults to 0.0.
    """

    def __init__(self, dt: float, values: np.ndarray, t0: float = 0.0, dtype=float):

        super().__init__(np.zeros_like(values[0, :, :], dtype=dtype), dtype=dtype)
        self.dt = dt
        self.t0 = float(t0)
        self.values = values
        self.piecewise_constant = True

    @property
    def dt(self) -> float:
        """The sampling period.

        Returns
        -------
        float
            The sampling period.
        """
        return self._Ts

    @dt.setter
    def dt(self, value: float):
        if not isinstance(value, float):
            raise TypeError("Ts must be a float")
        if value <= 0:
            raise ValueError("Ts must be a positive number")
        self._Ts = value

    @property
    def values(self) -> np.ndarray:
        """The values of the zero-order hold signal.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            The values.
        """
        return self._values

    @values.setter
    def values(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("values must be a numpy array")
        if value.ndim > 3:
            raise ValueError("values must be a 1, 2 or 3-dimensional numpy array")
        if value.ndim < 3:
            value = value.reshape((-1, self.L, self.J))
        if value.shape[1:] != (self.L, self.J):
            raise ValueError(
                f"values must have shape (size, {self.L}, {self.J}) not {value.shape}"
            )
        self.size = value.shape[0]
        self._values = value.astype(self.dtype)

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the signal at time t.)

        Parameters
        ----------
        t : `numpy.ndarray`, shape=(size,)
            the time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            The analog signal values
        """

        tmp_index = np.mod(np.floor((t - self.t0) / self.dt).astype(int), self.size)
        return self._values[tmp_index]

    def impulse_response(self, t: np.ndarray) -> np.ndarray:
        """Impulse response of the signal.

        Used for discretization and is not well defined for
        non LTI systems.

        Parameters
        ----------
        t : `numpy.ndarray`, shape=(size,)
            the time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            The impulse response of the signal
        """
        return np.ones((t.size, self.L, self.J), dtype=self.dtype)

    def resample(self, new_dt: float) -> "ZeroOrderHold":
        """Resample the zero-order hold signal to a new sampling period.

        Parameters
        ----------
        new_dt : `float`
            The new sampling period.

        Returns
        -------
        ZeroOrderHold
            The resampled zero-order hold signal.
        """
        if not isinstance(new_dt, float):
            raise TypeError("new_dt must be a float")
        if new_dt <= 0:
            raise ValueError("new_dt must be a positive number")

        ratio = int(np.round(self.dt / new_dt))
        if not np.isclose(ratio * new_dt, self.dt):
            raise ValueError("new_dt must be an integer multiple of the current dt")
        new_values: np.ndarray = _resample(
            self._values, num=int(ratio * self.size), axis=0
        )
        # new_values = self._values[::ratio, :, :]
        return ZeroOrderHold(new_dt, new_values, self.t0)

    @staticmethod
    def binary_reference_signal(
        dt: float,
        amplitude: Optional[np.ndarray] = None,
        t0: float = 0.0,
        size: int = 1 << 16,
        offset: Optional[np.ndarray] = None,
        seed: int = 23423402967101203431687465321,
    ) -> "ZeroOrderHold":
        """Create a binary reference signal.

        Parameters
        ----------
        dt : `float`
            The sampling period.
        amplitude : `numpy.ndarray`, shape=(L, J), optional
            The amplitude of the signal, defaults to numpy.ones((1, 1), dtype=float).
        t0 : `float`, optional
            The initial time, defaults to 0.0.)
        size : `int`, optional
            The size of the signal, defaults to 1 << 20.
        offset : `numpy.ndarray`, shape=(L,), optional
            The offset value, defaults to numpy.zeros_like(amplitude, dtype=float).
        seed
            The seed for the random number generator.

        Returns
        -------
        ZeroOrderHold
            The zero order hold signal.
        """
        if amplitude is None:
            amplitude = np.ones((1, 1), dtype=float)
        if amplitude.ndim == 1:
            amplitude = amplitude.reshape((-1, 1))
        if offset is None:
            offset = np.zeros_like(amplitude, dtype=float)
        if offset.ndim == 1:
            offset = offset.reshape((-1, 1))
        if amplitude.shape != offset.shape:
            raise ValueError("amplitude and offset must have the same shape")

        rng = np.random.default_rng(seed)
        L, J = amplitude.shape
        # binary_set = np.zeros((2,), dtype=float)
        # binary_set[0] = amplitude + offset
        # binary_set[1] = -amplitude + offset
        values = (rng.choice(2, size=(size, L, J), axis=0) * 2 - 1) * amplitude[
            np.newaxis, :, :
        ] + offset[np.newaxis, :, :]
        print(values.shape)
        return ZeroOrderHold(dt, values, t0)

    @staticmethod
    def ternary_reference_signal(
        dt: float,
        amplitude: Optional[np.ndarray] = None,
        t0: float = 0.0,
        size: int = 1 << 16,
        offset: Optional[np.ndarray] = None,
        seed: int = 890012391238219123057,
    ) -> "ZeroOrderHold":
        """Create a ternary reference signal.

        Parameters
        ----------
        dt : `float`
            The sampling period.
        amplitude : `numpy.ndarray`, shape=(L, J), optional
            The amplitude of the signal, defaults to numpy.ones((1, 1), dtype=float).
        t0 : `float`, optional
            The initial time, defaults to 0.0.
        size : `int`, optional
            The size of the signal, defaults to 1 << 20.
        offset : `numpy.ndarray`, shape=(L,), optional
            The offset value, defaults to numpy.zeros_like(amplitude, dtype=float).
        seed
            The seed for the random number generator.
        """
        if amplitude is None:
            amplitude = np.ones((1, 1), dtype=float)
        if amplitude.ndim == 1:
            amplitude = amplitude.reshape((-1, 1))
        if offset is None:
            offset = np.zeros_like(amplitude, dtype=float)
        if offset.ndim == 1:
            offset = offset.reshape((-1, 1))
        if amplitude.shape != offset.shape:
            raise ValueError("amplitude and offset must have the same shape")

        rng = np.random.default_rng(seed)
        L, J = amplitude.shape
        values = (rng.choice(3, size=(size, L, J), axis=0) - 1.0) * amplitude[
            np.newaxis, :, :
        ] + offset[np.newaxis, :, :]
        return ZeroOrderHold(dt, values, t0)

    @staticmethod
    def gaussian_reference_signal(
        dt: float,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        t0: float = 0.0,
        size: int = 1 << 16,
        seed: int = 98132712381291025,
    ) -> "ZeroOrderHold":
        """Create a Gaussian reference signal.

        Parameters
        ----------
        dt : `float`
            The sampling period.
        mean : `numpy.ndarray`, shape=(L, J), optional
            The mean of the signal, defaults to numpy.zeros(1, dtype=float).
        std : `numpy.ndarray`, shape=(L, J), optional
            The standard deviation of the signal, defaults to numpy.ones(1, dtype=float).
        t0 : `float`, optional
            The initial time, defaults to 0.0.
        size : `int`, optional
            The size of the signal, defaults to 1 << 20.
        seed
            The seed for the random number generator.

        Notes
        -----
        See :py:class:`numpy.random.Generator.normal` for more information.
        """
        if mean is None:
            mean = np.zeros((1, 1), dtype=float)
        if mean.ndim == 1:
            mean = mean.reshape((-1, 1))

        if std is None:
            std = np.ones((1, 1), dtype=float)
        if std.ndim == 1:
            std = std.reshape((-1, 1))
        if mean.shape != std.shape:
            raise ValueError("mean and std must have the same shape")

        L, J = mean.shape
        rng = np.random.default_rng(seed)
        values = rng.normal(mean, std, size=(size, L, J))

        return ZeroOrderHold(dt, values, t0)

    @staticmethod
    def uniform_reference_signal(
        dt: float,
        low: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
        t0: float = 0.0,
        size: int = 1 << 16,
        seed: int = 503147101294755601,
    ) -> "ZeroOrderHold":
        """Create a uniform reference signal.

        Parameters
        ----------
        dt : `float`
            The sampling period.
        low : `numpy.ndarray`, shape=(L, J), optional
            The lower bound of the signal, defaults to -numpy.ones(1, dtype=float).
        high : `numpy.ndarray`, shape=(L, J), optional
            The upper bound of the signal, defaults to numpy.ones(1, dtype=float).
        t0 : `float`, optional
            The initial time, defaults to 0.0.
        size : `int`, optional
            The size of the signal, defaults to 1 << 20.
        seed
            The seed for the random number generator.

        Notes
        -----
        See :py:class:`numpy.random.Generator.uniform` for more information.

        """
        if low is None:
            low = -np.ones((1, 1), dtype=float)
        if low.ndim == 1:
            low = low.reshape((-1, 1))
        if high is None:
            high = np.ones((1, 1), dtype=float)
        if high.ndim == 1:
            high = high.reshape((-1, 1))
        if low.shape != high.shape:
            raise ValueError("low and high must have the same shape")

        L, J = low.shape
        rng = np.random.default_rng(seed)
        values = rng.uniform(low, high, size=(size, L, J))

        return ZeroOrderHold(dt, values, t0)


class ModulatedSignal(AnalogSignal):
    """A Modulated signal.

    Parameters
    ----------
    signals : `list[AnalogSignal]`
        The signals to be modulated.

    Attributes
    ----------
    L : `int`
        The dimension of the input.
    J: `int`
        The number of inputs, used for parallel simulation.
    piecewise_constant : `bool`
        A flag indicating if the signal is piecewise constant.
    signals : `list[AnalogSignal]`
        The list of signals.


    """

    def __init__(self, *signals: AnalogSignal, dtype=float):
        self._dtype = dtype
        self.signals = signals

    @property
    def signals(self) -> tuple[AnalogSignal]:
        """The list of signals.

        Returns
        -------
        list[AnalogSignal]
            The list of signals.
        """
        return self._signals

    @signals.setter
    def signals(self, value: Optional[tuple[AnalogSignal]]):
        if not value:
            raise ValueError("At least one signal must be provided")
        if not all(isinstance(signal, AnalogSignal) for signal in value):
            raise TypeError("All elements must be instances of AnalogSignal")
        if any(signal.L != value[0].L for signal in value) or any(
            signal.J != value[0].J for signal in value
        ):
            raise ValueError("All signals must have the same dimension")
        if any(np.dtype(signal.dtype) != np.dtype(value[0].dtype) for signal in value):
            raise ValueError("All signals must have the same dtype")
        self._signals = value
        self.L = value[0].L
        self.J = value[0].J
        self._offset = np.zeros((self.L, self.J), dtype=self._dtype)

        self.piecewise_constant = all(signal.piecewise_constant for signal in value)

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the signal at time t.

        Parameters
        ----------
        t : `numpy.ndarray`, shape=(size,)
            the time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            The analog signal values
        """
        res = np.ones((t.size, self.L, self.J), dtype=self.dtype)
        for signal in self._signals:
            res *= signal.evaluate(t)
        return res

    def impulse_response(self, t: np.ndarray) -> np.ndarray:
        """Impulse response of the signal.

        Used for discretization and is not well defined for
        non LTI systems.

        Parameters
        ----------
        t : `numpy.ndarray`, shape=(size,)
            the time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L)
            The impulse response of the signal
        """
        res = np.ones((t.size, self.L, self.J), dtype=self.dtype)
        for signal in self._signals:
            res *= signal.impulse_response(t)
        return res

    def __str__(self):
        return "Modulation of of:\n" + "\n".join(
            [str(signal) for signal in self._signals]
        )


class SuperpositionSignal(ModulatedSignal):
    """Superposition signals.

    Parameters
    ----------
    signals : `list[AnalogSignal]`
        The signals to be superimposed.

    Attributes
    ----------
    L : `int`
        The dimension of the input.
    J: `int`
        The number of inputs, used for parallel simulation.
    piecewise_constant : `bool`
        A flag indicating if the signal is piecewise constant.
    signals : `list[AnalogSignal]`
        The list of signals.
    """

    def __init__(self, *signals: AnalogSignal, sub: list[bool] = None, dtype=float):
        self._dtype = dtype
        self.signals = signals
        if sub is None:
            sub = [False for _ in self.signals]
        if not all(isinstance(x, bool) for x in sub) or len(signals) != len(sub):
            raise ValueError(
                f"subtraction must be a list of booleans of same length as signals not signals: {signals}, sub: {sub}"
            )
        else:
            self.sub = sub

    @property
    def signals(self) -> tuple[AnalogSignal]:
        """The list of signals.

        Returns
        -------
        list[AnalogSignal]
            The list of signals.
        """
        return self._signals

    @signals.setter
    def signals(self, value: Optional[tuple[AnalogSignal]]):
        if not value:
            raise ValueError("At least one signal must be provided")
        if not all(isinstance(signal, AnalogSignal) for signal in value):
            raise TypeError("All elements must be instances of AnalogSignal")
        if any(signal.L != value[0].L for signal in value) or any(
            signal.J != value[0].J for signal in value
        ):
            raise ValueError("All signals must have the same dimension")
        if any(np.dtype(signal.dtype) != np.dtype(value[0].dtype) for signal in value):
            raise ValueError("All signals must have the same dtype")
        self._signals = value
        self.L = value[0].L
        self.J = value[0].J
        self._offset = np.zeros((self.L, self.J), dtype=self._dtype)
        self.piecewise_constant = all(signal.piecewise_constant for signal in value)

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the signal at time t.

        Parameters
        ----------
        t : `numpy.ndarray`, shape=(size,)
            the time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            The analog signal values
        """
        res = np.zeros((t.size, self.L, self.J), dtype=self.dtype)
        for i, signal in enumerate(self._signals):
            res += (-1 if self.sub[i] else 1) * signal.evaluate(t)
        return res

    def impulse_response(self, t: np.ndarray) -> np.ndarray:
        """Impulse response of the signal.

        Used for discretization and is not well defined for
        non LTI systems.

        Parameters
        ----------
        t : `numpy.ndarray`, shape=(size,)
            the time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            The impulse response of the signal
        """
        res = np.zeros((t.size, self.L, self.J), dtype=self.dtype)
        for signal in self._signals:
            res += signal.impulse_response(t)
        return res

    def __str__(self):
        return "Superposition of:\n" + "\n".join(
            [str(signal) for signal in self._signals]
        )


class ConcatenatedSignals(AnalogSignal):
    """A compound signal.

    Parameters
    ----------
    signals : `list[AnalogSignal]`
        The signals to be superimposed.

    Attributes
    ----------
    L : `int`
        The dimension of the input.
    J: `int`
        The number of inputs, used for parallel simulation.
    piecewise_constant : `bool`
        A flag indicating if the signal is piecewise constant.
    signals : `list[AnalogSignal]`
        The list of signals.
    """

    def __init__(self, *signals: AnalogSignal, dtype=float):
        self._dtype = dtype
        self.signals = signals

    @property
    def signals(self) -> tuple[AnalogSignal]:
        """The list of signals.

        Returns
        -------
        list[AnalogSignal]
            The list of signals.
        """
        return self._signals

    @signals.setter
    def signals(self, value: tuple[AnalogSignal]):
        if not value:
            raise ValueError("At least one signal must be provided")
        if not all(isinstance(signal, AnalogSignal) for signal in value):
            raise TypeError("All elements must be instances of AnalogSignal")
        if any(signal.J != value[0].J for signal in value):
            raise ValueError("All signals must have the same number of inputs")
        if any(np.dtype(signal.dtype) != np.dtype(value[0].dtype) for signal in value):
            raise ValueError("All signals must have the same dtype")
        self.J = value[0].J
        self.L = sum(signal.L for signal in value)
        self._signals = value
        self._offset = np.concatenate([signal.offset for signal in value], axis=0)
        self.piecewise_constant = all(signal.piecewise_constant for signal in value)
        i = 0
        self._l_slices = []
        for signal in value:
            self._l_slices.append(slice(i, i + signal.L))
            i += signal.L

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the signal at time t.

        Parameters
        ----------
        t : `numpy.ndarray`, shape=(size,)
            the time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            The analog signal values
        """
        res = np.zeros((t.size, self.L, self.J), dtype=self.dtype)
        for slc, signal in zip(self._l_slices, self._signals):
            res[:, slc, :] = signal.evaluate(t)
        return res

    def impulse_response(self, t: np.ndarray) -> np.ndarray:
        """Impulse response of the signal.

        Used for discretization and is not well defined for
        non LTI systems.

        Parameters
        ----------
        t : `numpy.ndarray`, shape=(size,)
            the time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            The impulse response of the signal
        """
        res = np.zeros((t.size, self.L, self.J), dtype=self.dtype)
        for slc, signal in zip(self._l_slices, self._signals):
            res[:, slc, :] = signal.impulse_response(t)
        return res

    def __str__(self):
        return "Compound signal of:\n" + "\n".join(
            [str(signal) for signal in self._signals]
        )


class PartitionedSignal(AnalogSignal):
    """A signal formed by stacking AnalogSignal instances along the J (parallel) axis.

    Parameters
    ----------
    *signals : AnalogSignal
        Two or more signals. All must share the same L.

    Attributes
    ----------
    L : int
        Signal dimension, shared across all partitions.
    J : int
        Total parallel signals; sum of all constituent J values.
    signals : tuple[AnalogSignal, ...]
        The constituent signals.
    piecewise_constant : bool
        True if all constituents are piecewise constant.
    """

    def __init__(self, *signals: AnalogSignal, dtype=float):
        if len(signals) < 2:
            raise ValueError("At least two signals must be provided")
        if not all(isinstance(s, AnalogSignal) for s in signals):
            raise TypeError("All elements must be instances of AnalogSignal")
        if any(s.L != signals[0].L for s in signals):
            raise ValueError("All signals must have the same L")
        if any(np.dtype(s.dtype) != np.dtype(signals[0].dtype) for s in signals):
            raise ValueError("All signals must have the same dtype")
        self._signals = signals
        super().__init__(
            offset=np.concatenate([s.offset for s in signals], axis=1),
            dtype=dtype,
        )
        self.piecewise_constant = all(s.piecewise_constant for s in signals)
        j = 0
        self._j_slices = []
        for s in self._signals:
            self._j_slices.append(slice(j, j + s.J))
            j += s.J

    @property
    def signals(self) -> tuple["AnalogSignal", ...]:
        """The constituent signals.

        Returns
        -------
        tuple[AnalogSignal, ...]
            The signals stacked along the J axis.
        """
        return self._signals

    def partition_indices(self) -> list[slice]:
        """Return slices along the J axis for each partition.

        Returns
        -------
        list[slice]
            One slice per constituent signal, in order.
            Use as ``result[:, :, slc]`` to recover the partition.
        """
        return self._j_slices

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate the signal at time t.

        Parameters
        ----------
        t : numpy.ndarray, shape=(size,)
            The time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            Constituent signals concatenated along the J axis.
        """
        return np.concatenate([s.evaluate(t) for s in self._signals], axis=2)

    def impulse_response(self, t: np.ndarray) -> np.ndarray:
        """Impulse response stacked along the J axis.

        Parameters
        ----------
        t : numpy.ndarray, shape=(size,)
            The time instances for evaluation.

        Returns
        -------
        numpy.ndarray, shape=(size, L, J)
            Constituent impulse responses concatenated along the J axis.
        """
        return np.concatenate([s.impulse_response(t) for s in self._signals], axis=2)

    def __str__(self):
        return "PartitionedSignal of:\n" + "\n".join([str(s) for s in self._signals])
