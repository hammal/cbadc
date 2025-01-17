"""The default digital control."""

from typing import Optional
import numpy as np

import logging as _logging

logger = _logging.getLogger(__name__)


class DigitalControl:
    """A digital control system.

    A digital control models the digital interactions in the control loop
    such as DAC waveform, timings, and quantization levels and offsets.

    Parameters
    ----------
    M : `int`
        number of controls.
    dt : `float`
        the time period between consecutive control updates.
    alpha : `numpy.ndarray`, optional
        the relative start time of the impulse response, defaults to
        np.zeros(M).
    beta : `numpy.ndarray`, optional
        the relative end time of the impulse response, defaults to
        np.ones(M).
    quantization_level : `numpy.ndarray`, optional
        the number of quantization levels, defaults to 2 * np.ones(M).
    quantization_gain : `numpy.ndarray`, optional
        the quantization gain, defaults to np.ones(M).
    dac_waveform : `str`, optional
        the DAC waveform, defaults to 'nrz'.
        - 'nrz': Non-return-to-zero
        - 'rz': Return-to-zero
        - 'ld': Linear decaying
        - 'qd': Quadratic decaying
        - 'scr': Step controlled response
        - 'cos': Cosine
        - 'ls': Linearly shaped
        - 'nls': Non-linearly shaped

    Attributes
    ----------
    M : `int`
        number of controls.
    dt : `float`
        the period between consecutive control updates.
    t0: `np.ndarray`, shape=(M, 1)
        the earliest start time of the impulse response.
    td: `np.ndarray`, shape=(M, 1)
        the delay time of the impulse response.
    tp: `np.ndarray`, shape=(M, 1)
        the time period of the impulse response.
    t1: `np.ndarray`, shape=(M, 1)
        the end time of the impulse response.
    tend: `np.ndarray`, shape=(M, 1)
        the latest end time of the impulse response.
    quantization_level: `np.ndarray`, shape=(M, 1)
        the number of quantization levels.
    quantization_gain: `np.ndarray`, shape=(M, 1)
        the quantization gain.
    dac_waveform: `str`
        the DAC waveform.

    """

    def __init__(
        self,
        M: int,
        dt: float,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        quantization_level: Optional[np.ndarray] = None,
        quantization_gain: Optional[np.ndarray] = None,
        dac_waveform: Optional[str] = None,
        **kwargs,
    ):
        if not isinstance(M, int) or M <= 0:
            raise ValueError("M must be a non negative integer.")
        self._M = M
        if not isinstance(dt, float) or dt <= 0:
            raise ValueError("Ts must be a positive float.")
        self._dt = dt

        if alpha is None:
            alpha = np.zeros(M)
        if beta is None:
            beta = np.ones(M)
        alpha = alpha.flatten()
        beta = beta.flatten()

        if (alpha < 0).any():
            raise ValueError("alphas must be a non negative float.")
        elif (alpha >= 1.0).any():
            raise ValueError("alphas must be less than Ts.")
        if (beta < 0).any():
            raise ValueError("betas must be a non negative float.")
        elif (alpha > beta).any():
            raise ValueError("alphas must be less than corresponding betas.")
        # elif (beta > 1.0).any():
        #     raise NotImplementedError("tp must be less than Ts.")
        self.alpha = alpha
        self.beta = beta
        self.t0 = np.zeros(M)
        self.tend = dt * np.ones(M)
        self.td = dt * alpha
        self.t1 = dt * beta
        self.tp = self.t1 - self.td
        # sanity check
        if (self.t0 + self.tend + self.td + self.t1 + self.tp).shape != (M,):
            raise ValueError("t0, tend, td, t1, and tp must be of shape (M,).")

        self.quantization_gain = quantization_gain
        self.quantization_level = quantization_level

        self._dac_waveform = None
        self._impulse_response = self.nonreturn2zero
        if dac_waveform is None:
            if (alpha == 0).all() and (beta == 1).all():
                self.dac_waveform = "nrz"
            else:
                self.dac_waveform = "rz"

    @property
    def M(self):
        """The number of controls."""
        return self._M

    @property
    def dt(self):
        """The time period between consecutive control updates."""
        return self._dt

    @dt.setter
    def dt(self, value: float):
        if not isinstance(value, float) or value <= 0:
            raise ValueError("Ts must be a positive float.")
        dt_old = self._dt
        self._dt = value
        self.t0 *= self._dt / dt_old
        self.tend *= self._dt / dt_old
        self.td *= self._dt / dt_old
        self.t1 *= self._dt / dt_old
        self.tp = self.t1 - self.td

    @property
    def quantization_level(self):
        """The number of quantization levels.

        Returns
        -------
        `np.ndarray`, shape=(M, 1)
            the number of quantization levels for each digital control.
        """
        return self._quantization_levels

    @quantization_level.setter
    def quantization_level(self, value: Optional[np.ndarray] = None):
        if value is None:
            value = 2 * np.ones(self.M, dtype=int)
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ValueError(
                f"Quantization levels must be a 1D array of length self.M, i.e., {self.M}."
            )
        if value.dtype != int:
            logger.warning("Quantization levels should be an integer array.")
            value = value.astype(int)
        if (value < 2).any():
            raise ValueError("Must be at least two quantization levels.")
        if value.size != self.M:
            raise ValueError("Quantization levels must be of length self.M.")
        self._quantization_levels = value.reshape((-1, 1))
        # shape(M, 1), [0, 1, 0, 1, 1, ...]
        self._mid_rise = np.array((self._quantization_levels % 2) == 0, dtype=float)
        # shape(M, 1), [1, 0, 1, 0, 0, ...]
        self._mid_thread = np.ones_like(self._mid_rise, dtype=float) - np.array(
            self._mid_rise, dtype=float
        )
        self._max = self._quantization_levels - 1
        self._min = -self._max

    @property
    def max(self):
        """The maximum output value."""
        return self._max

    @property
    def min(self):
        """The minimum output value."""
        return self._min

    @property
    def quantization_gain(self):
        """The quantization gain.

        Returns
        -------
        `np.ndarray`, shape=(M, 1)
            the quantization gain for each digital control.
        """
        return self._quantization_gain

    @quantization_gain.setter
    def quantization_gain(self, value: Optional[np.ndarray] = None):
        if value is None:
            value = np.ones(self.M, dtype=float)
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ValueError(
                f"Quantization gain must be a 1D array of length self.M, i.e., {self.M}."
            )
        if value.size != self.M:
            raise ValueError("Quantization gain must be of length self.M.")
        self._quantization_gain = value.reshape((-1, 1))
        self._pre_gain = 0.5 * self._quantization_gain

    @property
    def dac_waveform(self):
        """The DAC waveform.

        The DAC waveform can be one of the following:
        - 'nrz': Non-return-to-zero
        - 'rz': Return-to-zero
        - 'ld': Linear decaying
        - 'qd': Quadratic decaying
        - 'scr': Step controlled response
        - 'cos': Cosine
        - 'ls': Linearly shaped
        - 'nls': Non-linearly shaped

        """
        return self._dac_waveform

    @dac_waveform.setter
    def dac_waveform(self, value: str, **kwargs):
        if value == "nrz":
            self._impulse_response = self.nonreturn2zero
        elif value == "rz":
            self._impulse_response = self.return2zero
        elif value == "ld":
            self._impulse_response = self._lineardecaying
        elif value == "qd":
            self._impulse_response = self._quadraticdecaying
        elif value == "scr":
            self._scr_tau = kwargs.get("tau_DAC", 1.0)
            self._impulse_response = self._scr
        elif value == "cos":
            self._impulse_response = self._cos
        elif value == "ls":
            self._impulse_response = self._ls
        elif value == "nls":
            self._impulse_response = self._nls
        else:
            raise ValueError(
                "DAC waveform must be one of 'nrz', 'rz', 'ld', 'qd', 'scr', 'cos', 'ls', or 'nls'."
            )
        self._dac_waveform = value

    def nonreturn2zero(self, t: np.ndarray) -> np.ndarray:
        """Nonreturn-to-zero DAC impulse response

        as per Eq. (3.1) in

        M. Ortmanns and F. Gerfers, Continuous-time sigma-delta A/D conversion:
        fundamentals, performance limits, and robust implementations. in Springer
        series in advanced microelectronics, no. 21. Berlin; New York: Springer,
        2006, doi: 10.1007/3-540-28473-7.

        Parameters
        ----------
        t : `np.ndarray`
            time instances for evaluation.

        Returns
        -------
        `np.ndarray`, shape=(M, t.size)
            the impulse response evaluated at time t.
        """
        # Make sure broadcast correctly
        # note that tmp is (1, t.size) shaped and
        # self.t0 and self.tend are (M, 1) shaped.
        tmp = t.reshape((1, -1))
        return np.array((tmp >= self.t0) & (tmp < self.tend), dtype=float)

    def return2zero(self, t: np.ndarray) -> np.ndarray:
        """Return-to-zero DAC impulse response

        as per Eq. (3.2) in

        M. Ortmanns and F. Gerfers, Continuous-time sigma-delta A/D conversion:
        fundamentals, performance limits, and robust implementations. in Springer
        series in advanced microelectronics, no. 21. Berlin; New York: Springer,
        2006, doi: 10.1007/3-540-28473-7.

        Parameters
        ----------
        t : `np.ndarray`
            time instances for evaluation.

        Returns
        -------
        `np.ndarray`, shape=(M, t.size)
            the impulse response evaluated at time t.
        """
        tmp = t.reshape((1, -1))
        return np.array((tmp >= self.td) & (tmp < self.t1), dtype=float)

    def _lineardecaying(self, t: np.ndarray) -> np.ndarray:
        tmp = t.reshape((1, -1))
        return (1.0 - (tmp - self.td) / self.tp) * self.return2zero(tmp)

    def _quadraticdecaying(self, t: np.ndarray) -> np.ndarray:
        tmp = t.reshape((1, -1))
        return (1.0 - (tmp - self.td) / self.tp) ** 2 * self.return2zero(tmp)

    def _scr(self, t: np.ndarray) -> np.ndarray:
        tmp = t.reshape((1, -1))
        return self.return2zero(tmp) * np.exp(-(tmp - self.td) / self._scr_tau)

    def _cos(self, t: np.ndarray) -> np.ndarray:
        tmp = t.reshape((1, -1))
        return (1.0 - (tmp - self.td) / self.tp) ** 2 * (
            np.array(tmp >= 0.0 and tmp < self.t1, dtype=float)
        )

    def _ls(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Not implemented yet.")

    def _nls(self, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Not implemented yet.")

    def quantize(self, value: np.ndarray) -> np.ndarray:
        """Quantize the input value.

        Parameters
        ----------
        value : `np.ndarray`, shape=(M,...)
            the input value to be quantized.

        Returns
        -------
        `np.ndarray`, shape=(M,...)
            the quantized value.
        """
        if value.shape[0] != self.M:
            raise ValueError("The input value must have M rows.")
        broadcasting_shape = np.ones(value.ndim, dtype=int)
        broadcasting_shape[0] = self.M

        return np.clip(
            2.0
            * np.floor(
                self._pre_gain.reshape(broadcasting_shape) * value
                + self._mid_thread.reshape(broadcasting_shape)
            )
            + self._mid_rise.reshape(broadcasting_shape),
            self._min,
            self._max,
        )

    def impulse_response(self, t_delay: np.ndarray) -> np.ndarray:
        """The DAC waveform at time t.

        Parameters
        ----------
        t_delay : `numpy.ndarray`, shape=(size,)
            the times at which the DAC waveform is evaluated.

        Returns
        -------
        `numpy.ndarray`, shape=(M, size)
            the DAC waveform at time t_delay.
        """
        if not isinstance(t_delay, np.ndarray):
            raise ValueError("t_delay must be a numpy array.")
        return self._impulse_response(np.array([t_delay]))

    def evaluate(self, t_delay: float, value: np.ndarray) -> np.ndarray:
        """Evaluate the control contribution at time t.

        Parameters
        ----------
        t_delay : `float`
            time at which the digital control is evaluated.
        value : `array_like`, shape=(M, number_of_delays, ...)
            the quantizer inputs evaluated at time t_delay, t_delay-Ts,
            t_delay-2Ts, ..., t_delay-N * Ts.

        Returns
        -------
        `np.ndarray`, shape=(M, ...)
            the control contribution at time t_delay.
        """

        if not isinstance(t_delay, float) or t_delay < 0:
            raise ValueError("t_delay must be a non negative float.")

        # shape(M, delays)
        time_shifts = t_delay + self.dt * np.arange(value.shape[1])
        # shape(M, delays)
        dac_waveform = self._impulse_response(time_shifts)
        broadcast_shape = np.ones(value.ndim, dtype=int)
        broadcast_shape[:2] = value.shape[:2]
        # shape(M, delays, ...)
        quantized = self.quantize(value)
        return np.sum(quantized * dac_waveform.reshape(broadcast_shape), axis=1)

    def __call__(self, t: float, value: np.ndarray) -> np.ndarray:
        return self.evaluate(t, value)

    def delay_steps(self) -> np.ndarray:
        """The number of delay steps.

        Compute the number of delay steps, N, required to evaluate
        the full DAC waveform.

        Returns
        -------
        `int`
            the number of delay steps.
        """
        return np.ceil(np.maximum(self.tend, self.t1) / self.dt).astype(int) - 1
