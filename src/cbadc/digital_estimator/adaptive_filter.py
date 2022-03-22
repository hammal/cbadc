"""an adaptive FIR filter.
"""
from typing import Tuple
from .fir_estimator import FIRFilter

import numpy as np
import logging

logger = logging.getLogger(__name__)


class AdaptiveFilter(FIRFilter):
    """An adaptive filter

    Parameters
    ----------
    initial_filter: :py:class:`cbadc.digital_estimator.FIRFilter`
        an initial filter to setup initial filter parameters and coefficients.
    reference_control_id: `int`
        an index indicating which control signal s_0, ..., s_M
        (and corresponding filter coefficients) are considered fixed references.
    size: `int`, `optional`
        a possible buffer size determining how many data points that should be
        used. Defaults to 0 which is interepreted as no buffered signals.

    """

    def __init__(
        self, initial_filter: FIRFilter, reference_control_id: int, size: int = 0
    ):
        self.K1 = initial_filter.K1
        self.K2 = initial_filter.K2
        self.K3 = initial_filter.K3
        self._filter_lag = self.K2 - 2
        self.analog_system = initial_filter.analog_system
        self.digital_control = initial_filter.digital_control
        self.eta2 = initial_filter.eta2
        self.control_signal = initial_filter.control_signal
        self.number_of_iterations = initial_filter.number_of_iterations
        self._iteration = 0
        self.Ts = initial_filter.Ts
        self.mid_point = initial_filter.mid_point
        self.downsample = int(initial_filter.downsample)
        self._temp_controls = np.zeros((self.downsample, self.analog_system.M))
        self.eta2Matrix = np.eye(self.analog_system.CT.shape[0]) * self.eta2
        self.h = initial_filter.h[:, :, :]
        self._control_signal_valued = np.zeros((self.K3, self.analog_system.M))
        self.offset = initial_filter.offset
        self.fixed_point = False
        self.ref_id = reference_control_id
        self.mask = np.ones(self.digital_control.M, dtype=bool)
        self.mask[self.ref_id] = False
        self._number_of_gradient_evaluations = 0
        self.gradient_steps = []
        self.step_size_template = np.ones_like(self.h)
        self._G = None
        self._Delta_Theta = None

        if size < 0:
            raise Exception("size must be nonnegative number.")
        if size > 0:
            self.size = size
            self._training_data_s = np.zeros(
                (
                    self.size,
                    initial_filter.K3,
                    initial_filter.analog_system.M,
                )
            )
            self._training_data_collected = False
            self._index = 0
            self.buffered_data = True
            self._gradient = self._gradient_buffered
        else:
            self.buffered_data = False
            self._gradient = self._gradient_unbuffered

    def _gradient_buffered(self, stochastic_delay=0) -> Tuple[np.ndarray, float, float]:
        if not self._training_data_collected:
            logger.info("Initializing training data")
            for index in range(self.size):
                self.__next__()
                self._training_data_s[index, :, :] = self._control_signal_valued[:, :]
                if stochastic_delay > 0:
                    for _ in range(np.random.randint(0, stochastic_delay)):
                        error_signal = self.__next__()
            self._training_data_collected = True

        error_signal = (
            np.tensordot(
                self.h, self._training_data_s[self._index, :, :], axes=((1, 2), (0, 1))
            )
            + self.offset
        )
        gradient = (
            2 * error_signal * self._training_data_s[self._index, :, :][:, self.mask]
        )
        offset_gradient = 2 * error_signal

        self._index = np.random.randint(0, self.size)
        # self._index = (self._index + 1) % self.size

        return (gradient, error_signal, offset_gradient)
        # self.h.shape -> (L, K3, M)
        # self._control_signal_valued.shape -> (K3, M)

    def _gradient_unbuffered(
        self, stochastic_delay=0
    ) -> Tuple[np.ndarray, float, float]:
        # self.h.shape -> (L, K3, M)
        # self._control_signal_valued.shape -> (K3, M)
        error_signal = self.__next__()
        if stochastic_delay > 0:
            for _ in range(np.random.randint(0, stochastic_delay)):
                error_signal = self.__next__()
        return (
            2 * error_signal * self._control_signal_valued[:, self.mask],
            error_signal,
            2 * error_signal,
        )

    def stochastic_gradient_decent(
        self,
        step_size: float,
        batch_size: int,
        stochastic_delay=0,
    ):
        """run a stochastic gradient decent batch

        Parameters
        ----------
        step_size: `float`
            a fixed step size for the given batch
        batch_size: `int`
            the number of gradient steps to be batched together.
        stochastic_delay: `int`, `optional`
            a number determining a uniformly random delay, in the
            interval [0, stochastic_delay), between data points used
            in the gradient descent. Defaults to 0.
        """

        step_sizes = self.step_size_template * step_size

        error_signals = np.zeros(batch_size)

        batch = np.zeros_like(self.h)
        offset_batch = 0.0
        for index in range(batch_size):
            gradient, error_signals[index], offset_graident = self._gradient(
                stochastic_delay
            )
            batch[:, :, self.mask] += gradient
            offset_batch += offset_graident
            self._number_of_gradient_evaluations += 1
        gradient_step = step_sizes * batch / batch_size
        offset_gradient_step = step_size * offset_batch / batch_size
        self.h = self.h - gradient_step
        self.offset = self.offset - offset_gradient_step
        return error_signals

    def compute_step_size_template(self, averaging_window_size=30):
        """set a step size profile dependent on the current
        filter weights.

        Parameters
        ----------
        averaging_window_size: `int`, `optional`
            the window size determinging how many neighbouring samples
            to average when computing the step size template, defaults
            to 30 samples.
        """
        filter_taps = np.ones(averaging_window_size)
        for m in range(self.h.shape[2]):
            # self.h.shape -> (L, K3, M)
            self.step_size_template[0, :, m] = np.convolve(
                np.abs(self.h[0, :, m]), filter_taps, mode="same"
            )

        self.step_size_template /= np.max(self.step_size_template)

    def stats(self) -> str:
        """return a string with the current stats of the average filter."""
        return f"""Gradient evaluations: {self._number_of_gradient_evaluations}"""

    def adadelta(
        self,
        epsilon: float = 1e-8,
        gamma: float = 0.9,
        batch_size: int = 1,
        stochastic_delay=0,
    ):
        """run an adadelta gradient update step

        Parameters
        ----------
        epsilon: `float`
            a lower bound on step size
        gamma: `float`
            related to momentum.
        batch_size: `int`
            the number of gradient steps to be batched together.
        stochastic_delay: `int`, `optional`
            a number determining a uniformly random delay, in the
            interval [0, stochastic_delay), between data points used
            in the gradient descent. Defaults to 0.
        """
        logger.warning("Adadelta does not yet estimate offsets.")
        error_signals = np.zeros(batch_size)
        if self._G is None:
            self._G = np.ones_like(self.h[:, :, self.mask]) * 1e-3

        if self._Delta_Theta is None:
            self._Delta_Theta_2 = np.zeros_like(self.h[:, :, self.mask])
            self._RMS_Delta_H = np.zeros_like(self.h[:, :, self.mask])

        if gamma > 1.0 or gamma < 0.0:
            raise Exception("gamma should be 0.0 < gamma < 1.0")
        gamma_minus = 1.0 - gamma

        for index in range(batch_size):
            gradient, error_signals[index], offset_error = self._gradient(
                stochastic_delay
            )
            self._number_of_gradient_evaluations += 1
            self._G = gamma * self._G + gamma_minus * (gradient**2)

            RMS_G = np.sqrt(self._G + epsilon)
            self._RMS_Delta_H = (
                -np.sqrt(self._Delta_Theta_2 + epsilon) / RMS_G * gradient
            )

            self.h[:, :, self.mask] = self.h[:, :, self.mask] + self._RMS_Delta_H

            self._Delta_Theta_2 = gamma * self._Delta_Theta_2 + gamma_minus * (
                self._RMS_Delta_H**2
            )
        return error_signals
