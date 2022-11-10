"""The digital FIR estimator"""
from dataclasses import dataclass
from typing import Iterator, Union, List
import cbadc
import logging
import os
import numpy as np
from .batch_estimator import BatchEstimator
from ._filter_coefficients import FilterComputationBackend

logger = logging.getLogger(__name__)


class FIRFilter(BatchEstimator):
    """FIR filter implementation of the digital estimator.

    Specifically, the FIR filter estimator estimates a filtered version :math:`\hat{\mathbf{u}}(t)` (shaped by
    :py:func:`signal_transfer_function`) of the
    input signal :math:`\mathbf{u}(t)` from a sequence of control signals :math:`\mathbf{s}[k]`.

    Specifically, the estimate is of the form

    :math:`\hat{\mathbf{u}}(k T) = \hat{\mathbf{u}}_0 + \sum_{\ell=-K_1}^{K_2} \mathbf{h}[\ell] \mathbf{s}[k + \ell]`

    where

    :math:`\mathbf{h}[\ell]=\\begin{cases}\mathbf{W}^{\mathsf{T}} \mathbf{A}_b^\ell \mathbf{B}_b & \mathrm{if} \, \ell \geq 0 \\\  -\mathbf{W}^{\mathsf{T}} \mathbf{A}_f^{-\ell + 1} \mathbf{B}_f & \mathrm{else} \\end{cases}`

    and :math:`\mathbf{W}^{\mathsf{T}}`, :math:`\mathbf{A}_b`,
    :math:`\mathbf{B}_b`, :math:`\mathbf{A}_f`, and :math:`\mathbf{B}_f`
    are computed based on the analog system, the sample period :math:`T_s`, and the
    digital control's DAC waveform as described in
    `control-bounded converters <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf?sequence=1&isAllowed#=ypage=67/>`_.

    Parameters
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system (necessary to compute the estimators filter coefficients).
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        a digital control (necessary to determine the corresponding DAC waveform).
    eta2: `float`
        the :math:`\eta^2` parameter determines the bandwidth of the estimator.
    K1: `int`
        The lookback size
    K2: `int`, `optional`
        lookahead size, defaults to 0.
    stop_after_number_of_iterations: `int`
        determine a max number of iterations by the iterator, defaults to  :math:`2^{63}`.
    Ts: `float`, `optional`
        the sampling time, defaults to the time period of the digital control.
    mid_point: `bool`, `optional`
        set samples in between control updates, i.e., :math:`\hat{u}(kT + T/2)`, defaults to False.
    downsample: `int`, `optional`
        specify down sampling rate in relation to the control period :math:`T`, defaults to 1, i.e.,
        no down sampling.
    offset: `array_like`, shape=(L), `optional`
        the estimate offset :math:`\hat{\mathbf{u}}_0`, defaults to a zero vector.
    fixed_point: :py:class:`cbadc.utilities.FixedPoint`, `optional`
        fixed point arithmetic configuration, defaults to None.
    solver_type: :py:class:`cbadc.digital_estimator._filter_coefficients.FilterComputationBackend`
        determine which solver type to use when computing filter coefficients.

    Attributes
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        analog system as in :py:class:`cbadc.analog_system.AnalogSystem` or from
        derived class.
    eta2: float
        eta2, or equivalently :math:`\eta^2`, sets the bandwidth of the estimator.
    control_signal: :py:class:`cbadc.digital_control.DigitalControl`
        a iterator suppling control signals as :py:class:`cbadc.digital_control.DigitalControl`.
    number_of_iterations: `int`
        number of iterations until iterator raises :py:class:`StopIteration`.
    K1: `int`
        number of samples, prior to estimate, used in estimate
    K2: `int`
        number of lookahead samples per computed batch.
    Ts: `float`
        spacing between samples in seconds.
    mid_point: `bool`
        estimated samples shifted in between control updates, i.e., :math:`\hat{u}(kT + T/2)`.
    downsample: `int`, `optional`
        down sampling rate in relation to the control period :math:`T`.
    Af: `array_like`, shape=(N, N)
        The Af matrix
    Ab: `array_like`, shape=(N, N)
        The Ab matrix
    Bf: `array_like`, shape=(N, M)
        The Bf matrix
    Bb: `array_like`, shape=(N, M)
        The Bb matrix
    WT: `array_like`, shape=(L, N)
        The W matrix transposed
    h: `array_like`, shape=(L, K1 + K2, M)
        filter impulse response
    offset: `array_like`, shape=(L)
        the estimate offset :math:`\hat{\mathbf{u}}_0`.
    fixed_point: `bool`
        using fixed point?
    solver_type: :py:class:`cbadc.digital_estimator._filter_coefficients.FilterComputationBackend`
        The solver used for computing the filter coefficients.


    Yields
    ------
    `array_like`, shape=(L,)
        an input estimate sample :math:`\hat{\mathbf{u}}(t)`
    """

    def __init__(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
        eta2: float,
        K1: int,
        K2: int,
        stop_after_number_of_iterations: int = (1 << 63),
        Ts: float = None,
        mid_point: bool = False,
        downsample: int = 1,
        offset: np.ndarray = None,
        fixed_point: cbadc.utilities.FixedPoint = None,
        solver_type: FilterComputationBackend = FilterComputationBackend.mpmath,
        modulation_frequency: float = None,
    ):
        """Initializes filter coefficients"""
        if K1 < 0:
            raise Exception("K1 must be non negative integer.")
        self.K1 = K1
        if K2 < 1:
            raise Exception("K2 must be a positive integer.")
        self.K2 = K2
        self.K3 = K1 + K2
        self._filter_lag = self.K2 - 2
        self.analog_system = analog_system
        self.digital_control = digital_control
        if eta2 < 0.0:
            raise Exception("eta2 must be non negative.")
        self.eta2 = eta2
        self.control_signal = None
        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        if Ts:
            self.Ts = Ts
        else:
            self.Ts = digital_control.clock.T
        if mid_point:
            raise NotImplementedError("Planned for v.0.1.0")
        self.mid_point = mid_point
        self.downsample = int(downsample)
        self._temp_controls = np.zeros((self.downsample, self.analog_system.M))

        if offset is not None:
            self.offset = np.array(offset, dtype=np.float64)
            if self.offset.size != self.analog_system.L:
                raise Exception("offset is not of size L")
        else:
            self.offset = np.zeros(self.analog_system.L, dtype=np.float64)

        if fixed_point is not None:
            self.fixed_point = True
            self.__fixed_point = fixed_point
            self.__fixed_to_float = np.vectorize(self.__fixed_point.fixed_to_float)
            self.__float_to_fixed = np.vectorize(self.__fixed_point.float_to_fixed)
        else:
            self.fixed_point = False

        # For transfer functions
        self.eta2Matrix = np.eye(self.analog_system.CT.shape[0]) * self.eta2
        # Compute filter coefficients
        self.solver_type = solver_type
        self._compute_filter_coefficients(analog_system, digital_control, eta2)

        # Initialize filter.
        if self.fixed_point:
            self.h = np.zeros(
                (self.analog_system.L, self.K3, self.analog_system.M), dtype=np.int64
            )
        else:
            self.h = np.zeros(
                (self.analog_system.L, self.K3, self.analog_system.M), dtype=np.double
            )
        # Compute lookback.
        temp1 = np.copy(self.Bf)
        for k1 in range(self.K1 - 1, -1, -1):
            if self.fixed_point:
                self.h[:, k1, :] = self.__float_to_fixed(-np.dot(self.WT, temp1))
            else:
                self.h[:, k1, :] = -np.dot(self.WT, temp1)
            temp1 = np.dot(self.Af, temp1)

        # Compute lookahead.
        temp2 = np.copy(self.Bb)
        for k2 in range(self.K1, self.K3):
            if self.fixed_point:
                self.h[:, k2, :] = self.__float_to_fixed(np.dot(self.WT, temp2))
            else:
                self.h[:, k2, :] = np.dot(self.WT, temp2)
            temp2 = np.dot(self.Ab, temp2)
        self._control_signal_valued = np.zeros((self.K3, self.analog_system.M))

        # For modulation
        self._time_index = 0
        if modulation_frequency is not None:
            self._modulation_frequency = modulation_frequency
        else:
            self._modulation_frequency = 0

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        # Check if control signal iterator is set.
        if self.control_signal is None:
            raise Exception("No iterator set.")

        # Check if the end of prespecified size
        self._iteration += self.downsample
        if self.number_of_iterations and self.number_of_iterations < self._iteration:
            raise StopIteration

        # Rotate control_signal vector
        self._control_signal_valued = np.roll(
            self._control_signal_valued, -self.downsample, axis=0
        )

        # insert new control signal
        try:
            for index in range(self.downsample):
                self._temp_controls[index, :] = 2 * self.control_signal.__next__() - 1
        except RuntimeError:
            logger.warning("Estimator received Stop Iteration")
            raise StopIteration

        self._control_signal_valued[
            self.K3 - self.downsample :, :
        ] = self._temp_controls

        # self._control_signal_valued.shape -> (K3, M)
        # self.h.shape -> (L, K3, M)
        res = (
            np.tensordot(self.h, self._control_signal_valued, axes=((1, 2), (0, 1)))
            + self.offset
        )
        self._time_index += 1
        if self.fixed_point:
            return self.__fixed_to_float(res)
        else:
            return res
        # return np.einsum('lkm,km', self.h, self._control_signal_valued)
        # the Einstein summation results in:
        # result = np.zeros(self._L)
        # for l in range(self._L):
        #    for k in range(self.K1 + self.K2):
        #        for m in range(self._M):
        #            result[l] += self.h[l, k, m] * self._control_signal_valued[k, m]
        # return result

    def lookback(self):
        """Return lookback size :math:`K1`.

        Returns
        -------
        int
            lookback size.
        """
        return self.K1

    def __str__(self):
        return f"FIR estimator is parameterized as \neta2 = {self.eta2:.2f}, {10 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},\nand\nnumber_of_iterations = {self.number_of_iterations}.\nResulting in the filter coefficients\nh = \n{self.h}."

    def filter_lag(self):
        """Return the lag of the filter.

        Returns
        -------
        `int`
            The filter lag.

        """
        return self._filter_lag

    def fir_filter_transfer_function(self, Ts: float = 1.0):
        """Compute the FFT of the system impulse response (FIR filter coefficients).

        Parameters
        ----------
        Ts: `float`
            the sample period of the corresponding impulse response.

        Returns
        -------
        `array_like`, shape=(L, K3 // 2, M)
            the FFT of the corresponding impulse responses.
        """
        frequency_response = np.fft.rfft(self.h, axis=1)
        frequencies = np.fft.rfftfreq(self.K3, d=Ts)
        return (frequencies, frequency_response)

    def convolve(self, filter: np.ndarray):
        """Shape :math:`\mathbf{h}` filter by convolving with filter

        Parameters
        ----------
        filter: `array_like`, shape=(K)
            filter to be applied for each digital control filter
            equivalently.
        """
        for l in range(self.analog_system.L):
            for m in range(self.analog_system.M):
                # self.h.shape -> (L, K3, M)
                temp = np.convolve(self.h[l, :, m], filter, mode="full")
                if temp.size == self.K3:
                    self.h[l, :, m] = temp
                else:
                    mid_point = temp.size // 2
                    half_length = self.K3 // 2
                    self.h[l, :, m] = temp[
                        (mid_point - half_length) : (mid_point + half_length)
                    ]

    def write_C_header(self, filename: str):
        """Write the FIR filter coefficients h into
        a C header file.

        Parameters
        ----------
        filename: `str`
            filename of header file.
        """
        if not filename.endswith(".h"):
            filename += ".h"

        with open(filename, "w") as f:
            f.write(
                "// This file was autogenerated using the src/cbadc/digital_estimator.py FIRFilter class's write_C_header function."
                + os.linesep
            )
            f.write(f"#define L {self.analog_system.L}" + os.linesep)
            f.write(f"#define M {self.analog_system.M}" + os.linesep)
            f.write(f"#define K1 {self.K1}" + os.linesep)
            f.write(f"#define K2 {self.K2}" + os.linesep)
            f.write(f"#define K3 {self.K3}" + os.linesep)
            # self.h.shape -> (L, K3, M)
            if self.fixed_point:
                f.write("int h[L][M][K3] = ")
            else:
                f.write("double h[L][M][K3] = ")
            f.write("{")
            for l in range(self.analog_system.L):
                f.write("{")
                for m in range(self.analog_system.M):
                    f.write("{")
                    if self.fixed_point:
                        for k in range(self.K3 - 1):
                            f.write(f"{self.h[l,k,m]},")
                        f.write(f"{self.h[l,-1,m]}" + "}")
                    else:
                        for k in range(self.K3 - 1):
                            f.write(f"{self.h[l,k,m]:.17E},")
                        f.write(f"{self.h[l,-1,m]:.17E}" + "}")
                    if m < (self.analog_system.M - 1):
                        f.write(",")
                f.write("}")
                if l < (self.analog_system.L - 1):
                    f.write(",")
            f.write("};" + os.linesep)

    def write_rust_module(self, filename: str):
        """Write the FIR filter coefficients into a rust module

        Parameters
        ----------
        filename: `str`
            filename of header file.
        """
        if not filename.endswith(".rs"):
            filename += ".rs"

        with open(filename, "w") as f:
            f.write(
                "// This file was autogenerated using the src/cbadc/digital_estimator.py FIRFilter class's write_rust_module function."
                + os.linesep
            )
            f.write(f"let L: usize = {self.analog_system.L};" + os.linesep)
            f.write(f"let M: usize = {self.analog_system.M};" + os.linesep)
            f.write(f"let K1: usize = {self.K1};" + os.linesep)
            f.write(f"let K2: usize = {self.K2};" + os.linesep)
            f.write(f"let K3: usize = {self.K3};" + os.linesep)
            # self.h.shape -> (L, K3, M)
            if self.fixed_point:
                f.write("let mut h = ")
            else:
                f.write("let mut h = ")
            f.write("[")
            for l in range(self.analog_system.L):
                f.write("[")
                for m in range(self.analog_system.M):
                    f.write("[")
                    if self.fixed_point:
                        for k in range(self.K3 - 1):
                            f.write(f"{self.h[l,k,m]},")
                        f.write(f"{self.h[l,-1,m]}" + "]")
                    else:
                        for k in range(self.K3 - 1):
                            f.write(f"{self.h[l,k,m]:.17e},")
                        f.write(f"{self.h[l,-1,m]:.17e}" + "]")
                    if m < (self.analog_system.M - 1):
                        f.write(",")
                f.write("]")
                if l < (self.analog_system.L - 1):
                    f.write(",")
            f.write("];" + os.linesep)

    def number_of_filter_coefficients(self) -> int:
        """Number of non-zero filter coefficients

        Returns
        -------
        `int`
            total number of non-zero filter coefficients
        """
        return np.sum(self.h > 0)

    def impulse_response(self) -> np.ndarray:
        """Return the filter's impulse response

        Returns
        -------
        `array_like`, shape=(L, K3, M)
            the impulse response
        """
        # (self.analog_system.L, self.K3, self.analog_system.M)
        return self.h[:, ::-1, :]


@dataclass
class FixedStepSize:
    """Step size class

    Parameters
    ----------
    step_size: `float`
        step size
    """

    step_size: float = 1e-3

    def __init__(self, step_size: float):
        self.step_size: float = float(step_size)

    def __call__(self, iteration: int) -> float:
        """Return the step size

        Parameters
        ----------
        iteration: `int`
            iteration number

        Returns
        -------
        `float`
            step size
        """
        return self.step_size


@dataclass
class ExponentialStepSize(FixedStepSize):
    """Exponential decay step size class


    Parameters
    ----------
    step_size: `float`
        step size
    decay_rate: `float`
        decay rate
    decay_steps: `int`
        decay steps
    """

    decay_rate: float = 1.0 - 1e-5
    decay_steps: int = 1000

    def __init__(self, step_size: float, decay_rate: float, decay_steps: int):
        super().__init__(step_size)
        self.decay_rate = float(decay_rate)
        self.decay_steps = int(decay_steps)

    def __call__(self, iteration: int) -> float:
        """Return the step size

        returns step_size * decay_rate ** (iteration / decay_steps)

        Parameters
        ----------
        iteration: `int`
            iteration number

        Returns
        -------
        `float`
            step size
        """
        return self.step_size * (self.decay_rate ** (iteration // self.decay_steps))


# @dataclass
# class PolynomialStepSize:
#     """Polynomial decay step size class

#     Parameters
#     ----------
#     step_size: `float`
#         step size
#     decay: `float`
#         decay rate
#     """

#     initial_step_size: float = 1.0 - 1e-5
#     final_step_size: float = 1.0 - 1e-7
#     power: int = 2
#     decay_steps: int = 1000

#     def __init__(
#         self,
#         initial_step_size: float,
#         final_step_size: float,
#         power: int,
#         decay_steps: int,
#     ):
#         self.initial_step_size = float(initial_step_size)
#         self.final_step_size = float(final_step_size)
#         self.power = int(power)
#         self.decay_steps = int(decay_steps)

#     def __call__(self, iteration: int) -> float:
#         """Return the step size

#         Parameters
#         ----------
#         iteration: `int`
#             iteration number

#         Returns
#         -------
#         `float`
#             step size
#         """
#         return (self.initial_step_size - self.final_step_size) * (
#             1 - iteration // self.decay_steps
#         ) ** self.power + self.final_step_size


StepSize = Union[FixedStepSize, ExponentialStepSize]


class Training:
    """
    A utility class for training parameters of a digital estimator

    Attributes
    ----------
    training_iterations: `int`
        number of training iterations
    V: `array_like`, shape=(K_sum, K_sum)
        covariance matrix for RLS training
    k_sum: `int`
        total number of filter coefficients
    method: `str`
        training method, possible choices are: 'rls', 'lms', and 'adam'
    rls_lambda: :py:class:`cbadc.digital_estimator.fir_estimator.StepSize`
        step size for RLS training
    rls_delta: `float`
        regularization parameter for RLS training
    lms_learning_rate: :py:class:`cbadc.digital_estimator.fir_estimator.StepSize`
        step size for LMS training
    lms_momentum: `float`
        momentum parameter for LMS training
    adam_learning_rate: :py:class:`cbadc.digital_estimator.fir_estimator.StepSize`
        step size for Adam training
    adam_beta1: `float`
        beta1 parameter for Adam training
    adam_beta2: `float`
        beta2 parameter for Adam training
    adam_epsilon: `float`
        epsilon parameter for Adam training
    adam_t: `int`
        iteration number for Adam training
    """

    training_iterations: int = 0
    V: np.ndarray = None
    k_sum: int = 0
    # methods for training
    method: str = "rls"
    # RLS parameters
    rls_lambda: StepSize = FixedStepSize(0.99999)
    rls_delta: float = 1e4

    # stochastic gradient descent parameters
    lms_learning_rate: StepSize = FixedStepSize(1e-3)
    lms_momentum: float = 0.999

    # ADAM parameters
    adam_learning_rate: StepSize = FixedStepSize(1e-3)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_t: int = 1


class ControlSignalBuffer:
    """Buffer for storing control signals when training and testing calibration of a digital estimator

    Parameters
    ----------
    M: `int`
        number of control signals
    K: [int]
        number of filter coefficients for each control signal
    buffer_size: `int`
        size of the buffer
    downsample: `int`
        downsample factor. Used when filling up buffer
    dtype: `type`
        data type of the buffer
    randomized_order: `bool`
        if True, the order of the control signals is randomized when filling up the buffer, defaults to False
    stochastic_delay: `int`
        use a stochastic delay when filling up the buffer, defaults to 1, i.e., no stochastic delay

    """

    M: int
    K: List[int]
    buffer_size: int
    downsample: int
    randomized_order: bool
    stochastic_delay: int

    def __init__(
        self,
        M: int,
        K: List[int],
        buffer_size: int = 0,
        downsample: int = 1,
        dtype: type = np.float64,
        randomized_order: bool = False,
        stochastic_delay: int = 1,
    ):
        self.M = M
        self.K = [k for k in K]
        self._K_max = max(self.K)
        self.K_start_end = [
            ((self._K_max - self.K[m]) // 2, (self._K_max + self.K[m]) // 2)
            for m in range(self.M)
        ]
        self._index = 0
        if buffer_size > 0:
            self._full_buffer = False
            self.buffer_size = buffer_size
            self.buffer = True
        else:
            self.buffer_size = 1
            self.buffer = False
        self._buffer = [
            np.zeros((self.buffer_size, self._K_max), dtype=dtype) for m in range(M)
        ]
        self.downsample = downsample
        if randomized_order and stochastic_delay != 1:
            raise ValueError(
                "Randomized order and stochastic delay are mutually exclusive."
            )
        self.randomized_order = randomized_order
        if stochastic_delay < 1:
            raise ValueError("Stochastic delay must be >= 1.")
        self.stochastic_delay = stochastic_delay

    def __next__(self) -> List[np.ndarray]:
        if self.buffer:
            # with buffer
            if self._full_buffer:
                # Randomized order
                if self.randomized_order:
                    return [
                        np.copy(
                            self._buffer[m][
                                np.random.randint(0, self.buffer_size),
                                self.K_start_end[m][0] : self.K_start_end[m][1],
                            ]
                        )
                        for m in range(self.M)
                    ]
                # Stochastic delay
                else:
                    index = (
                        self._index + np.random.randint(0, self.stochastic_delay)
                    ) % self.buffer_size
                    self._index += 1
                    return [
                        np.copy(
                            self._buffer[m][
                                index, self.K_start_end[m][0] : self.K_start_end[m][1]
                            ]
                        )
                        for m in range(self.M)
                    ]
            else:
                self.fill_buffer()
                return self.__next__()
        else:
            # without buffer
            return self._receive_data()

    def fill_buffer(self):
        # logger.info("Filling up control-signal buffer...")
        self._full_buffer = False
        while not self._full_buffer:
            _ = self._receive_data()
            if self._index == self.buffer_size - 1:
                self._full_buffer = True
                self._index = 0

    def set_iterator(self, iterator: Iterator[np.ndarray]):
        """Set iterator for buffer

        Parameters
        ----------
        iterator: `Iterator`
            iterator for buffer
        """
        self._iterator = iterator

    def __call__(self, iterator: Iterator[np.ndarray]):
        self.set_iterator(iterator)

    def warm_up(self):
        """Warm up buffer"""
        for _ in range(self._K_max):
            self._receive_data()
            self.reset()

    def _receive_data(self) -> List[np.ndarray]:

        temp = np.zeros((self.M, self.downsample))

        # Retrieve data from iterator
        try:
            for index in range(self.downsample):
                temp[:, index] = 2 * next(self._iterator) - 1
        except StopIteration:
            logger.warning("Estimator received Stop Iteration")
            raise StopIteration

        # Update buffer
        for m in range(self.M):
            # Rotate control_signal vector
            self._buffer[m][
                (self._index + 1) % self.buffer_size, : -self.downsample
            ] = self._buffer[m][self._index, self.downsample :]

            # Add new control_signal
            self._buffer[m][
                (self._index + 1) % self.buffer_size, -self.downsample :
            ] = temp[m, :]

        self._index = (self._index + 1) % self.buffer_size

        return [
            self._buffer[m][
                self._index, self.K_start_end[m][0] : self.K_start_end[m][1]
            ]
            for m in range(self.M)
        ]

    def reset(self):
        self._index = 0
        self._full_buffer = False


def initial_wiener_filter(h_wiener: np.ndarray):
    """Reformat an initial Wiener filter into an initial reference filter
    used for calibration of a digital estimator

    Parameters
    ----------
    h_wiener: `np.ndarray`
        initial Wiener filter as FIR filter

    Returns
    -------
    h: `List[List[np.ndarray]]`
        an initialized reference filter for calibration
    """
    L = h_wiener.shape[0]
    K = h_wiener.shape[1]
    M = h_wiener.shape[2]
    h = [[np.zeros(K, dtype=np.float64) for m in range(M)] for _ in range(L)]
    for l in range(L):
        for m in range(M):
            h[l][m] = h_wiener[l, :, m]
    return h


def initial_filter(
    h0: List[np.ndarray], K: List[int], reference_index: List[int]
) -> List[List[np.ndarray]]:
    """
    Factory function for initial reference filters

    Parameters
    ----------
    h0: `List[np.ndarray]`
        Initial reference filters
    K: `List[int]`
        number of filter coefficients per control signal
    reference_index: `List[int]`
        index of reference signal(s)

    Returns
    -------
    h: `List[List[np.ndarray]]`
        an initialized reference filter for calibration
    """
    h = [
        [np.zeros(K[m], dtype=np.float64) for m in range(len(K))]
        for _ in range(len(h0))
    ]
    # insert reference filter coefficients
    for l_index, ref_index in enumerate(reference_index):
        if h0[l_index].size != K[ref_index]:
            raise ValueError(f"h0[{l_index}] must have length K[{ref_index}]")
        h[l_index][ref_index] = h0[l_index][:]
    return h


class AdaptiveFIRFilter:
    """Adaptive FIR filter which can be used for both calibration and estimation

    Parameters
    ----------
    h: `List[List[np.ndarray]]`
        initial reference filter
    K: `List[int]`
        number of filter coefficients per control signal
    reference_index: `List[int]`
        index of reference signal(s)
    downsample: `int`
        downsampling factor, default: 1
    method: `str`
        method for calibration, default: 'rls'
    projection: `bool`
        use projection when calibrating using gradient approaches, default: False
    randomized_order: `bool`
        use randomized sample order when calibrating, default: False
    stochastic_delay: `int`
        use stochastic delay when calibrating, default: 1, i.e., no stochastic delay

    Attributes
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        analog system
    K: `List[int]`
        number of filter coefficients per control signal
    downsample: `int`
        downsampling factor
    offset: `np.ndarray`
        offset of the analog system
    h: `List[List[np.ndarray]]`
        estimated filter coefficients

    """

    _index: int = 0
    downsample: int
    analog_system: cbadc.analog_system.AnalogSystem
    h: List[List[np.ndarray]]
    offset: np.ndarray
    training: Training
    control_signal_buffer: np.ndarray

    def __init__(
        self,
        h: List[List[np.ndarray]],
        K: List[int],
        reference_index: List[int],
        downsample: int = 1,
        method='rls',
        **kwargs,
    ):

        self.analog_system = cbadc.analog_system.AnalogSystem(
            np.zeros((1, 1)),
            np.zeros((1, len(h))),
            np.zeros((1, 1)),
            np.zeros((1, len(K))),
            np.zeros((len(K), 1)),
        )
        self.K = [k for k in K]
        self._K_max = max(self.K)
        self.downsample = downsample

        # self._temp_buffer = np.zeros((self.analog_system.M, self.downsample))
        # # reset control signal buffer
        # self.control_signals = np.zeros((self.analog_system.M, self._K_max))

        self.reference_index = [s for s in reference_index]
        if len(self.reference_index) != self.analog_system.L:
            raise ValueError("reference_index must have length L")

        self.calibration_indexes = []
        for l in range(self.analog_system.L):
            temp = []
            for m in range(self.analog_system.M):
                if m != self.reference_index[l]:
                    temp.append(m)
            self.calibration_indexes.append(temp)

        self.offset = np.zeros(self.analog_system.L, dtype=np.float64)
        self.h = [
            [np.zeros(self.K[m], dtype=np.float64) for m in range(self.analog_system.M)]
            for _ in range(self.analog_system.L)
        ]
        # size L x M x K[m]
        if len(h) != self.analog_system.L:
            raise ValueError("h must have length L")
        for l, h_l in enumerate(h):
            if len(h_l) != self.analog_system.M:
                raise ValueError("h must have length M")
            for m, h_lm in enumerate(h_l):
                if h_lm.size != self.K[m]:
                    raise ValueError(
                        f"h[l][m] must have length K[m], h_lm.size={h_lm.size}, K[m]={self.K[m]}"
                    )
                self.h[l][m][:] = np.copy(h_lm[:])

        self.projection = kwargs.get('projection', False)
        self.randomized_order = kwargs.get('randomized_order', False)
        self.stochastic_delay = kwargs.get('stochastic_delay', 1)

        if self.projection:
            # Projection Matrix
            self.P = [
                [
                    np.zeros((2 * self.K[m], self.K[m]))
                    for m in self.calibration_indexes[l]
                ]
                for l in range(self.analog_system.L)
            ]

            for l in range(self.analog_system.L):
                for m_index, m in enumerate(self.calibration_indexes[l]):
                    start_index = 0
                    end_index = self.K[m] // 2
                    for k in range(self.K[m]):
                        self.P[l][m_index][k, start_index:end_index] = h[l][
                            self.reference_index[l]
                        ][::-1][start_index - end_index :]
                        if k < self.K[m] // 2:
                            end_index += 1
                        else:
                            start_index += 1
                    self.P[l][m_index] = np.dot(
                        self.P[l][m_index].T,
                        np.dot(
                            np.linalg.pinv(
                                np.dot(self.P[l][m_index], self.P[l][m_index].T)
                            ),
                            self.P[l][m_index],
                        ),
                    )

        # setup training statistics object
        self.training = Training()
        self._training_index = 0

        if method == 'rls':
            # Covariance matrix for recursive least squares
            self.training.method = 'rls'
            self.training.rls_delta = kwargs.get("delta", 1e5)
            self.training.rls_lambda = kwargs.get(
                "learning_rate", FixedStepSize(0.99999)
            )
            self.training.k_sum = (
                np.sum([self.K[m] for m in self.calibration_indexes[0]]) + 1
            ) * self.analog_system.L
            self.training.V = (
                np.eye(
                    self.training.k_sum,
                    dtype=np.float64,
                )
                / self.training.rls_delta
            )
        elif method == 'lms':
            self.training.method = 'lms'
            self.training.lms_learning_rate = kwargs.get(
                "learning_rate", FixedStepSize(1e-3)
            )
            self.training.lms_momentum = kwargs.get("momentum", 0.9)
            self._first_moment_update = [
                [np.zeros(self.K[m]) for m in self.calibration_indexes[l]]
                for l in range(self.analog_system.L)
            ]
            self._first_moment_update_offset = np.zeros(self.analog_system.L)
        elif method == 'adam':
            self.training.method = 'adam'
            self.training.adam_learning_rate = kwargs.get(
                "learning_rate", FixedStepSize(1e-3)
            )
            self.training.adam_beta1 = kwargs.get("beta1", 0.9)
            self.training.adam_beta2 = kwargs.get("beta2", 0.999)
            self.training.adam_epsilon = kwargs.get("epsilon", 1e-8)
            self._first_moment_update = [
                [np.zeros(self.K[m]) for m in self.calibration_indexes[l]]
                for l in range(self.analog_system.L)
            ]
            self._first_moment_update_offset = np.zeros(self.analog_system.L)
            self._second_moment_update = [
                [np.zeros(self.K[m]) for m in self.calibration_indexes[l]]
                for l in range(self.analog_system.L)
            ]
            self._second_moment_update_offset = np.zeros(self.analog_system.L)
            self.training.adam_t = 1
        else:
            raise ValueError(f"Unknown method: {method}")

    def number_of_filter_coefficients(self) -> int:
        """Number of non-zero filter coefficients

        Returns
        -------
        `int`
            total number of non-zero filter coefficients
        """
        sum: int = 0
        for l in range(self.analog_system.L):
            for m in range(self.analog_system.M):
                sum += np.sum(self.h[l][m] != 0)
        return sum

    def impulse_response(self):
        """Return the filter's impulse response

        Returns
        -------
        `array_like`, shape=(L, K3, M)
            the impulse response
        """
        # (self.analog_system.L, self.analog_system.M, self.K[m])
        return self.h[:][:][::-1]

    def convolve(self, filter: np.ndarray):
        """Shape :math:`\mathbf{h}` filter by convolving with filter

        Parameters
        ----------
        filter: `array_like`, shape=(K)
            filter to be applied for each digital control filter
            equivalently.
        """
        for l in range(self.analog_system.L):
            for m in self.calibration_indexes[l]:
                # self.h.shape -> (L, M, K[m])
                temp = np.convolve(self.h[l][m][:], filter, mode="full")
                if temp.size == self.K[m]:
                    self.h[l][m][:] = temp
                else:
                    mid_point = temp.size // 2
                    half_length = self.K[m] // 2
                    self.h[l][m][:] = temp[
                        (mid_point - half_length) : (mid_point + half_length)
                    ]

    def write_C_header(self, filename: str):
        raise NotImplementedError

    def fir_filter_transfer_function(self, Ts: float = 1.0):
        """Compute the FFT of the system impulse response (FIR filter coefficients).

        Parameters
        ----------
        Ts: `float`
            the sample period of the corresponding impulse response.

        Returns
        -------
        `array_like`, shape=(L, M, K[m] // 2)
            the FFT of the corresponding impulse responses.
        """
        frequency_response = [
            [np.fft.rfft(self.h[l][m][:]) for m in range(self.analog_system.M)]
            for l in range(self.analog_system.L)
        ]
        frequencies = [
            [np.fft.rfftfreq(self.K[m]) for m in range(self.analog_system.M)]
            for l in range(self.analog_system.L)
        ]
        return (frequencies, frequency_response)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        # Check if control signal iterator is set.
        if self.training_signals is None or self.testing_signals is None:
            raise Exception("No iterators set.")
        return self.estimate(next(self.testing_signals))

    def estimate(self, control_signal: List[np.ndarray]) -> np.ndarray:
        """Estimate the input signal from the control signal."""
        res = np.zeros_like(self.offset)
        for l in range(self.analog_system.L):
            for m in range(self.analog_system.M):
                res[l] = res[l] + np.dot(self.h[l][m][:], control_signal[m][:])
        return res + self.offset[:]

    def train(self, training_iterations: int = 1) -> np.ndarray:
        """Train the adaptive filter

        Parameters
        ----------
        training_iterations: `int`
            number of training iterations
        """
        # Check if control signal iterator is set.
        if self.training_signals is None:
            raise Exception("No training iterator set.")

        if self.training.method == 'lms':
            training_error = self._lms(training_iterations)
        elif self.training.method == 'rls':
            training_error = self._rls(training_iterations)
        elif self.training.method == 'adam':
            training_error = self._adam(training_iterations)
        else:
            raise ValueError(f"Unknown method: {self.training.method}")

        self.training.training_iterations += training_iterations

        # return average training error
        return training_error / training_iterations

    def _gradient(
        self, control_signal: List[np.ndarray], observation_signal: np.ndarray
    ):
        return (
            [
                [
                    2 * observation_signal[l] * control_signal[m][:]
                    for m in self.calibration_indexes[l]
                ]
                for l in range(self.analog_system.L)
            ],
            2 * observation_signal[:],
        )

    def _lms(self, batch_size: int):
        training_error = np.zeros(self.analog_system.L)
        batch = [
            [np.zeros(self.K[m]) for m in self.calibration_indexes[l]]
            for l in range(self.analog_system.L)
        ]
        offset_batch = np.zeros(self.analog_system.L)

        for _ in range(batch_size):
            control_signal = next(self.training_signals)
            error_signal = self.estimate(control_signal)
            gradient = self._gradient(control_signal, error_signal)
            offset_batch = offset_batch + gradient[1][:]
            training_error = training_error + error_signal
            for l in range(self.analog_system.L):
                for m_index in range(len(self.calibration_indexes[l])):
                    batch[l][m_index][:] = (
                        batch[l][m_index][:] + gradient[0][l][m_index][:]
                    )

        batch_training_rate = (
            self.training.lms_learning_rate(
                self.training.training_iterations // batch_size
            )
            / batch_size
        )
        self._first_moment_update = [
            [
                self.training.lms_momentum * self._first_moment_update[l][m][:]
                - batch_training_rate * batch[l][m][:]
                for m in range(len(self.calibration_indexes[l]))
            ]
            for l in range(self.analog_system.L)
        ]
        self._first_moment_update_offset = (
            self.training.lms_momentum * self._first_moment_update_offset[:]
            - batch_training_rate * offset_batch[:]
        )
        for l in range(self.analog_system.L):
            for m_index, m in enumerate(self.calibration_indexes[l]):
                self.h[l][m][:] = (
                    self.h[l][m][:] + self._first_moment_update[l][m_index][:]
                )
                if self.projection:
                    self.h[l][m][:] = np.dot(self.P[l][m_index], self.h[l][m][:])
        self.offset = self.offset + self._first_moment_update_offset

        return training_error

    def _adam(self, batch_size: int):
        training_error = np.zeros(self.analog_system.L)
        first_moment_batch = [
            [np.zeros(self.K[m]) for m in self.calibration_indexes[l]]
            for l in range(self.analog_system.L)
        ]
        second_moment_batch = [
            [np.zeros(self.K[m]) for m in self.calibration_indexes[l]]
            for l in range(self.analog_system.L)
        ]
        first_moment_offset_batch = np.zeros(self.analog_system.L)
        second_moment_offset_batch = np.zeros(self.analog_system.L)
        # Compute batches
        for _ in range(batch_size):
            control_signal = next(self.training_signals)
            error_signal = self.estimate(control_signal)
            gradient = self._gradient(control_signal, error_signal)
            training_error = training_error + error_signal
            for l in range(self.analog_system.L):
                for m_index in range(len(self.calibration_indexes[l])):
                    first_moment_batch[l][m_index][:] = (
                        first_moment_batch[l][m_index][:] + gradient[0][l][m_index][:]
                    )
                    second_moment_batch[l][m_index][:] = second_moment_batch[l][
                        m_index
                    ][:] + np.power(gradient[0][l][m_index][:], 2)
            first_moment_offset_batch = first_moment_offset_batch + gradient[1][:]
            second_moment_offset_batch = second_moment_offset_batch + np.power(
                gradient[1][:], 2
            )
        # Update parameters

        self._first_moment_update_offset = (
            self.training.adam_beta1 * self._first_moment_update_offset[:]
            + (1.0 - self.training.adam_beta1) * first_moment_offset_batch / batch_size
        )
        self._second_moment_update_offset = (
            self.training.adam_beta2 * self._second_moment_update_offset[:]
            + (1.0 - self.training.adam_beta2) * second_moment_offset_batch / batch_size
        )

        first_moment_estimate_offset = self._first_moment_update_offset / (
            1.0 - np.power(self.training.adam_beta1, float(self.training.adam_t))
        )
        second_moment_estimate_offset = self._second_moment_update_offset / (
            1.0 - np.power(self.training.adam_beta2, float(self.training.adam_t))
        )

        self._first_moment_update = [
            [
                self.training.adam_beta1 * self._first_moment_update[l][m][:]
                + (1.0 - self.training.adam_beta1)
                * first_moment_batch[l][m][:]
                / batch_size
                for m in range(len(self.calibration_indexes[l]))
            ]
            for l in range(self.analog_system.L)
        ]
        self._second_moment_update = [
            [
                self.training.adam_beta2 * self._second_moment_update[l][m][:]
                + (1.0 - self.training.adam_beta2)
                * second_moment_batch[l][m][:]
                / batch_size
                for m in range(len(self.calibration_indexes[l]))
            ]
            for l in range(self.analog_system.L)
        ]

        first_moment_estimate = [
            [
                self._first_moment_update[l][m][:]
                / (
                    1.0
                    - np.power(self.training.adam_beta1, float(self.training.adam_t))
                )
                for m in range(len(self.calibration_indexes[l]))
            ]
            for l in range(self.analog_system.L)
        ]
        second_moment_estimate = [
            [
                self._second_moment_update[l][m][:]
                / (
                    1.0
                    - np.power(self.training.adam_beta2, float(self.training.adam_t))
                )
                for m in range(len(self.calibration_indexes[l]))
            ]
            for l in range(self.analog_system.L)
        ]

        self.training.adam_t += 1

        for l in range(self.analog_system.L):
            for m_index, m in enumerate(self.calibration_indexes[l]):
                self.h[l][m][:] = self.h[l][m][:] - self.training.adam_learning_rate(
                    self.training.training_iterations // batch_size
                ) * first_moment_estimate[l][m_index][:] / (
                    np.sqrt(second_moment_estimate[l][m_index][:])
                    + self.training.adam_epsilon
                )
                if self.projection:
                    self.h[l][m][:] = np.dot(self.P[l][m_index], self.h[l][m][:])
        self.offset = self.offset - self.training.adam_learning_rate(
            self.training.training_iterations // batch_size
        ) * first_moment_estimate_offset / (
            np.sqrt(second_moment_estimate_offset) + self.training.adam_epsilon
        )

        return training_error

    def _vectorize_control_signal(self, control_signal: List[np.ndarray]):
        # ones such that offset is included
        vectorized_control_signal = np.ones(
            np.sum([self.K[m] for m in self.calibration_indexes[0]]) + 1,
            dtype=np.float64,
        )
        k_sum = 0
        for m in self.calibration_indexes[0]:
            vectorized_control_signal[k_sum : k_sum + self.K[m]] = control_signal[m][:]
            k_sum += self.K[m]
        if self.analog_system.L > 1:
            return np.hstack(
                [vectorized_control_signal for _ in range(self.analog_system.L)]
            )
        return vectorized_control_signal

    def _de_vectorize_g(self, vectorize_g):
        g = [
            [np.zeros(self.K[m]) for m in self.calibration_indexes[l]]
            for l in range(self.analog_system.L)
        ]
        for l in range(self.analog_system.L):
            k_sum = 0
            for m_index, m in enumerate(self.calibration_indexes[l]):
                temp_index = l * self.analog_system.M * np.sum(self.K) + k_sum
                g[l][m_index][:] = vectorize_g[temp_index : temp_index + self.K[m]]
                k_sum += self.K[m]
        return g

    def _rls(self, training_iterations: int):
        """Train the adaptive filter using recursive least squares

        Parameters
        ----------
        training_iterations: `int`
            number of training iterations
        """
        training_error = np.zeros(self.analog_system.L)
        for _ in range(training_iterations):
            control_signal = next(self.training_signals)
            vectorized_control_signal = self._vectorize_control_signal(control_signal)
            error_signal = self.estimate(control_signal)
            training_error[:] = training_error[:] + error_signal[:]

            alpha = np.dot(self.training.V, vectorized_control_signal)
            g = alpha / (
                self.training.rls_lambda(
                    self.training.training_iterations // training_iterations
                )
                + np.inner(vectorized_control_signal, alpha)
            )
            self.training.V = (
                self.training.V - np.outer(g, alpha)
            ) / self.training.rls_lambda(
                self.training.training_iterations // training_iterations
            )

            de_vectorized_g = self._de_vectorize_g(g)
            for l in range(self.analog_system.L):
                self.offset[l] = self.offset[l] - error_signal[l] * g[-1]
                for m_index, m in enumerate(self.calibration_indexes[l]):
                    self.h[l][m] = (
                        self.h[l][m] - de_vectorized_g[l][m_index] * error_signal[l]
                    )
                    if self.projection:
                        self.h[l][m] = np.dot(self.P[l][m_index], self.h[l][m])

        return training_error

    def set_iterators(
        self,
        training_signals: Iterator[np.ndarray],
        testing_signals: Iterator[np.ndarray],
        buffer_size_training: int = 1 << 16,
        buffer_size_testing: int = 0,
        **kwargs,
    ):
        """Set the iterators for the training and testing signals.

        Parameters
        ----------
        training_signals: `Iterator[np.ndarray]`
            iterator for the training signals
        testing_signals: `Iterator[np.ndarray]`
            iterator for the testing signals
        buffer_size_training: `int`
            size of the training buffer, defaults to 2^16
        buffer_size_testing: `int`
            size of the testing buffer, defaults to 0 (no buffer)
        randomized_order: `bool`
            if True, the training signals are presented in a randomized order
        stochastic_delay: `int`
            if > 1, the training signals are presented with a random delay no larger than this value
        """
        self.training_signals = ControlSignalBuffer(
            self.analog_system.M,
            self.K,
            buffer_size=buffer_size_training,
            downsample=self.downsample,
            randomized_order=kwargs.get('randomized_order', self.randomized_order),
            stochastic_delay=kwargs.get('stochastic_delay', 1),
        )
        self.training_signals(training_signals)

        if kwargs.get('warm_up', False):
            self.training_signals.warm_up()

        self.testing_signals = ControlSignalBuffer(
            self.analog_system.M,
            self.K,
            buffer_size=buffer_size_testing,
            downsample=self.downsample,
            randomized_order=False,
            stochastic_delay=self.stochastic_delay,
        )
        self.testing_signals(testing_signals)

    def set_testing_iterator(
        self,
        testing_signals: Iterator[np.ndarray],
        buffer_size_testing: int = 0,
    ):
        """Set the testing iteration

        Parameters
        ----------
        testing_iteration: `int`
            testing iteration
        """
        self.testing_signals = ControlSignalBuffer(
            self.analog_system.M,
            self.K,
            buffer_size=buffer_size_testing,
            downsample=self.downsample,
            randomized_order=False,
            stochastic_delay=self.stochastic_delay,
        )
        self.testing_signals(testing_signals)

    def __call__(
        self,
        training_signals: Iterator[np.ndarray],
        testing_signals: Iterator[np.ndarray],
        buffer_size_training: int = 1 << 16,
        buffer_size_testing: int = 0,
        **kwargs,
    ):
        self.set_iterators(
            training_signals,
            testing_signals,
            buffer_size_training,
            buffer_size_testing,
            **kwargs,
        )

    def noise_transfer_function(self, omega: np.ndarray):
        raise NotImplementedError

    def signal_transfer_function(self, omega: np.ndarray):
        raise NotImplementedError

    def control_signal_transfer_function(self, omega: np.ndarray):
        raise NotImplementedError

    def save(self, filename: str):
        """Pickle object for later use.

        Uses :py:func:`cbadc.utilities.pickle_load`
        to save object for later use.

        Parameters
        ----------
        filename: `str`
            filename to save object to.
        """
        cbadc.utilities.pickle_dump(self, filename)
