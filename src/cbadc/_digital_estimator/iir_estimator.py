"""The digital IIR estimator"""

import cbadc
import logging
import numpy as np
from .batch_estimator import BatchEstimator
from ._filter_coefficients import FilterComputationBackend

logger = logging.getLogger(__name__)


class IIRFilter(BatchEstimator):
    """IIR filter implementation of the digital estimator.

    Specifically, the IIR filter estimator estimates a filtered version :math:`\hat{\mathbf{u}}(t)` (shaped by
    :py:func:`signal_transfer_function`) of the
    input signal :math:`\mathbf{u}(t)` from a sequence of control signals :math:`\mathbf{s}[k]`.

    Specifically, the estimate is of the form

    :math:`\hat{\mathbf{u}}(k T) = - \mathbf{W}^{\mathsf{T}} \overrightarrow{\mathbf{m}}_k + \sum_{\ell=0}^{K_2} \mathbf{h}[\ell] \mathbf{s}[k + \ell]`

    where

    :math:`\mathbf{h}[\ell]=\mathbf{W}^{\mathsf{T}} \mathbf{A}_b^\ell \mathbf{B}_b`

    :math:`\overrightarrow{\mathbf{m}}_k = \mathbf{A}_f \mathbf{m}_{k-1} + \mathbf{B}_f \mathbf{s}[k-1]`

    and :math:`\mathbf{W}^{\mathsf{T}}`, :math:`\mathbf{A}_b`,
    :math:`\mathbf{B}_b`, :math:`\mathbf{A}_f`, and :math:`\mathbf{B}_f`
    are computed based on the analog system, the sample period :math:`T_s`, and the
    digital control's DAC waveform as described in
    # page=67/>`_.
    `control-bounded converters <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf?sequence=1&isAllowed=y

    Parameters
    ----------
    analog_filter : :py:class:`cbadc.analog_filter.AnalogSystem`
        an analog system (necessary to compute the estimators filter coefficients).
    digital_control : :py:class:`cbadc.digital_control.DigitalControl`
        a digital control (necessary to determine the corresponding DAC waveform).
    eta2 : `float`
        the :math:`\eta^2` parameter determines the bandwidth of the estimator.
    K2 : `int`, `optional`
        lookahead size, defaults to 0.
    stop_after_number_of_iterations : `int`
        determine a max number of iterations by the iterator, defaults to  :math:`2^{63}`.
    Ts: `float`, `optional`
        the sampling time, defaults to the time period of the digital control.
    mid_point: `bool`, `optional`
        set samples in between control updates, i.e., :math:`\hat{u}(kT + T/2)`, defaults to False.
    downsample: `int`, `optional`
        specify down sampling rate in relation to the control period :math:`T`, defaults to 1, i.e.,
        no down sampling.
    solver_type: :py:class:`cbadc.digital_estimator._filter_coefficients.FilterComputationBackend`
        determine which solver type to use when computing filter coefficients.

    Attributes
    ----------
    analog_filter : :py:class:`cbadc.analog_filter.AnalogSystem`
        analog system as in :py:class:`cbadc.analog_filter.AnalogSystem` or from
        derived class.
    digital_control : :py:class:`cbadc.digital_control.DigitalControl`
        digital control as in :py:class:`cbadc.digital_control.DigitalControl` or from
        derived class.
    eta2 : float
        eta2, or equivalently :math:`\eta^2`, sets the bandwidth of the estimator.
    control_signal : :py:class:`cbadc.digital_control.DigitalControl`
        a iterator suppling control signals as :py:class:`cbadc.digital_control.DigitalControl`.
    number_of_iterations : `int`
        number of iterations until iterator raises :py:class:`StopIteration`.
    mid_point: `bool`
        estimated samples shifted in between control updates, i.e., :math:`\hat{u}(kT + T/2)`.
    K2 : `int`
        number of lookahead samples per computed batch.
    Ts : `float`
        spacing between samples in seconds.
    Af : `array_like`, shape=(N, N)
        The Af matrix.
    Ab : `array_like`, shape=(N, N)
        The Ab matrix.
    Bf : `array_like`, shape=(N, M)
        The Bf matrix.
    Bb : `array_like`, shape=(N, M)
        The Bb matrix.
    WT : `array_like`, shape=(L, N)
        The W matrix transposed.
    h : `array_like`, shape=(L, K2, M)
        filter impulse response.
    downsample: `int`
        down sampling rate in relation to the control period :math:`T`.
    solver_type: :py:class:`cbadc.digital_estimator._filter_coefficients.FilterComputationBackend`
        The solver used for computing the filter coefficients.

    Yields
    ------
    `array_like`, shape=(L,)
        an input estimate sample :math:`\hat{\mathbf{u}}(t)`

    """

    def __init__(
        self,
        analog_filter: cbadc.analog_filter.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
        eta2: float,
        K2: int,
        stop_after_number_of_iterations: int = (1 << 63),
        Ts: float = None,
        mid_point: bool = False,
        downsample: int = 1,
        solver_type: FilterComputationBackend = FilterComputationBackend.mpmath,
        modulation_frequency: float = None,
        *args,
        **kwargs,
    ):
        """Initializes filter coefficients"""
        if K2 < 0:
            raise Exception("K2 must be non negative integer.")
        self.K2 = K2
        self._filter_lag = self.K2 - 2
        self.analog_filter = analog_filter
        self.digital_control = digital_control
        if eta2 < 0:
            raise Exception("eta2 must be non negative.")
        self.eta2 = eta2
        self.control_signal = None
        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        if Ts:
            self.Ts = Ts
        else:
            self.Ts = digital_control.clock.T

        self.downsample = int(downsample)

        self.mid_point = mid_point

        # For transfer functions
        self.eta2Matrix = np.eye(self.analog_filter.CT.shape[0]) * self.eta2

        # Compute filter coefficients
        self.solver_type = solver_type
        BatchEstimator._compute_filter_coefficients(
            self, analog_filter, digital_control, eta2
        )

        # Initialize filter
        self.h = np.zeros(
            (self.analog_filter.L, self.K2, self.analog_filter.M), dtype=np.double
        )
        # Compute lookback
        temp2 = np.copy(self.Bb)
        for k2 in range(self.K2):
            self.h[:, k2, :] = np.dot(self.WT, temp2)
            temp2 = np.dot(self.Ab, temp2)
        self._control_signal_valued = np.zeros((self.K2, self.analog_filter.M))
        self._mean = np.zeros(self.analog_filter.N, dtype=np.double)

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
        self._iteration += 1
        if self.number_of_iterations and self.number_of_iterations < self._iteration:
            raise StopIteration

        # Rotate control_signal vector
        self._control_signal_valued = np.roll(self._control_signal_valued, -1, axis=0)

        # insert new control signal
        try:
            temp = self.control_signal.__next__()
        except RuntimeError:
            logger.warning("Estimator received Stop Iteration")
            raise StopIteration

        self._control_signal_valued[-1, :] = np.asarray(2 * temp - 1)

        # self._control_signal_valued.shape -> (K2, M)
        # self.h.shape -> (L, K2, M)
        result = -np.dot(self.WT, self._mean)
        self._mean = np.dot(self.Af, self._mean) + np.dot(
            self.Bf, self._control_signal_valued[0, :]
        )
        if ((self._iteration - 1) % self.downsample) == 0:
            self._time_index += 1
            return (
                np.tensordot(self.h, self._control_signal_valued, axes=((1, 2), (0, 1)))
                + result
            )
            # return np.einsum('ijk,jk', self.h, self._control_signal_valued) + result
        return self.__next__()

    def lookahead(self):
        """Return lookahead size :math:`K2`

        Returns
        -------
        int
            lookahead size.
        """
        return self.K2

    def __str__(self):
        return f"IIR estimator is parameterized as \neta2 = {self.eta2:.2f}, {10 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK2 = {self.K2},\nand\nnumber_of_iterations = {self.number_of_iterations}.\nResulting in the filter coefficients\nAf = \n{self.Af},\nBf = \n{self.Bf},WT = \n{self.WT},\n and h = \n{self.h}."
