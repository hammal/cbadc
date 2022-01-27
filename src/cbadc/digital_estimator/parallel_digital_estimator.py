"""The digital parallel estimator."""
import cbadc
import numpy as np
import logging
from .batch_estimator import BatchEstimator
from ._filter_coefficients import FilterComputationBackend

logger = logging.getLogger(__name__)


class ParallelEstimator(BatchEstimator):
    """Parallelized batch estimator implementation.

    The parallel estimator estimates a filtered version
    :math:`\hat{\mathbf{u}}(t)` (shaped by :py:func:`signal_transfer_function`)
    of the input signal :math:`\mathbf{u}(t)` from a sequence of control
    signals :math:`\mathbf{s}[k]`.

    Specifically, the parallel estimator is a modified version of the default
    estimator :py:class:`cbadc.digital_estimator.BatchEstimator` where the
    the filter matrices are diagonalized enabling a more efficient and
    possible parallelizable filter implementation. The estimate is computed as

    :math:`\hat{\mathbf{u}}(k T)[\ell] = \sum_{n=0}^N f_w[n] \cdot \overrightarrow{\mathbf{m}}[k][n] + b_w[n] \cdot \overleftarrow{\mathbf{m}}[k][n]`

    where

    :math:`\overrightarrow{\mathbf{m}}[k][n] = f_a[n] \cdot \overrightarrow{\mathbf{m}}[k-1][n] + \sum_{m=0}^{M-1} f_b[n, m] \cdot \mathbf{s}[k-1][m]`

    and

    :math:`\overleftarrow{\mathbf{m}}[k][n] = b_a \cdot \overrightarrow{\mathbf{m}}[k+1][n] + \sum_{m=0}^{M-1} b_b[n, m] \cdot \mathbf{s}[k][m]`.

    Furthermore, :math:`f_a, b_a \in \mathbb{R}^{N}`, :math:`f_b, b_b \in \mathbb{R}^{N \\times M}`,
    and :math:`f_w, b_w \in \mathbb{R}^{L \\times N}` are the precomputed filter coefficient formed
    from the filter coefficients as in :py:class:`cbadc.digital_estimator.BatchEstimator`.

    Parameters
    ----------
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system (necessary to compute the estimators filter coefficients).
    digital_control : :py:class:`cbadc.digital_control.DigitalControl`
        a digital control (necessary to determine the corresponding DAC waveform).
    eta2 : `float`
        the :math:`\eta^2` parameter determines the bandwidth of the estimator.
    K1 : `int`
        batch size.
    K2 : `int`, `optional`
        lookahead size, defaults to 0.
    stop_after_number_of_iterations : `int`
        determine a max number of iterations by the iterator, defaults to :math:`2^{63}`.
    Ts: `float`, `optional`
        the sampling time, defaults to the time period of the digital control.
    mid_point: `bool`, `optional`
        set samples in between control updates, i.e., :math:`\hat{u}(kT + T/2)`, defaults to False.
    downsample: `int`, `optional`
        set a downsampling factor compared to the control signal rate, defaults to 1, i.e.,
        no downsampling.
    solver_type: :py:class:`cbadc.digital_estimator._filter_coefficients.FilterComputationBackend`
        determine which solver type to use when computing filter coefficients.


    Attributes
    ----------
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        analog system as in :py:class:`cbadc.analog_system.AnalogSystem` or from
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
    downsample: `int`, `optional`
        The downsampling factor compared to the rate of the control signal.
    mid_point: `bool`
        estimated samples shifted in between control updates, i.e., :math:`\hat{u}(kT + T/2)`.
    K1 : `int`
        number of samples per estimate batch.
    K2 : `int`
        number of lookahead samples per computed batch.
    Ts : `float`
        spacing between samples in seconds.
    Af : `array_like`, shape=(N, N), readonly
        The Af matrix
    Ab : `array_like`, shape=(N, N), readonly
        The Ab matrix
    Bf : `array_like`, shape=(N, M), readonly
        The Bf matrix
    Bb : `array_like`, shape=(N, M), readonly
        The Bb matrix
    WT : `array_like`, shape=(L, N), readonly
        The W matrix transposed.
    f_a : `array_like`, shape=(N), readonly
        The :math:`f_a` vector.
    b_a : `array_like`, shape=(N), readonly
        The :math:`b_a` vector.
    f_b : `array_like`, shape=(N, M), readonly
        The :math:`f_b` matrix.
    b_b : `array_like`, shape=(N, M), readonly
        The :math:`b_b` matrix.
    f_w : `array_like`, shape=(L, N), readonly
        The :math:`f_w` matrix.
    b_w : `array_like`, shape=(L, N), readonly
        The :math:`b_w` matrix.
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
        K2: int = 0,
        stop_after_number_of_iterations: int = (1 << 63),
        Ts: float = None,
        mid_point: bool = False,
        downsample: int = 1,
        solver_type: FilterComputationBackend = FilterComputationBackend.mpmath,
    ):
        # Check inputs
        if K1 < 1:
            raise Exception("K1 must be a positive integer.")
        self.K1 = K1
        if K2 < 0:
            raise Exception("K2 must be a non negative integer.")
        self.K2 = K2
        self.K3 = K1 + K2
        self._filter_lag = -1
        self.analog_system = analog_system
        self.digital_control = digital_control
        if eta2 < 0:
            raise Exception("eta2 must be non negative.")
        if Ts:
            self.Ts = Ts
        else:
            self.Ts = digital_control.clock.T
        self.eta2 = eta2
        self.control_signal = None

        if downsample != 1:
            raise NotImplementedError(
                "Downsampling currently not implemented for ParallelEstimator"
            )

        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        self._estimate_pointer = self.K1

        # For transfer functions
        self.eta2Matrix = np.eye(self.analog_system.CT.shape[0]) * self.eta2

        self._stop_iteration = False

        self.mid_point = mid_point

        # Initialize filters
        self.solver_type = solver_type
        self._compute_filter_coefficients(analog_system, digital_control, eta2)
        self._allocate_memory_buffers()

    def _compute_filter_coefficients(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
        eta2: float,
    ):
        # Compute filter coefficients from base class
        BatchEstimator._compute_filter_coefficients(
            self, analog_system, digital_control, eta2
        )
        # Parallelize
        temp, Q_f = np.linalg.eig(self.Af)
        self.forward_a = np.array(temp, dtype=np.complex128)
        Q_f_inv = np.linalg.pinv(Q_f, rcond=1e-100)
        temp, Q_b = np.linalg.eig(self.Ab)
        self.backward_a = np.array(temp, dtype=np.complex128)
        Q_b_inv = np.linalg.pinv(Q_b, rcond=1e-100)

        self.forward_b = np.array(np.dot(Q_f_inv, self.Bf), dtype=np.complex128)
        self.backward_b = np.array(np.dot(Q_b_inv, self.Bb), dtype=np.complex128)

        self.forward_w = -np.array(np.dot(self.WT, Q_f), dtype=np.complex128)
        self.backward_w = np.array(np.dot(self.WT, Q_b), dtype=np.complex128)

    def _allocate_memory_buffers(self):
        # Allocate memory buffers
        self._control_signal = np.zeros((self.K3, self.analog_system.M), dtype=np.int8)
        self._estimate = np.zeros((self.K1, self.analog_system.L), dtype=np.double)
        self._control_signal_in_buffer = 0
        self._mean = np.zeros((self.analog_system.N), dtype=np.complex128)

    def _compute_batch(self):
        logger.info("Computing Batch")
        mean: np.complex128 = np.complex128(0)
        # check if ready to compute buffer
        if self._control_signal_in_buffer < self.K3:
            raise Exception("Control signal buffer not full")

        self._estimate = np.zeros((self.K1, self.analog_system.L), dtype=np.double)

        for n in range(self.analog_system.N):
            mean = self._mean[n]
            for k1 in range(self.K1):
                for l in range(self.analog_system.L):
                    self._estimate[k1, l] += np.real(self.forward_w[l, n] * mean)
                mean = self.forward_a[n] * mean
                for m in range(self.analog_system.M):
                    if self._control_signal[k1, m]:
                        mean += self.forward_b[n, m]
                    else:
                        mean -= self.forward_b[n, m]
            self._mean[n] = mean
            mean = np.complex128(0.0)
            for k3 in range(self.K3 - 1, -1, -1):
                mean = self.backward_a[n] * mean
                for m in range(self.analog_system.M):
                    if self._control_signal[k3, m]:
                        mean += self.backward_b[n, m]
                    else:
                        mean -= self.backward_b[n, m]
                if k3 < self.K1:
                    for l in range(self.analog_system.L):
                        self._estimate[k3, l] += np.real(self.backward_w[l, n] * mean)
        self._control_signal = np.roll(self._control_signal, -self.K1, axis=0)
        self._control_signal_in_buffer -= self.K1

    def _input(self, s: np.ndarray) -> bool:
        if self._control_signal_in_buffer == (self.K3):
            raise Exception(
                "Input buffer full. You must compute batch before adding more control signals"
            )
        self._control_signal[self._control_signal_in_buffer, :] = np.asarray(
            s, dtype=np.int8
        )
        self._control_signal_in_buffer += 1
        return self._control_signal_in_buffer > (self.K3 - 1)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        # Check if control signal iterator is set.
        if self.control_signal is None:
            raise Exception("No iterator set.")

        # Check if the end of prespecified size
        if self.number_of_iterations < self._iteration:
            raise StopIteration
        self._iteration += 1

        # Check if there are estimates in the estimate buffer
        if self._estimate_pointer < self.K1:
            temp = np.array(self._estimate[self._estimate_pointer, :], dtype=np.double)
            self._estimate_pointer += 1
            return temp
        # Check if stop iteration has been raised in previous batch
        if self._stop_iteration:
            logger.info("StopIteration received by estimator.")
            raise StopIteration
        # Otherwise start receiving control signals
        full = False

        # Fill up batch with new control signals.
        while not full:
            # next(self.control_signal) calls the control signal
            # generator and thus recives new control
            # signal samples
            try:
                control_signal_sample = next(self.control_signal)
            except RuntimeError:
                self._stop_iteration = True
                control_signal_sample = np.zeros((self.analog_system.M), dtype=np.int8)
            full = self._input(control_signal_sample)

        # Compute new batch of K1 estimates
        self._compute_batch()
        # adjust pointer to indicate that estimate buffer
        # is non empty
        self._estimate_pointer -= self.K1

        # recursively call itself to return new estimate
        return self.__next__()

    def __str__(self):
        return f"Parallel estimator is parameterized as \neta2 = {self.eta2:.2f}, {10 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},\nand\nnumber_of_iterations = {self.number_of_iterations}\nResulting in the filter coefficients\nf_a = \n{self.forward_a},\nb_a = \n{self.backward_b},\nf_b = \n{self.forward_b},\nb_b = \n{self.backward_b},\nf_w = \n{self.forward_w},\nand b_w = \n{self.backward_w}."
