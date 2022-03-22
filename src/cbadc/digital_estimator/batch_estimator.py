"""The digital batch estimator
"""
from typing import Iterator

from scipy.misc import derivative
import cbadc
from cbadc.digital_estimator._filter_coefficients import (
    compute_filter_coefficients,
    FilterComputationBackend,
)
import cbadc.utilities
import scipy.integrate
import numpy as np
import sympy as sp
import logging

logger = logging.getLogger(__name__)


class BatchEstimator(Iterator[np.ndarray]):
    """Batch estimator implementation.

    The digital estimator estimates a filtered version
    :math:`\hat{\mathbf{u}}(t)` (shaped by :py:func:`signal_transfer_function`)
    of the input signal :math:`\mathbf{u}(t)` from a sequence of control
    signals :math:`\mathbf{s}[k]`.

    Specifically, the estimates are computed as

    :math:`\overrightarrow{\mathbf{m}}[k] = \mathbf{A}_f \overrightarrow{\mathbf{m}}[k-1] + \mathbf{B}_f \mathbf{s}[k-1]`,

    :math:`\overleftarrow{\mathbf{m}}[k] = \mathbf{A}_b \overrightarrow{\mathbf{m}}[k+1] + \mathbf{B}_b \mathbf{s}[k]`,

    and

    :math:`\hat{\mathbf{u}}(k T) = \mathbf{W}^\mathsf{T}\\left(\overleftarrow{\mathbf{m}}[k] -  \overrightarrow{\mathbf{m}}[k]\\right)`

    where :math:`\mathbf{A}_f, \mathbf{A}_b \in \mathbb{R}^{N \\times N}`,
    :math:`\mathbf{B}_f, \mathbf{B}_b \in \mathbb{R}^{N \\times M}`, and
    :math:`\mathbf{W}^\mathsf{T} \in \mathbb{R}^{L \\times N}` are the
    precomputed filter coefficient based on the choice of
    :py:class:`cbadc.analog_system.AnalogSystem` and
    :py:class:`cbadc.digital_control.DigitalControl`.

    Parameters
    ----------
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system (necessary to compute the estimators filter
        coefficients).
    digital_control : :py:class:`cbadc.digital_control.DigitalControl`
        a digital control (necessary to determine the corresponding DAC
        waveform).
    eta2 : `float`
        the :math:`\eta^2` parameter determines the bandwidth of the estimator.
    K1 : `int`
        batch size.
    K2 : `int`, `optional`
        lookahead size, defaults to 0.
    stop_after_number_of_iterations : `int`
        determine a max number of iterations by the iterator, defaults to
        :math:`2^{63}`.
    Ts: `float`, `optional`
        the sampling time, defaults to the time period of the digital control.
    mid_point: `bool`, `optional`
        set samples in between control updates, i.e., :math:`\hat{u}(kT + T/2)`
        , defaults to False.
    downsample: `int`, `optional`
        set a downsampling factor compared to the control signal rate,
        defaults to 1, i.e., no downsampling.
    solver_type: :py:class:`cbadc.digital_estimator._filter_coefficients.FilterComputationBackend`
        determine which solver type to use when computing filter coefficients.


    Attributes
    ----------

    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        analog system as in :py:class:`cbadc.analog_system.AnalogSystem` or
        from derived class.
    digital_control : :py:class:`cbadc.digital_control.DigitalControl`
        digital control as in :py:class:`cbadc.digital_control.DigitalControl`
        or from derived class.
    eta2 : float
        eta2, or equivalently :math:`\eta^2`, sets the bandwidth of the
        estimator.
    control_signal : :py:class:`cbadc.digital_control.DigitalControl`
        a iterator suppling control signals as
        :py:class:`cbadc.digital_control.DigitalControl`.
    number_of_iterations : `int`
        number of iterations until iterator raises :py:class:`StopIteration`.
    downsample: `int`, `optional`
        The downsampling factor compared to the rate of the control signal.
    mid_point: `bool`
        estimated samples shifted in between control updates, i.e.,
        :math:`\hat{u}(kT + T/2)`.
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
        The W matrix transposed
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

        if not np.allclose(self.analog_system.D, np.zeros_like(self.analog_system.D)):
            raise Exception(
                """Can't compute filter coefficients for system with non-zero
                D matrix. Consider chaining for removing D"""
            )

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
                "Downsampling currently not implemented for DigitalEstimator"
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
        self._ntf_lambda = None
        self._stf_lambda = None

    def filter_lag(self):
        """Return the lag of the filter.

        As the filter computes the estimate as

        ---------
        |   K2  |
        ---------
        ^
        |
        u_hat[k]

        Returns
        -------
        `int`
            The filter lag.

        """
        return self._filter_lag

    def warm_up(self, samples=0):
        """Warm up filter by population control signals.

        Effectively removes the filter lag.

        Parameters
        ----------
        samples: `int`, `optional`
            number of warmup samples, defaults to filter_lag
        """
        logger.debug("Warming up estimator.")

        if samples > 0:
            for _ in range(samples):
                self.__next__()
        else:
            while self._filter_lag > 0:
                _ = self.__next__()
                self._filter_lag -= 1

    def set_iterator(self, control_signal_sequence: Iterator[np.ndarray]):
        """Set iterator of control signals

        Parameters
        -----------
        control_signal_sequence : iterator
            a iterator which outputs a sequence of control signals.
        """
        self.control_signal = control_signal_sequence

    def _compute_filter_coefficients(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
        eta2: float,
    ):
        logger.info(
            f"Computing filter coefficients. Using solver type: {self.solver_type.name}"
        )

        Af, Ab, Bf, Bb, WT = compute_filter_coefficients(
            analog_system,
            digital_control,
            eta2,
            solver_type=self.solver_type,
            mid_point=self.mid_point,
        )
        self.Af = np.array(Af, dtype=np.float64).reshape(
            (analog_system.N, analog_system.N)
        )
        self.Bf = np.array(Bf, dtype=np.float64).reshape(
            (analog_system.N, analog_system.M)
        )
        self.Ab = np.array(Ab, dtype=np.float64).reshape(
            (analog_system.N, analog_system.N)
        )
        self.Bb = np.array(Bb, dtype=np.float64).reshape(
            (analog_system.N, analog_system.M)
        )
        self.WT = np.array(WT, dtype=np.float64).reshape(
            (analog_system.L, analog_system.N)
        )

    def _allocate_memory_buffers(self):
        # Allocate memory buffers
        self._control_signal = np.zeros((self.K3, self.analog_system.M), dtype=np.int8)
        self._estimate = np.zeros((self.K1, self.analog_system.L), dtype=np.double)
        self._control_signal_in_buffer = 0
        self._mean = np.zeros((self.K1 + 1, self.analog_system.N), dtype=np.double)

    def _compute_batch(self):
        logger.info("Computing batch.")
        temp_forward_mean = np.zeros(self.analog_system.N, dtype=np.double)
        # check if ready to compute buffer
        if self._control_signal_in_buffer < self.K3:
            raise Exception("Control signal buffer not full")
        # compute lookahead
        for k1 in range(self.K3 - 1, self.K1 - 1, -1):
            temp = np.dot(self.Ab, self._mean[self.K1, :]) + np.dot(
                self.Bb, self._control_signal[k1, :]
            )
            for n in range(self.analog_system.N):
                self._mean[self.K1, n] = temp[n]
        # compute forward recursion
        for k2 in range(self.K1):
            temp = np.dot(self.Af, self._mean[k2, :]) + np.dot(
                self.Bf, self._control_signal[k2, :]
            )
            if k2 < self.K1 - 1:
                for n in range(self.analog_system.N):
                    self._mean[k2 + 1, n] = temp[n]
            else:
                for n in range(self.analog_system.N):
                    temp_forward_mean[n] = temp[n]
        # compute backward recursion and estimate
        for k3 in range(self.K1 - 1, -1, -1):
            temp = np.dot(self.Ab, self._mean[k3 + 1, :]) + np.dot(
                self.Bb, self._control_signal[k3, :]
            )
            temp_estimate = np.dot(self.WT, temp - self._mean[k3, :])
            self._estimate[k3, :] = temp_estimate[:]
            self._mean[k3, :] = temp[:]
        # reset intital means
        for n in range(self.analog_system.N):
            self._mean[0, n] = temp_forward_mean[n]
            self._mean[self.K1, n] = 0
        # rotate buffer to make place for new control signals
        self._control_signal = np.roll(self._control_signal, -self.K1, axis=0)
        self._control_signal_in_buffer -= self.K1

    def _input(self, s: np.ndarray) -> bool:
        if self._control_signal_in_buffer == (self.K3):
            raise Exception(
                """Input buffer full. You must compute batch before adding
                more control signals"""
            )
        for m in range(self.analog_system.M):
            self._control_signal[self._control_signal_in_buffer, :] = np.asarray(
                2 * s - 1, dtype=np.int8
            )
        self._control_signal_in_buffer += 1
        return self._control_signal_in_buffer > (self.K3 - 1)

    def __call__(self, control_signal_sequence: Iterator[np.ndarray]):
        return self.set_iterator(control_signal_sequence)

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
            logger.warning("Warning: StopIteration received by estimator.")
            raise StopIteration
        # Otherwise start receiving control signals
        full = False

        # Fill up batch with new control signals.
        while not full:
            # next(self.control_signal) calls the control signal
            # iterator and thus recives new control
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

    def _lazy_initialise_ntf(self):
        logger.info("Computing analytical noise-transfer function")
        if self.analog_system._atf_lambda is None:
            self.analog_system._lazy_initialize_ATF()
        GH = self.analog_system._atf_s_matrix.transpose().conjugate()
        GGH = self.analog_system._atf_s_matrix * GH
        self._ntf_s_matrix = sp.simplify(
            GH * (GGH + self.eta2 * sp.eye(self.analog_system.N_tilde)).inv()
        )
        self._ntf_lambda = sp.lambdify((self.analog_system.omega), self._ntf_s_matrix)

    def noise_transfer_function(self, omega: np.ndarray):
        """Compute the noise transfer function (NTF) at the angular
        frequencies of the omega array.

        Specifically, computes

        :math:`\\text{NTF}( \omega) = \mathbf{G}( \omega)^\mathsf{H} \\left( \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N \\right)^{-1}`

        for each angular frequency in omega where where
        :math:`\mathbf{G}(\omega)\in\mathbb{R}^{N \\times L}` is the ATF
        matrix of the analog system and :math:`\mathbf{I}_N` represents a
        square identity matrix.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for
            evaluation.

        Returns
        -------
        `array_like`, shape=(L, N_tilde, K)
            return NTF evaluated at K different angular frequencies.
        """
        result = np.zeros(
            (self.analog_system.L, self.analog_system.N_tilde, omega.size),
            dtype=np.complex128,
        )
        # if self._ntf_lambda is None:
        #     self._lazy_initialise_ntf()
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o]))
            G = G.reshape((self.analog_system.N_tilde, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, :, index] = np.abs(
                np.dot(GH, np.linalg.pinv(GGH + self.eta2Matrix))
            )
            # result[:, :, index] = self._ntf_lambda(o)
        return result

    def _lazy_initialise_stf(self):
        logger.info("Computing analytical signal transfer function.")
        if self._ntf_lambda is None:
            self._lazy_initialise_ntf()
        self._stf_s_matrix = sp.simplify(
            self._ntf_s_matrix * self.analog_system._atf_s_matrix
        )
        self._stf_lambda = sp.lambdify((self.analog_system.omega), self._stf_s_matrix)

    def signal_transfer_function(self, omega: np.ndarray):
        """Compute the signal transfer function (STF) at the angular
        frequencies of the omega array.

        Specifically, computes

        :math:`\\text{STF}( \omega) = \mathbf{G}( \omega)^\mathsf{H} \\left( \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N \\right)^{-1} \mathbf{G}( \omega)`

        for each angular frequency in omega where where
        :math:`\mathbf{G}(\omega)\in\mathbb{R}^{N \\times L}` is the ATF
        matrix of the analog system and :math:`\mathbf{I}_N` represents a
        square identity matrix.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for
            evaluation.

        Returns
        -------
        `array_like`, shape=(L, K)
            return STF evaluated at K different angular frequencies.
        """
        result = np.zeros((self.analog_system.L, omega.size), dtype=np.complex128)
        # if self._stf_lambda is None:
        #     self._lazy_initialise_stf()
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o])).reshape(
                (self.analog_system.N_tilde, self.analog_system.L)
            )
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, index] = np.abs(
                np.dot(GH, np.dot(np.linalg.pinv(GGH + self.eta2Matrix), G))
            )
            # result[:, index] = self._stf_lambda(o)
        return result

    def control_signal_transfer_function(self, omega: np.ndarray):
        """Compute the control signal transfer function at the angular
        frequencies of the omega array.

        Specifically, computes

        :math:`\\begin{pmatrix}\hat{u}_1(\omega) / s_1(\omega) & \\dots & \hat{u}_1(\omega) / s_M(\omega) \\\ \\vdots & \\ddots & \\vdots \\\ \hat{u}_L(\omega) / s_1(\omega) & \\dots & \hat{u}_L(\omega) / s_M(\omega)  \\end{pmatrix}= \mathbf{G}( \omega)^\mathsf{H} \\left( \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N \\right)^{-1} \\bar{\mathbf{G}}( \omega)`

        for each angular frequency in omega where where
        :math:`\\bar{\mathbf{G}}( \omega)=  \mathbf{C}^\mathsf{T} \\left(\mathbf{A} - i \omega \mathbf{I}_N\\right)^{-1} \mathbf{\\Gamma} \in\mathbb{R}^{N \\times M}`
        is the transfer function from the control signals to the output and :math:`\mathbf{I}_N` represents a
        square identity matrix.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for
            evaluation.

        Returns
        -------
        `array_like`, shape=(L, M, K)
            return STF evaluated at K different angular frequencies.
        """
        result = np.zeros((self.analog_system.L, self.analog_system.M, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o])).reshape(
                (self.analog_system.N_tilde, self.analog_system.L)
            )
            G_bar = self.analog_system.control_signal_transfer_function_matrix(
                np.array([o])
            ).reshape((self.analog_system.N_tilde, self.analog_system.M))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, index] = np.abs(
                np.dot(GH, np.dot(np.linalg.inv(GGH + self.eta2Matrix), G_bar))
            )
        return result

    def general_transfer_function(self, omega: np.ndarray):
        """Compute the general transfer functions from additive sources into each state varible
         to the estimates.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for
            evaluation.

        Returns
        -------
        `array_like`, shape=(L, N, K)
            return transfer function from each state N to each input estimate L for each frequency K.
        """
        #  shape=(N_tilde, N, K)
        stf_array = self.analog_system.transfer_function_matrix(omega, general=True)
        # shape=(L, N_tilde, K)
        ntf_array = self.noise_transfer_function(omega)
        return np.einsum('lnk,nxk->lxk', ntf_array, stf_array)

    def thermal_noise_estimate(self, noise_variances: np.ndarray, BW: np.ndarray):
        if len(noise_variances.shape) > 1:
            raise Exception("noise_variances must be a 1 dimensional vector")
        if any(noise_variances < 0):
            raise Exception("noise variance must be non-negative numbers.")
        if noise_variances.size != self.analog_system.N:
            raise Exception(
                f"N={self.analog_system.N} noise variances must be specified."
            )

        def _derivative(omega, x):
            _omega = np.array([omega])
            return (
                np.abs(self.general_transfer_function(_omega)).flatten(order="C") ** 2
            )

        integrated_tf = (
            scipy.integrate.solve_ivp(
                _derivative,
                (BW[0], BW[-1]),
                np.zeros(self.analog_system.L * self.analog_system.N),
            )
            .y[:, -1]
            .reshape((self.analog_system.L, self.analog_system.N), order="C")
        )
        for n, sigma_z_2 in enumerate(noise_variances):
            integrated_tf[:, n] *= sigma_z_2
        return integrated_tf

    def max_harmonic_estimate(self, BW: np.ndarray, SFDR: float):

        sfdr = cbadc.fom.snr_from_dB(SFDR)

        num = 1000
        _omega = np.logspace(np.log(BW[0]), np.log(BW[-1]), num)
        tfs = np.abs(self.general_transfer_function(_omega))

        max_input = np.max(tfs)
        # value =

        temp = np.max(tfs, axis=2) / max_input
        # this could probably be normalized by somethings like np.linalg.norm(temp)
        local_harmonics = 1.0 / temp
        local_harmonics /= np.linalg.norm(local_harmonics, ord=1)
        return local_harmonics / sfdr

    def __str__(self):
        return f"""Digital estimator is parameterized as
        \neta2 = {self.eta2:.2f}, {10 * np.log10(self.eta2):.0f} [dB],
        \nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},
        \nand\nnumber_of_iterations = {self.number_of_iterations}
        \nResulting in the filter coefficients\nAf = \n{self.Af},
        \nAb = \n{self.Ab},
        \nBf = \n{self.Bf},
        \nBb = \n{self.Bb},
        \nand WT = \n{self.WT}."""

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
