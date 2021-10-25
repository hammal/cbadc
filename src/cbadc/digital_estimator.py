"""Digital estimators.

This module provides alternative implementations for the control-bounded A/D
conterter's digital estimator.
"""
from typing import Iterator, Union
import cbadc
import cbadc.utilities
import scipy.linalg
import scipy.integrate
import numpy as np
from numpy.linalg import LinAlgError
import time
import logging
import os

logger = logging.getLogger(__name__)


def bruteForceCare(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    tau=1e-12,
    rtol=1e-200,
    atol=1e-300,
) -> np.ndarray:
    timelimit = 10 * 60
    start_time = time.time()
    try:
        V = np.array(care(A, B, Q, R), dtype=np.float128)
    except:
        V = np.eye(A.shape[0], dtype=np.float128)
    V_tmp = np.ones_like(V) * 1e300
    RInv = np.array(np.linalg.inv(R), dtype=np.float128)

    shrink = 1.0 - 1e-2

    while not np.allclose(V, V_tmp, rtol=rtol, atol=atol):
        if time.time() - start_time > timelimit:
            raise Exception("Brute Force CARE solver ran out of time")
        V_tmp = V[:, :]
        try:
            V = V + tau * (
                np.dot(A.transpose(), V)
                + np.dot(V.transpose(), A)
                + Q
                - np.dot(
                    V, np.dot(B, np.dot(RInv, np.dot(B.transpose(), V.transpose())))
                )
            )
            V = 0.5 * (V + V.transpose())
        except FloatingPointError:
            logger.warning("V_frw:\n{}\n".format(V))
            logger.warning("V_frw.dot(V_frw):\n{}".format(np.dot(V, V)))
            raise FloatingPointError
        # print(np.linalg.norm(V - V_tmp, ord="fro"))
        tau *= shrink
    return np.array(V, dtype=np.double)


def care(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    This function solves the forward and backward continuous Riccati equation.
    """
    A = np.array(A, dtype=np.double)
    B = np.array(B, dtype=np.double)
    Q = np.array(Q, dtype=np.double)
    R = np.array(R, dtype=np.double)

    V = np.zeros_like(A)

    try:
        V = scipy.linalg.solve_continuous_are(A, B, Q, R, balanced=True)
    except LinAlgError:
        logger.warning(
            """Cholesky Method Failed for computing the CARE of Vf.
            Starting brute force"""
        )
        V = bruteForceCare(A, B, Q, R)
    return np.array(V, dtype=np.double)


class DigitalEstimator(Iterator[np.ndarray]):
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
    ):
        # Check inputs
        if K1 < 1:
            raise BaseException("K1 must be a positive integer.")
        self.K1 = K1
        if K2 < 0:
            raise BaseException("K2 must be a non negative integer.")
        self.K2 = K2
        self.K3 = K1 + K2
        self._filter_lag = -1
        self.analog_system = analog_system

        if not np.allclose(self.analog_system.D, np.zeros_like(self.analog_system.D)):
            raise BaseException(
                """Can't compute filter coefficients for system with non-zero
                D matrix. Consider chaining for removing D"""
            )

        self.digital_control = digital_control
        if eta2 < 0:
            raise BaseException("eta2 must be non negative.")
        if Ts:
            self.Ts = Ts
        else:
            self.Ts = digital_control.T
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
        self._compute_filter_coefficients(analog_system, digital_control, eta2)
        self._allocate_memory_buffers()

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

    def warm_up(self):
        """Warm up filter by population control signals.

        Effectively removes the filter lag.
        """
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
        # Compute filter coefficients
        A = np.array(analog_system.A).transpose()
        B = np.array(analog_system.CT).transpose()
        Q = np.dot(np.array(analog_system.B), np.array(analog_system.B).transpose())
        R = eta2 * np.eye(analog_system.N_tilde)
        # Solve care
        Vf = care(A, B, Q, R)
        Vb = care(-A, B, Q, R)
        CCT: np.ndarray = np.dot(
            np.array(analog_system.CT).transpose(), np.array(analog_system.CT)
        )
        tempAf: np.ndarray = analog_system.A - np.dot(Vf, CCT) / eta2
        tempAb: np.ndarray = analog_system.A + np.dot(Vb, CCT) / eta2
        self.Af: np.ndarray = np.asarray(scipy.linalg.expm(tempAf * self.Ts))
        self.Ab: np.ndarray = np.asarray(scipy.linalg.expm(-tempAb * self.Ts))
        Gamma = np.array(analog_system.Gamma)
        # Solve IVPs
        self.Bf: np.ndarray = np.zeros((self.analog_system.N, self.analog_system.M))
        self.Bb: np.ndarray = np.zeros((self.analog_system.N, self.analog_system.M))

        atol = 1e-200
        rtol = 1e-10
        max_step = self.Ts / 1000.0
        if self.mid_point:
            for m in range(self.analog_system.M):

                def _derivative_forward(t, x):
                    return np.dot(tempAf, x) + np.dot(
                        Gamma, digital_control.impulse_response(m, t)
                    )

                solBf = scipy.integrate.solve_ivp(
                    _derivative_forward,
                    (0, self.Ts / 2.0),
                    np.zeros(self.analog_system.N),
                    atol=atol,
                    rtol=rtol,
                    max_step=max_step,
                    # method="Radau",
                ).y[:, -1]

                def _derivative_backward(t, x):
                    return np.dot(-tempAb, x) - np.dot(
                        Gamma, digital_control.impulse_response(m, self.Ts - t)
                    )

                solBb = scipy.integrate.solve_ivp(
                    _derivative_backward,
                    (0, self.Ts / 2.0),
                    np.zeros(self.analog_system.N),
                    atol=atol,
                    rtol=rtol,
                    max_step=max_step,
                    # method="Radau",
                ).y[:, -1]
                for n in range(self.analog_system.N):
                    self.Bf[n, m] = solBf[n]
                    self.Bb[n, m] = solBb[n]
            self.Bf = np.dot(
                np.eye(self.analog_system.N)
                + scipy.linalg.expm(tempAf * self.Ts / 2.0),
                self.Bf,
            )
            self.Bb = np.dot(
                np.eye(self.analog_system.N)
                + scipy.linalg.expm(tempAb * self.Ts / 2.0),
                self.Bb,
            )
        else:
            for m in range(self.analog_system.M):

                def _derivative_forward_2(t, x):
                    return np.dot(tempAf, x) + np.dot(
                        Gamma, digital_control.impulse_response(m, t)
                    )

                def impulse_start(t, x):
                    return t - self.digital_control._impulse_response[m].t0

                # impulse_start.terminate = True
                impulse_start.direction = 1.0

                solBf = scipy.integrate.solve_ivp(
                    _derivative_forward_2,
                    (0, self.Ts),
                    np.zeros(self.analog_system.N),
                    atol=atol,
                    rtol=rtol,
                    max_step=max_step,
                    # method="Radau",
                    events=(impulse_start,),
                ).y[:, -1]

                def _derivative_backward_2(t, x):
                    return np.dot(-tempAb, x) - np.dot(
                        Gamma, digital_control.impulse_response(m, self.Ts - t)
                    )

                def impulse_start(t, x):
                    return self.Ts - self.digital_control._impulse_response[m].t0 - t

                # impulse_start.terminate = True
                impulse_start.direction = -1.0

                solBb = scipy.integrate.solve_ivp(
                    _derivative_backward_2,
                    (0, self.Ts),
                    np.zeros(self.analog_system.N),
                    atol=atol,
                    rtol=rtol,
                    max_step=max_step,
                    # method="Radau",
                    events=(impulse_start,),
                ).y[:, -1]
                self.Bf[:, m] = solBf
                self.Bb[:, m] = solBb
        # self.WT = solve(Vf + Vb, analog_system.B).transpose()
        W, _, _, _ = np.linalg.lstsq(Vf + Vb, analog_system.B, rcond=None)
        self.WT = W.transpose()

    def _allocate_memory_buffers(self):
        # Allocate memory buffers
        self._control_signal = np.zeros((self.K3, self.analog_system.M), dtype=np.int8)
        self._estimate = np.zeros((self.K1, self.analog_system.L), dtype=np.double)
        self._control_signal_in_buffer = 0
        self._mean = np.zeros((self.K1 + 1, self.analog_system.N), dtype=np.double)

    def _compute_batch(self):
        temp_forward_mean = np.zeros(self.analog_system.N, dtype=np.double)
        # check if ready to compute buffer
        if self._control_signal_in_buffer < self.K3:
            raise BaseException("Control signal buffer not full")
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
            raise BaseException(
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
            raise BaseException("No iterator set.")
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
            (self.analog_system.L, self.analog_system.N_tilde, omega.size)
        )
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o]))
            G = G.reshape((self.analog_system.N_tilde, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, :, index] = np.abs(
                np.dot(GH, np.linalg.inv(GGH + self.eta2Matrix))
            )
        return result

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
        result = np.zeros((self.analog_system.L, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o])).reshape(
                (self.analog_system.N_tilde, self.analog_system.L)
            )
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, index] = np.abs(
                np.dot(GH, np.dot(np.linalg.inv(GGH + self.eta2Matrix), G))
            )
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


class ParallelEstimator(DigitalEstimator):
    """Parallelized batch estimator implementation.

    The parallel estimator estimates a filtered version
    :math:`\hat{\mathbf{u}}(t)` (shaped by :py:func:`signal_transfer_function`)
    of the input signal :math:`\mathbf{u}(t)` from a sequence of control
    signals :math:`\mathbf{s}[k]`.

    Specifically, the parallel estimator is a modified version of the default
    estimator :py:class:`cbadc.digital_estimator.DigitalEstimator` where the
    the filter matrices are diagonalized enabling a more efficient and
    possible parallelizable filter implementation. The estimate is computed as

    :math:`\hat{\mathbf{u}}(k T)[\ell] = \sum_{n=0}^N f_w[n] \cdot \overrightarrow{\mathbf{m}}[k][n] + b_w[n] \cdot \overleftarrow{\mathbf{m}}[k][n]`

    where

    :math:`\overrightarrow{\mathbf{m}}[k][n] = f_a[n] \cdot \overrightarrow{\mathbf{m}}[k-1][n] + \sum_{m=0}^{M-1} f_b[n, m] \cdot \mathbf{s}[k-1][m]`

    and

    :math:`\overleftarrow{\mathbf{m}}[k][n] = b_a \cdot \overrightarrow{\mathbf{m}}[k+1][n] + \sum_{m=0}^{M-1} b_b[n, m] \cdot \mathbf{s}[k][m]`.

    Furthermore, :math:`f_a, b_a \in \mathbb{R}^{N}`, :math:`f_b, b_b \in \mathbb{R}^{N \\times M}`,
    and :math:`f_w, b_w \in \mathbb{R}^{L \\times N}` are the precomputed filter coefficient formed
    from the filter coefficients as in :py:class:`cbadc.digital_estimator.DigitalEstimator`.

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
    ):
        # Check inputs
        if K1 < 1:
            raise BaseException("K1 must be a positive integer.")
        self.K1 = K1
        if K2 < 0:
            raise BaseException("K2 must be a non negative integer.")
        self.K2 = K2
        self.K3 = K1 + K2
        self._filter_lag = -1
        self.analog_system = analog_system
        self.digital_control = digital_control
        if eta2 < 0:
            raise BaseException("eta2 must be non negative.")
        if Ts:
            self.Ts = Ts
        else:
            self.Ts = digital_control.T
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
        self._compute_filter_coefficients(analog_system, digital_control, eta2)
        self._allocate_memory_buffers()

    def _compute_filter_coefficients(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
        eta2: float,
    ):
        # Compute filter coefficients from base class
        DigitalEstimator._compute_filter_coefficients(
            self, analog_system, digital_control, eta2
        )
        # Parallelize
        temp, Q_f = np.linalg.eig(self.Af)
        self.forward_a = np.array(temp, dtype=np.complex128)
        Q_f_inv = np.linalg.pinv(Q_f, rcond=1e-20)
        temp, Q_b = np.linalg.eig(self.Ab)
        self.backward_a = np.array(temp, dtype=np.complex128)
        Q_b_inv = np.linalg.pinv(Q_b, rcond=1e-20)

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
        mean: np.complex128 = np.complex128(0)
        # check if ready to compute buffer
        if self._control_signal_in_buffer < self.K3:
            raise BaseException("Control signal buffer not full")

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
            raise BaseException(
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
            raise BaseException("No iterator set.")

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
            logger.warning("StopIteration received by estimator.")
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


class IIRFilter(DigitalEstimator):
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
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
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
        K2: int,
        stop_after_number_of_iterations: int = (1 << 63),
        Ts: float = None,
        mid_point: bool = False,
        downsample: int = 1,
    ):
        """Initializes filter coefficients"""
        if K2 < 0:
            raise BaseException("K2 must be non negative integer.")
        self.K2 = K2
        self._filter_lag = self.K2 - 2
        self.analog_system = analog_system
        self.digital_control = digital_control
        if eta2 < 0:
            raise BaseException("eta2 must be non negative.")
        self.eta2 = eta2
        self.control_signal = None
        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        if Ts:
            self.Ts = Ts
        else:
            self.Ts = digital_control.T

        self.downsample = int(downsample)

        self.mid_point = mid_point

        # For transfer functions
        self.eta2Matrix = np.eye(self.analog_system.CT.shape[0]) * self.eta2

        # Compute filter coefficients
        DigitalEstimator._compute_filter_coefficients(
            self, analog_system, digital_control, eta2
        )

        # Initialize filter
        self.h = np.zeros(
            (self.analog_system.L, self.K2, self.analog_system.M), dtype=np.double
        )
        # Compute lookback
        temp2 = np.copy(self.Bb)
        for k2 in range(self.K2):
            self.h[:, k2, :] = np.dot(self.WT, temp2)
            temp2 = np.dot(self.Ab, temp2)
        self._control_signal_valued = np.zeros(
            (self.K2, self.analog_system.M), dtype=np.int8
        )
        self._mean = np.zeros(self.analog_system.N, dtype=np.double)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        # Check if control signal iterator is set.
        if self.control_signal is None:
            raise BaseException("No iterator set.")

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

        self._control_signal_valued[-1, :] = np.asarray(2 * temp - 1, dtype=np.int8)

        # self._control_signal_valued.shape -> (K2, M)
        # self.h.shape -> (L, K2, M)
        result = -np.dot(self.WT, self._mean)
        self._mean = np.dot(self.Af, self._mean) + np.dot(
            self.Bf, self._control_signal_valued[0, :]
        )
        if ((self._iteration - 1) % self.downsample) == 0:
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


class FIRFilter(DigitalEstimator):
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
        fixed_point: Union[cbadc.utilities.FixedPoint, None] = None,
        # fixed_point=None,
    ):
        """Initializes filter coefficients"""
        if K1 < 0:
            raise BaseException("K1 must be non negative integer.")
        self.K1 = K1
        if K2 < 1:
            raise BaseException("K2 must be a positive integer.")
        self.K2 = K2
        self.K3 = K1 + K2
        self._filter_lag = self.K2 - 2
        self.analog_system = analog_system
        self.digital_control = digital_control
        if eta2 < 0.0:
            raise BaseException("eta2 must be non negative.")
        self.eta2 = eta2
        self.control_signal = None
        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        if Ts:
            self.Ts = Ts
        else:
            self.Ts = digital_control.T
        if mid_point:
            raise NotImplementedError("Planned for v.0.1.0")
        self.mid_point = mid_point
        self.downsample = int(downsample)
        self._temp_controls = np.zeros(
            (self.downsample, self.analog_system.M), dtype=np.int8
        )

        if offset is not None:
            self.offset = np.array(offset, dtype=np.float64)
            if self.offset.size != self.analog_system.L:
                raise BaseException("offset is not of size L")
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
        self._control_signal_valued = np.zeros(
            (self.K3, self.analog_system.M), dtype=np.int8
        )

    def __iter__(self):
        return self

    def _compute_filter_coefficients(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
        eta2: float,
    ):
        atol = 1e-30
        rtol = 2.3e-14
        steps = 1000
        max_step = self.Ts / steps

        # Compute filter coefficients
        A = np.array(analog_system.A).transpose()
        B = np.array(analog_system.CT).transpose()
        Q = np.dot(np.array(analog_system.B), np.array(analog_system.B).transpose())
        R = eta2 * np.eye(analog_system.N_tilde)
        # Solve care
        Vf = care(A, B, Q, R)
        # tau = 1e-6
        # rtol = 1e-150
        # atol = 1e-350
        # Vf = bruteForceCare(A, B, Q, R, tau=tau, rtol=rtol, atol=atol)
        Vb = care(-A, B, Q, R)
        # Vb = bruteForceCare(-A, B, Q, R, tau=tau, rtol=rtol, atol=atol)

        CCT: np.ndarray = np.dot(
            np.array(analog_system.CT).transpose(), np.array(analog_system.CT)
        )
        tempAf: np.ndarray = analog_system.A - np.dot(Vf, CCT) / eta2
        tempAb: np.ndarray = analog_system.A + np.dot(Vb, CCT) / eta2
        self.Af: np.ndarray = np.asarray(scipy.linalg.expm(tempAf * self.Ts))
        self.Ab: np.ndarray = np.asarray(scipy.linalg.expm(-tempAb * self.Ts))
        # Solve IVPs
        self.Bf: np.ndarray = np.zeros((self.analog_system.N, self.analog_system.M))
        self.Bb: np.ndarray = np.zeros((self.analog_system.N, self.analog_system.M))

        self._Bf_2T = np.zeros_like(self.Bf)

        if self.mid_point:
            for m in range(self.analog_system.M):

                def _derivative_forward(t, x):
                    return np.dot(tempAf, x) + np.dot(
                        self.analog_system.Gamma, digital_control.impulse_response(m, t)
                    )

                solBf = scipy.integrate.solve_ivp(
                    _derivative_forward,
                    (0, self.Ts / 2.0),
                    np.zeros(self.analog_system.N),
                    atol=atol,
                    rtol=rtol,
                    max_step=max_step,
                    t_eval=np.linspace(0, self.Ts, steps),
                    # method="Radau",
                ).y[:, -1]

                def _derivative_backward(t, x):
                    return -np.dot(tempAb, x) - np.dot(
                        self.analog_system.Gamma, digital_control.impulse_response(m, t)
                    )

                solBb = scipy.integrate.solve_ivp(
                    _derivative_backward,
                    (0, self.Ts / 2.0),
                    np.zeros(self.analog_system.N),
                    atol=atol,
                    rtol=rtol,
                    max_step=max_step,
                    t_eval=np.linspace(0, self.Ts, steps),
                    # method="Radau",
                ).y[:, -1]
                for n in range(self.analog_system.N):
                    self.Bf[n, m] = solBf[n]
                    self.Bb[n, m] = solBb[n]
            self.Bf = np.dot(
                np.eye(self.analog_system.N)
                + scipy.linalg.expm(tempAf * self.Ts / 2.0),
                self.Bf,
            )
            self.Bb = np.dot(
                np.eye(self.analog_system.N)
                + scipy.linalg.expm(tempAb * self.Ts / 2.0),
                self.Bb,
            )
        # elif isinstance(
        #     self.digital_control, cbadc.digital_control.SwitchedCapacitorControl
        # ):
        #     for m in range(self.analog_system.M):
        #         N = self.analog_system.N
        #         tempStateSpaceMatrix = np.zeros((N + 1, N + 1))

        #         # Forward Coefficients
        #         tempStateSpaceMatrix[:N, :N] = tempAf
        #         tempStateSpaceMatrix[N, N] = self.digital_control.A[m, m]
        #         tempStateSpaceMatrix[:N, N] = self.analog_system.Gamma[:, m]

        #         def _derivative_forward(t, x):
        #             return (
        #                 np.dot(tempStateSpaceMatrix, x).flatten()
        #                 # + np.random.randn(N + 1) * 1e-3
        #             )

        #         # print(
        #         #     f"m={m}, T1={self.digital_control.T1[m]}, T2={self.digital_control.T2[m]}"
        #         # )

        #         initial_state = np.zeros(N + 1)
        #         initial_state[N] = self.digital_control.VCap
        #         # First phase (discarging capacitor)
        #         if self.digital_control.T2[m] > self.digital_control.T:
        #             T2LT = True
        #             t_eval1 = np.array(
        #                 [
        #                     self.digital_control.T - self.digital_control.T1[m],
        #                     self.digital_control.T2[m] - self.digital_control.T1[m],
        #                 ]
        #             )
        #             tspan1 = (0, t_eval1[-1])
        #             tspace1 = (
        #                 min(
        #                     (
        #                         (
        #                             self.digital_control.T2[m]
        #                             - self.digital_control.T1[m]
        #                         ),
        #                         (self.digital_control.T - self.digital_control.T1[m]),
        #                     )
        #                 )
        #                 / steps
        #             )
        #             t_eval2 = np.array(
        #                 [0, 2 * self.digital_control.T - self.digital_control.T2[m]]
        #             )
        #             tspan2 = (0, t_eval2[-1])
        #             tspace2 = (
        #                 2 * self.digital_control.T - self.digital_control.T2[m]
        #             ) / steps
        #         else:
        #             T2LT = False
        #             t_eval1 = np.array(
        #                 [0, self.digital_control.T2[m] - self.digital_control.T1[m]]
        #             )
        #             tspan1 = (0, t_eval1[-1])
        #             tspace1 = (
        #                 self.digital_control.T2[m] - self.digital_control.T1[m]
        #             ) / steps
        #             t_eval2 = np.array(
        #                 [
        #                     self.digital_control.T - self.digital_control.T2[m],
        #                     2 * self.digital_control.T - self.digital_control.T2[m],
        #                 ]
        #             )
        #             tspan2 = (0, t_eval2[-1])
        #             if self.digital_control.T == self.digital_control.T2[m]:
        #                 tspace2 = (
        #                     2 * self.digital_control.T - self.digital_control.T2[m]
        #                 ) / steps
        #             else:
        #                 tspace2 = (
        #                     self.digital_control.T - self.digital_control.T2[m]
        #                 ) / steps

        #         # print(f"T2 > T = {T2LT}")

        #         odeDOP853 = scipy.integrate.DOP853(
        #             _derivative_forward,
        #             0,
        #             initial_state,
        #             t_bound=tspan1[1],
        #             first_step=tspace1,
        #             max_step=tspace1,
        #             rtol=rtol,
        #             atol=atol,
        #             vectorized=False,
        #         )

        #         sol1 = []
        #         for t in t_eval1:
        #             while odeDOP853.t < t:
        #                 odeDOP853.step()
        #             sol1.append(odeDOP853.y)

        #         empty_cap_at_T2 = sol1[-1][N]
        #         initial_state = sol1[-1][:]
        #         initial_state[N] = 0

        #         odeDOP853 = scipy.integrate.DOP853(
        #             _derivative_forward,
        #             0,
        #             initial_state,
        #             t_bound=tspan2[1],
        #             first_step=tspace2,
        #             max_step=tspace2,
        #             rtol=rtol,
        #             atol=atol,
        #             vectorized=False,
        #         )

        #         sol2 = []
        #         for t in t_eval2:
        #             while odeDOP853.t < t:
        #                 odeDOP853.step()
        #             sol2.append(odeDOP853.y)

        #         # T2 > T
        #         if T2LT:
        #             self.Bf[:, m] = sol1[0][:N]
        #             self._Bf_2T[:, m] = sol2[1][:N]
        #         else:
        #             self.Bf[:, m] = sol2[0][:N]
        #             self._Bf_2T[:, m] = sol2[1][:N]

        #         # Backward Coefficients
        #         tempStateSpaceMatrix[:N, :N] = -tempAb
        #         tempStateSpaceMatrix[N, N] = -self.digital_control.A[m, m]
        #         tempStateSpaceMatrix[:N, N] = -self.analog_system.Gamma[:, m]

        #         def _derivative_backward(t, x):
        #             return (
        #                 np.dot(tempStateSpaceMatrix, x).flatten()
        #                 # + np.random.randn(N + 1) * 1e-3
        #             )

        #         # First Phase
        #         initial_state = np.zeros(N + 1)
        #         initial_state[N] = empty_cap_at_T2
        #         # print(
        #         #     f"initial_state = {initial_state}, empty_cap_at_T2 = {empty_cap_at_T2}"
        #         # )
        #         t_eval1 = np.array(
        #             [0, self.digital_control.T2[m] - self.digital_control.T1[m]]
        #         )
        #         max_step = t_eval1[1] / steps

        #         odeDOP853 = scipy.integrate.DOP853(
        #             _derivative_backward,
        #             0,
        #             initial_state,
        #             t_bound=t_eval1[1],
        #             first_step=max_step,
        #             max_step=max_step,
        #             rtol=rtol,
        #             atol=atol,
        #             vectorized=False,
        #         )

        #         sol1 = []
        #         for t in t_eval1:
        #             while odeDOP853.t < t:
        #                 odeDOP853.step()
        #             sol1.append(odeDOP853.y)

        #         if self.digital_control.T1[m] > 0:
        #             initial_state = sol1[1][:]
        #             initial_state[N] = 0.0
        #             t_eval2 = np.array([0, self.digital_control.T1[m]])
        #             max_step = t_eval2[1] / steps

        #             odeDOP853 = scipy.integrate.DOP853(
        #                 _derivative_backward,
        #                 0,
        #                 initial_state,
        #                 t_bound=t_eval2[1],
        #                 first_step=max_step,
        #                 max_step=max_step,
        #                 rtol=rtol,
        #                 atol=atol,
        #                 vectorized=False,
        #             )

        #             sol2 = []
        #             for t in t_eval2:
        #                 while odeDOP853.t < t:
        #                     odeDOP853.step()
        #                 sol2.append(odeDOP853.y)
        #             self.Bb[:, m] = sol2[1][:N]
        #         else:
        #             self.Bb[:, m] = sol1[1][:N]
        else:
            for m in range(self.analog_system.M):

                def _derivative_forward_2(t, x):
                    return np.dot(tempAf, x) + np.dot(
                        self.analog_system.Gamma, digital_control.impulse_response(m, t)
                    )

                def impulse_start(t, x):
                    return t - self.digital_control._impulse_response[m].t0

                impulse_start.direction = 1.0

                solBf = scipy.integrate.solve_ivp(
                    _derivative_forward_2,
                    (0, self.Ts),
                    np.zeros(self.analog_system.N),
                    atol=atol,
                    rtol=rtol,
                    method="Radau",
                    vectorized=False,
                    jac=tempAf,
                    max_step=max_step,
                    events=(impulse_start,),
                ).y[:, -1]

                def _derivative_backward_2(t, x):
                    return np.dot(-tempAb, x) - np.dot(
                        self.analog_system.Gamma,
                        digital_control.impulse_response(m, self.Ts - t),
                    )

                def impulse_start(t, x):
                    return self.Ts - self.digital_control._impulse_response[m].t0 - t

                # impulse_start.terminate = True
                impulse_start.direction = -1.0

                solBb = scipy.integrate.solve_ivp(
                    _derivative_backward_2,
                    (0, self.Ts),
                    np.zeros(self.analog_system.N),
                    atol=atol,
                    rtol=rtol,
                    method="Radau",
                    vectorized=False,
                    jac=-tempAb,
                    max_step=max_step,
                    events=(impulse_start,),
                ).y[:, -1]

                self.Bf[:, m] = solBf
                self.Bb[:, m] = solBb
        W, _, _, _ = np.linalg.lstsq(
            np.array(Vf + Vb, dtype=np.double), analog_system.B, rcond=-1
        )
        self.WT = W.transpose()

    def __next__(self) -> np.ndarray:
        # Check if control signal iterator is set.
        if self.control_signal is None:
            raise BaseException("No iterator set.")

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

        As the filter computes the estimate as
        -----------------
        |   K1  |   K2  |
        -----------------
                ^
                |
                u_hat[k]


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
                    for k in range(self.K3 - 1):
                        f.write(f"{self.h[l,k,m]:.17E},")
                    f.write(f"{self.h[l,-1,m]:.17E}" + "}")
                    if m < (self.analog_system.M - 1):
                        f.write(",")
                f.write("}")
                if l < (self.analog_system.L - 1):
                    f.write(",")
            f.write("}" + os.linesep)

    def number_of_filter_coefficients(self) -> int:
        """Number of non-zero filter coefficients

        Returns
        -------
        `int`
            total number of non-zero filter coefficients
        """
        return np.sum(self.h > 0)


class NUVEstimator:
    """The NUV batch estimator

    The NUV estimator iteratively estimates estimates a filtered version
    :math:`\hat{\mathbf{u}}(t)` of the input signal :math:`\mathbf{u}(t)` from
    a sequence of control signals :math:`\mathbf{s}[k]`.

    In comparision to the linear complexity estimators

    - :py:class:`cbadc.digital_estimator.DigitalEstimator`
    - :py:class:`cbadc.digital_estimator.ParallelEstimator`
    - :py:class:`cbadc.digital_estimator.IIRFilter`
    - :py:class:`cbadc.digital_estimator.FIRFilter`

    the NUV estimator enforces the bounded outputs in an iterative
    scheme as described in

    * `R. Keusch and H.-A. Loeliger, A Binarizing NUV Prior and its Use for M-Level Control and Digital-to-Analog Conversion. <https://arxiv.org/pdf/2105.02599.pdf>`_

    Parameters
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system (necessary to compute the estimators filter
        coefficients).
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        a digital control (necessary to determine the corresponding DAC
        waveform).
    covU: `float`
        a prior covariance matrix for the sought estimate.
    bound_y: `float`
        the bound enforced on the outputs :math:`\mathbf{y}(t)` of the
        system.
    gamma: `float`
        a scale factor.
    K1 : `int`
        batch size.
    K2 : `int`, `optional`
        lookahead size, defaults to 0.
    iterations_per_batch: `int`, `optional`
        number of iteration steps per batch, defaults to 100.
    Ts: `float`, `optional`
        the sampling time, defaults to the time period of the digital control.
    oversample: `int`, `optional`
        add observations per control signal, defaults to 1, i.e., no oversampling.

    Attributes
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        analog system as in :py:class:`cbadc.analog_system.AnalogSystem` or
        from derived class.
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        digital control as in :py:class:`cbadc.digital_control.DigitalControl`
        or from derived class.
    K1: `int`
        number of samples per estimate batch.
    K2: `int`
        number of lookahead samples per computed batch.
    covU: `float`
        the prior covariance matrix (or scalar in the single input case)
        of the estimate :math:`\hat{\mathbf{u}}(t)`.
    bound_y: `float`
        the bound enforced on the outputs :math:`\mathbf{y}(t)` of the
        system.
    Ts: `float`
        the sample rate of the estimates
    iterations_per_batch: `int`
        number of iteration steps per batch.
    gamma: `float`
        a scale factor.
    """

    def __init__(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
        covU: float,
        bound_y: float,
        gamma: float,
        K1: int,
        K2: int = 0,
        iterations_per_batch: int = 100,
        Ts: float = None,
        oversample: int = 1,
    ):
        # Check inputs
        if K1 < 1:
            raise BaseException("K1 must be a positive integer.")
        self.K1 = K1
        if K2 < 0:
            raise BaseException("K2 must be a non negative integer.")
        self.K2 = K2
        self.K3 = K1 + K2
        self.analog_system = analog_system
        self.covU = covU
        self.bound_y = bound_y
        if not np.allclose(self.analog_system.D, np.zeros_like(self.analog_system.D)):
            raise BaseException(
                """Can't compute filter coefficients for system with non-zero
                D matrix. Consider chaining for removing D"""
            )

        self.digital_control = digital_control
        if Ts:
            self.Ts = Ts / oversample
        else:
            self.Ts = digital_control.T / oversample
        self.control_signal = None
        self._stop_iteration = False
        self._estimate_pointer = self.K1

        self.iterations_per_batch = iterations_per_batch
        self.gamma = gamma
        # Initialize filters
        self._compute_filter_coefficients(analog_system, digital_control)
        self._allocate_memory_buffers()

        # Set initial Covariances
        self._forward_CoVariance[0, :, :] = 1e12 * np.eye(self.analog_system.N)
        self._oversample = oversample

    def _allocate_memory_buffers(self):
        # Allocate memory buffers
        self._control_signal = np.zeros((self.K3, self.analog_system.M), dtype=np.int8)
        self._estimate = np.zeros((self.K1, self.analog_system.L), dtype=np.double)
        self._control_signal_in_buffer = 0
        self._forward_mean = np.zeros((self.K3, self.analog_system.N), dtype=np.double)
        self._forward_CoVariance = np.zeros(
            (self.K3, self.analog_system.N, self.analog_system.N), dtype=np.double
        )
        self._G = np.zeros(
            (self.K3, self.analog_system.N_tilde, self.analog_system.N_tilde),
            dtype=np.double,
        )
        self._F = np.zeros(
            (self.K3, self.analog_system.N, self.analog_system.N), dtype=np.double
        )

        self._y_mean = np.zeros((self.K3, self.analog_system.N_tilde))
        self._y_CoVariance = np.zeros(
            (self.K3, self.analog_system.N_tilde, self.analog_system.N_tilde)
        )
        # self._W_tilde = np.zeros(
        #     (
        #         self.K3,
        #         self.analog_system.N,
        #         self.analog_system.N
        #     ),
        #     dtype=np.double
        # )
        self._xi_tilde = np.zeros((self.K3 + 1, self.analog_system.N), dtype=np.double)
        self._sigma_squared_1 = np.ones(
            (
                self.K3,
                self.analog_system.N_tilde,
            ),
            dtype=np.double,
        )
        self._sigma_squared_2 = np.ones(
            (
                self.K3,
                self.analog_system.N_tilde,
            ),
            dtype=np.double,
        )
        self._posterior_observation_mean = np.zeros(
            (self.K3, self.analog_system.N_tilde), dtype=np.double
        )

    def set_iterator(self, control_signal_sequence: Iterator[np.ndarray]):
        """Set iterator of control signals

        Parameters
        -----------
        control_signal_sequence : iterator
            a iterator which outputs a sequence of control signals.
        """
        self.control_signal = control_signal_sequence

    def __call__(self, control_signal_sequence: Iterator[np.ndarray]):
        return self.set_iterator(control_signal_sequence)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        # Check if control signal iterator is set.
        if self.control_signal is None:
            raise BaseException("No iterator set.")

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
            for _ in range(self._oversample):
                full = self._input(control_signal_sample)

        # Compute new batch of K1 estimates
        self._reset_between_batch()
        self._compute_batch()
        # adjust pointer to indicate that estimate buffer
        # is non empty
        self._estimate_pointer -= self.K1

        # recursively call itself to return new estimate
        return self.__next__()

    def _input(self, s: np.ndarray) -> bool:
        if self._control_signal_in_buffer == (self.K3):
            raise BaseException(
                """Input buffer full. You must compute batch before adding
                more control signals"""
            )
        for _ in range(self.analog_system.M):
            self._control_signal[self._control_signal_in_buffer, :] = np.asarray(
                2 * s - 1, dtype=np.int8
            )
        self._control_signal_in_buffer += 1
        return self._control_signal_in_buffer > (self.K3 - 1)

    def _compute_filter_coefficients(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
    ):
        # Compute filter coefficients
        self.Af: np.ndarray = np.asarray(scipy.linalg.expm(analog_system.A * self.Ts))
        Gamma = np.array(analog_system.Gamma)
        # Solve IVPs
        self.Bf: np.ndarray = np.zeros((self.analog_system.N, self.analog_system.M))
        atol = 1e-200
        rtol = 1e-12
        max_step = self.Ts / 1000.0

        for m in range(self.analog_system.M):

            def _derivative_forward_2(t, x):
                return np.dot(analog_system.A, x) + np.dot(
                    Gamma, digital_control.impulse_response(m, t)
                )

            solBf = scipy.integrate.solve_ivp(
                _derivative_forward_2,
                (0, self.Ts),
                np.zeros(self.analog_system.N),
                atol=atol,
                rtol=rtol,
                max_step=max_step,
                method="Radau",
            ).y[:, -1]
            self.Bf[:, m] = solBf

        BBT = np.dot(analog_system.B, np.dot(self.covU, analog_system.B.transpose()))

        def _derivative_input(t, x):
            t_minus_tau = self.Ts - t
            A_t_minus_tau = analog_system.A * t_minus_tau
            return np.dot(
                scipy.linalg.expm(A_t_minus_tau),
                np.dot(BBT, scipy.linalg.expm(A_t_minus_tau.transpose())),
            ).flatten()

        self.Vu = np.dot(
            self.covU,
            scipy.integrate.solve_ivp(
                _derivative_input,
                (0, self.Ts),
                np.zeros(self.analog_system.N ** 2),
                atol=atol,
                rtol=rtol,
                max_step=max_step,
                method="Radau",
            )
            .y[:, -1]
            .reshape((self.analog_system.N, self.analog_system.N)),
        )

    def _compute_batch(self):
        # check if ready to compute buffer
        if self._control_signal_in_buffer < self.K3:
            raise BaseException("Control signal buffer not full")

        self._update_observation_statistics()
        self._MBF()
        for _ in range(self.iterations_per_batch):
            self._update_observation_bound_variances()
            self._update_observation_statistics()
            self._MBF()

        self._input_estimation()

        # Save forward mean and covariance for next batch
        self._forward_mean[0, :] = self._forward_mean[self.K1 - 1, :]
        # Compute posterior mean
        # self._forward_mean[0, :] = self._forward_mean[self.K1 - 1] - np.dot(
        #     self._forward_CoVariance[self.K1 - 1, :, :],
        #     self._xi_tilde[self.K1 - 1, :]
        # )
        self._forward_CoVariance[0, :, :] = self._forward_CoVariance[self.K1 - 1, :, :]

        # Compute posterior covariance
        # self._forward_CoVariance[0, :, :] = self._forward_CoVariance[self.K1 - 1, :, :] - np.dot(
        #     self._forward_CoVariance[self.K1 - 1, :, :],
        #     np.dot(
        #         self._W_tilde[self.K1 - 1, :, :],
        #         self._forward_CoVariance[self.K1 - 1, :, :]
        #     )
        # )

        self._sigma_squared_1[0, :] = self._sigma_squared_1[self.K1 - 1, :]
        self._sigma_squared_2[0, :] = self._sigma_squared_2[self.K1 - 1, :]

        # rotate buffer to make place for new control signals
        self._control_signal = np.roll(self._control_signal, -self.K1, axis=0)
        self._control_signal_in_buffer -= self.K1

    def _MBF(self):
        # Forward pass
        temp_K3 = self.K3 - 1
        for k in range(self.K3):
            self._G[k, :, :] = np.linalg.inv(
                self._y_CoVariance[k, :, :]
                + np.dot(
                    self.analog_system.CT,
                    np.dot(
                        self._forward_CoVariance[k, :, :],
                        self.analog_system.CT.transpose(),
                    ),
                )
            )
            self._F[k, :, :] = np.eye(self.analog_system.N) - np.dot(
                self._forward_CoVariance[k, :, :],
                np.dot(
                    self.analog_system.CT.transpose(),
                    np.dot(self._G[k, :, :], self.analog_system.CT),
                ),
            )

            if k < temp_K3:
                self._forward_mean[k + 1, :] = (
                    np.dot(
                        self.Af,
                        np.dot(self._F[k, :, :], self._forward_mean[k, :])
                        + np.dot(
                            self._forward_CoVariance[k, :, :],
                            np.dot(
                                self.analog_system.CT.transpose(),
                                np.dot(self._G[k, :, :], self._y_mean[k, :]),
                            ),
                        ),
                    )
                    + np.dot(self.Bf, self._control_signal[k, :])
                )

                self._forward_CoVariance[k + 1, :, :] = (
                    np.dot(
                        self.Af,
                        np.dot(
                            np.dot(self._F[k, :, :], self._forward_CoVariance[k, :, :]),
                            self.Af.transpose(),
                        ),
                    )
                    + self.Vu
                )

        # Backward pass
        for k in range(self.K3 - 1, -1, -1):

            if k < temp_K3:
                xi_tilde_z = np.dot(self.Af.transpose(), self._xi_tilde[k + 1, :])
                # W_tilde_z = np.dot(
                #     self.Af.transpose(),
                #     np.dot(
                #         self._W_tilde[k+1, :, :],
                #         self.Af
                #     )
                # )
            else:
                # Initialize with zero vector.
                xi_tilde_z = np.zeros(self.analog_system.N)
                # W_tilde_z = np.zeros(
                #     (self.analog_system.N, self.analog_system.N))

            # self._W_tilde[k, :, :] = np.dot(
            #     self._F[k, :, :],
            #     np.dot(
            #         W_tilde_z,
            #         self._F[k, :, :]
            #     )
            # ) + np.dot(
            #     self.analog_system.CT.transpose(),
            #     np.dot(
            #         self._G[k, :, :],
            #         self.analog_system.CT
            #     )
            # )

            temp = (
                np.dot(self.analog_system.CT.transpose(), self._forward_mean[k, :])
                - self._y_mean[k, :]
            )

            self._xi_tilde[k, :] = np.dot(
                self._F[k, :, :].transpose(), xi_tilde_z
            ) + np.dot(
                self.analog_system.CT.transpose(), np.dot(self._G[k, :, :], temp)
            )
            # if (k < 5):
            #     print(k,  self._xi_tilde[k, :])

    def _input_estimation(self):
        for k in range(self.K1):
            self._estimate[k, :] = -np.dot(
                self.covU,
                np.dot(self.analog_system.B.transpose(), self._xi_tilde[k, :]),
            )

    def _update_observation_bound_variances(self):
        for k in range(self.K3):
            self._posterior_observation_mean[k, :] = np.dot(
                self.analog_system.CT,
                self._forward_mean[k]
                - np.dot(self._forward_CoVariance[k, :, :], self._xi_tilde[k, :]),
            )
            # posterior_CoVariance = np.dot(
            #     self.analog_system.CT,
            #     np.dot(
            #         self._forward_CoVariance[k, :, :] - np.dot(
            #             self._forward_CoVariance[k, :, :],
            #             np.dot(
            #                 self._W_tilde[k, :, :],
            #                 self._forward_CoVariance[k, :, :]
            #             )
            #         ),
            #         self.analog_system.CT.transpose()
            #     )
            # )
            min_sigma_squared = 1e-100
            for ell in range(self.analog_system.N_tilde):
                self._sigma_squared_1[k, ell] = max(
                    np.abs(self._posterior_observation_mean[k, ell] - self.bound_y)
                    / self.gamma,
                    min_sigma_squared,
                )
                self._sigma_squared_2[k, ell] = max(
                    np.abs(self._posterior_observation_mean[k, ell] + self.bound_y)
                    / self.gamma,
                    min_sigma_squared,
                )

    def _update_observation_statistics(self):
        for k in range(self.K3):
            W1_back = np.diag(1.0 / self._sigma_squared_1[k, :])
            W2_back = np.diag(1.0 / self._sigma_squared_2[k, :])

            W_back = W1_back + W2_back
            Wm_back = self.bound_y * np.sum(W1_back, axis=-1) - self.bound_y * np.sum(
                W2_back, axis=-1
            )

            self._y_CoVariance[k, :, :] = np.linalg.inv(W_back)

            self._y_mean[k, :] = np.dot(self._y_CoVariance[k, :, :], Wm_back)

    def _reset_between_batch(self):
        self._sigma_squared_1[1:, :] = np.ones(
            (self.K3 - 1, self.analog_system.N_tilde), dtype=np.double
        )
        self._sigma_squared_2[1:, :] = np.ones(
            (self.K3 - 1, self.analog_system.N_tilde), dtype=np.double
        )

    def __str__(self):
        return f"""NUVEstimator is parameterized as
        \nobservation bound = {self.bound_y:.2f},
        \ncovU = {self.covU},
        \ngamma = {self.gamma},
        \nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},
        \nand\nnumber of iterations per batch = {self.iterations_per_batch}
        \nResulting in the filter coefficients\nAf = \n{self.Af},
        \nBf = \n{self.Bf},
        \nand Vu = \n{self.Vu}.
        """
