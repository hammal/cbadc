# cython: language_level=3
"""Digital estimators.

This module provides alternative implementations for the control-bounded A/D conterter's
digital estimator.
"""
from cbadc.digital_control cimport DigitalControl
from cbadc.analog_system cimport AnalogSystem
from scipy.linalg import expm, solve, solve_continuous_are
from scipy.integrate import solve_ivp
import numpy.linalg as linalg
cimport numpy as np
import numpy as np
import math
from numpy.linalg import LinAlgError
import time
import logging

logger = logging.getLogger(__name__)

cdef double [:, :] bruteForceCare(double [:, :] A, double [:, :] B, double [:, :] Q, double [:, :] R):
    timelimit = 10 * 60
    start_time = time.time()
    V = np.eye(A.shape[0])
    V_tmp = np.zeros_like(V)
    tau = 1e-5
    RInv = np.linalg.inv(R)

    while not np.allclose(V, V_tmp, rtol=1e-5, atol=1e-8):
        if time.time() - start_time > timelimit:
            raise Exception("Brute Force CARE solver ran out of time")
        V_tmp = V
        try:
            V = V + tau * (
                np.dot(A, V)
                + np.transpose(np.dot(A, V))
                + Q
                - np.dot(V, np.dot(B, np.dot(RInv, np.dot(B.transpose(), V))))
            )
        except FloatingPointError:
            print("V_frw:\n{}\n".format(V))
            print("V_frw.dot(V_frw):\n{}".format(np.dot(V, V)))
            raise FloatingPointError
    return V


cdef double[:, :] care(double [:, :] A, double [:, :] B, double [:, :] Q, double [:, :] R):
    """
    This function solves the forward and backward continuous Riccati equation.
    """
    A = np.array(A, dtype=np.double)
    A_minus = np.array(A, dtype=np.double)
    B = np.array(B, dtype=np.double)
    Q = np.array(Q, dtype=np.double)
    R = np.array(R, dtype=np.double)

    cdef double [:,:] V = np.zeros_like(A)

    try:
        V = solve_continuous_are(A, B, Q, R)
    except LinAlgError:
        logger.warning(
            """Cholesky Method Failed for computing the CARE of Vf.
            Starting brute force"""
        )
        V = bruteForceCare(A, B, Q, R)
    return V

class DigitalEstimator:
    """Batch estimator implementation.
    
    The digital estimator estimates a filtered version :math:`\hat{\mathbf{u}}(t)` (shaped by 
    :py:func:`signal_transfer_function`) of the 
    input signal :math:`\mathbf{u}(t)` from a sequence of control signals :math:`\mathbf{s}[k]`.

    Specifically, the estimates are computed as

    :math:`\overrightarrow{\mathbf{m}}[k] = \mathbf{A}_f \overrightarrow{\mathbf{m}}[k-1] + \mathbf{B}_f \mathbf{s}[k-1]`,
    
    :math:`\overleftarrow{\mathbf{m}}[k] = \mathbf{A}_b \overrightarrow{\mathbf{m}}[k+1] + \mathbf{B}_b \mathbf{s}[k]`,

    and

    :math:`\hat{\mathbf{u}}(k T) = \mathbf{W}^\mathsf{T}\\left(\overleftarrow{\mathbf{m}}[k] -  \overrightarrow{\mathbf{m}}[k]\\right)`

    where :math:`\mathbf{A}_f, \mathbf{A}_b \in \mathbb{R}^{N \\times N}`, :math:`\mathbf{B}_f, \mathbf{B}_b \in \mathbb{R}^{N \\times M}`,
    and :math:`\mathbf{W}^\mathsf{T} \in \mathbb{R}^{L \\times N}` are the precomputed filter coefficient based on the
    choise of :py:class:`cbadc.analog_system.AnalogSystem` and :py:class:`cbadc.digital_control.DigitalControl`.

    Parameters
    ----------

    control_signal_sequence : iterator
        a generator which outputs a sequence of control signals.
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
        determine a max number of iterations by the generator, defaults to :math:`2^{63}`.
    Ts: `float`, `optional`
        the sampling time, defaults to the time period of the digital control.
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
        a generator suppling control signals as :py:class:`cbadc.digital_control.DigitalControl`.
    number_of_iterations : `int`
        number of iterations until generator raises :py:class:`StopIteration`.
    downsample: `int`, `optional`
        The downsampling factor compared to the rate of the control signal.
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

    def __init__(self, 
        control_signal_sequence, 
        analog_system, 
        digital_control, 
        eta2, 
        K1, 
        K2 = 0, 
        stop_after_number_of_iterations = (1 << 63),
        Ts=None,
        downsample = 1):
        # Check inputs
        if (K1 < 1):
            raise BaseException("K1 must be a positive integer.")
        self.K1 = K1
        if (K2 < 0):
            raise BaseException("K2 must be a non negative integer.")
        self.K2 = K2
        self.K3 = K1 + K2
        self.analog_system = analog_system
        self.digital_control = digital_control
        if(eta2 < 0):
            raise BaseException("eta2 must be non negative.")
        if Ts:
            self.Ts = Ts
        else:
            self.Ts =  digital_control.T
        self.eta2 = eta2
        self.control_signal = control_signal_sequence

        if (downsample != 1):
            raise NotImplementedError("Downsampling currently not implemented for DigitalEstimator")

        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        self._estimate_pointer = self.K1
        
        # For transfer functions
        self.eta2Matrix = np.eye(self.analog_system.CT.shape[0]) * self.eta2

        self._stop_iteration = False

        # Initalize filters
        self._N = self.analog_system.N
        self._M = self.analog_system.M
        self._L = self.analog_system.L
        self._compute_filter_coefficients(analog_system, digital_control, eta2)
        self._allocate_memory_buffers()
    
    def _compute_filter_coefficients(self, AnalogSystem analog_system, DigitalControl digital_control, double eta2):
        # Compute filter coefficients
        cdef int row, col
        A = np.array(analog_system.A).transpose()
        B = np.array(analog_system.CT).transpose()
        Q = np.dot(np.array(analog_system.B), np.array(analog_system.B).transpose())
        R = eta2 * np.eye(analog_system.N_tilde)
        # Solve care
        cdef double[:,:] Vf = care(A, B, Q, R)
        cdef double[:,:] Vb = care(-A, B, Q, R)
        cdef double[:,:] Vf_plus_Vb = np.zeros((self._N, self._N))
        for row in range(self._N):
            for col in range(self._N):
                Vf_plus_Vb[row, col] = Vb[row, col] + Vf[row, col]        
        cdef double T = self.Ts
        CCT = np.dot(np.array(analog_system.CT).transpose(), np.array(analog_system.CT))
        tempAf = analog_system.A - np.dot(Vf, CCT) / eta2
        tempAb = analog_system.A + np.dot(Vb, CCT) / eta2
        self.Af = expm(tempAf * T)
        self.Ab = expm(-tempAb * T)
        Gamma = np.array(analog_system.Gamma)
        # Solve IVPs
        self.Bf = np.zeros((self._N, self._M))
        self.Bb = np.zeros((self._N, self._M))

        atol = 1e-200
        rtol = 1e-10
        max_step = T/1000.0
        for m in range(self._M):
            derivative = lambda t, x: np.dot(tempAf, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBf = solve_ivp(derivative, (0, T), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            derivative = lambda t, x: - np.dot(tempAb, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBb = -solve_ivp(derivative, (0, T), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            for n in range (self._N):
                self.Bf[n, m] = solBf[n]
                self.Bb[n, m] = solBb[n]
        self.WT = solve(Vf_plus_Vb, analog_system.B).transpose()

    def _allocate_memory_buffers(self):
        # Allocate memory buffers
        self._control_signal = np.zeros((self.K3, self._M), dtype=np.int8)
        self._estimate = np.zeros((self.K1, self._L), dtype=np.double)
        self._control_signal_in_buffer = 0
        self._mean = np.zeros((self.K1 + 1, self._N), dtype=np.double)

    def _compute_batch(self):
        cdef int k1, k2, k3, n
        cdef double [:] temp_forward_mean
        temp_forward_mean = np.zeros(self._N, dtype=np.double) 
        # check if ready to compute buffer
        if (self._control_signal_in_buffer < self.K3):
            raise BaseException("Control signal buffer not full")
        # compute lookahead
        for k1 in range(self.K3 - 1, self.K1 - 1, -1):
            temp = np.dot(self.Ab, self._mean[self.K1, :]) + np.dot(self.Bb, self._control_signal[k1, :])
            for n in range(self._N):
                self._mean[self.K1, n] = temp[n]
        # compute forward recursion
        for k2 in range(self.K1):
            temp = np.dot(self.Af, self._mean[k2, :]) + np.dot(self.Bf, self._control_signal[k2, :])
            if (k2 < self.K1 - 1):
                for n in range(self._N):
                    self._mean[k2 + 1, n] = temp[n]
            else:
                # temp = np.dot(self._Af, temp) + np.dot(self._Bf, self._control_signal[k2 + 1, :])
                for n in range(self._N):
                    temp_forward_mean[n] = temp[n]
        # compute backward recursion and estimate
        for k3 in range(self.K1 - 1, -1, -1):
            temp = np.dot(self.Ab, self._mean[k3 + 1, :]) + np.dot(self.Bb, self._control_signal[k3, :])
            temp_estimate = np.dot(self.WT, temp - self._mean[k3, :])
            for l in range(self._L):
                self._estimate[k3, l] = temp_estimate[l]
            for n in range(self._N):
                self._mean[k3, n] = temp[n]
        # reset intital means
        for n in range(self._N):
            self._mean[0, n] = temp_forward_mean[n]
            self._mean[self.K1, n] = 0
        # rotate buffer to make place for new control signals
        self._control_signal = np.roll(self._control_signal, -self.K1, axis=0)
        self._control_signal_in_buffer -= self.K1
        
    

    def _input(self, char [:] s):
        cdef int m
        if (self._control_signal_in_buffer == (self.K3)):
            raise BaseException("Input buffer full. You must compute batch before adding more control signals")
        for m in range(self._M):
            self._control_signal[self._control_signal_in_buffer, m] = 2 * s[m] - 1
        self._control_signal_in_buffer += 1
        return self._control_signal_in_buffer > (self.K3 - 1)    
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Check if the end of prespecified size
        if(self.number_of_iterations < self._iteration ):
            raise StopIteration
        self._iteration += 1

        # Check if there are estimates in the estimate buffer
        if(self._estimate_pointer < self.K1):
            temp = np.array(self._estimate[self._estimate_pointer, :], dtype=np.double) 
            self._estimate_pointer += 1
            return temp
        # Check if stop iteration has been raised in previous batch
        if (self._stop_iteration):
            print("Warning: StopIteration recived by estimator.")
            raise StopIteration
        # Otherwise start reciving control signals
        full = False

        # Fill up batch with new control signals.
        while (not full):
            # next(self.control_signal) calls the control signal
            # generator and thus recives new control 
            # signal samples
            try:
                control_signal_sample = next(self.control_signal)
            except StopIteration:
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

    def noise_transfer_function(self, double [:] omega):
        """Compute the noise transfer function (NTF) at the angular frequencies of the omega array.

        Specifically, computes

        :math:`\\text{NTF}( \omega) = \mathbf{G}( \omega)^\mathsf{H} \\left( \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N \\right)^{-1}`

        for each angular frequency in omega where where :math:`\mathbf{G}(\omega)\in\mathbb{R}^{N \\times L}` is the ATF matrix of the analog system
        and :math:`\mathbf{I}_N` represents a square identity matrix.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for evaluation.

        Returns
        -------
        `array_like`, shape=(L, N_tilde, K)
            return NTF evaluated at K different angular frequencies.
        """
        result = np.zeros((self.analog_system.L, self.analog_system.N, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o]))
            G = G.reshape((self.analog_system.N, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, :, index] = np.abs(np.dot(GH, linalg.inv(GGH + self.eta2Matrix)))
        return result


    def signal_transfer_function(self, double [:] omega):
        """Compute the signal transfer function (STF) at the angular frequencies of the omega array.

        Specifically, computes

        :math:`\\text{STF}( \omega) = \mathbf{G}( \omega)^\mathsf{H} \\left( \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N \\right)^{-1} \mathbf{G}( \omega)`

        for each angular frequency in omega where where :math:`\mathbf{G}(\omega)\in\mathbb{R}^{N \\times L}` is the ATF matrix of the analog system
        and :math:`\mathbf{I}_N` represents a square identity matrix.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for evaluation.

        Returns
        -------
        `array_like`, shape=(L, K)
            return STF evaluated at K different angular frequencies.
        """
        result = np.zeros((self.analog_system.L, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o]))
            G = G.reshape((self.analog_system.N_tilde, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, index] = np.abs(np.dot(GH, np.dot(linalg.inv(GGH + self.eta2Matrix), G)))
        return result

    def __str__(self):
        return f"Digital estimator is parameterized as \neta2 = {self.eta2:.2f}, {20 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},\nand\nnumber_of_iterations = {self.number_of_iterations}\nResulting in the filter coefficients\nAf = \n{self.Af},\nAb = \n{self.Ab},\nBf = \n{self.Bf},\nBb = \n{self.Bb},\nand WT = \n{self.WT}."


class ParallelEstimator:
    """Parallelized batch estimator implementation.
    
    The parallel estimator estimates a filtered version :math:`\hat{\mathbf{u}}(t)` (shaped by 
    :py:func:`signal_transfer_function`) of the 
    input signal :math:`\mathbf{u}(t)` from a sequence of control signals :math:`\mathbf{s}[k]`.

    Specifically, the parallel estimator is a modified version of the default estimator
    :py:class:`cbadc.digital_estimator.DigitalEstimator` where the the filter matrices are diagonlized
    enabling a more efficient and possible parallelizable filter implementation. The estimate is computed as

    :math:`\overrightarrow{\mathbf{m}}[k][n] = f_a[n] \cdot \overrightarrow{\mathbf{m}}[k-1][n] + \sum_{m=0}^{M-1} f_b[n, m] \cdot \mathbf{s}[k-1][m]`,
    :math:`\overleftarrow{\mathbf{m}}[k][n] = b_a \cdot \overrightarrow{\mathbf{m}}[k+1][n] + \sum_{m=0}^{M-1} b_b[n, m] \cdot \mathbf{s}[k][m]`,

    and

    :math:`\hat{\mathbf{u}}(k T)[\ell] = \sum_{n=0}^N f_w[n] \cdot \overrightarrow{\mathbf{m}}[k][n] + b_w[n] \cdot \overleftarrow{\mathbf{m}}[k][n]`

    where :math:`f_a, b_a \in \mathbb{R}^{N}`, :math:`f_b, b_b \in \mathbb{R}^{N \\times M}`,
    and :math:`f_w, b_w \in \mathbb{R}^{L \\times N}` are the precomputed filter coefficient formed
    from the filter coefficients as in :py:class:`cbadc.digital_estimator.DigitalEstimator`.

    Parameters
    ----------
    control_signal_sequence : iterator
        a generator which outputs a sequence of control signals.
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
        determine a max number of iterations by the generator, defaults to :math:`2^{63}`.
    Ts: `float`, `optional`
        the sampling time, defaults to the time period of the digital control.
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
        a generator suppling control signals as :py:class:`cbadc.digital_control.DigitalControl`.
    number_of_iterations : `int`
        number of iterations until generator raises :py:class:`StopIteration`.
    downsample: `int`, `optional`
        The downsampling factor compared to the rate of the control signal.
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

    def __init__(self, 
        control_signal_sequence, 
        analog_system, 
        digital_control, 
        eta2, 
        K1, 
        K2 = 0, 
        stop_after_number_of_iterations = (1 << 63),
        Ts=None,
        downsample = 1):
        # Check inputs
        if (K1 < 1):
            raise BaseException("K1 must be a positive integer.")
        self.K1 = K1
        if (K2 < 0):
            raise BaseException("K2 must be a non negative integer.")
        self.K2 = K2
        self.K3 = K1 + K2
        self.analog_system = analog_system
        self.digital_control = digital_control
        if(eta2 < 0):
            raise BaseException("eta2 must be non negative.")
        if Ts:
            self.Ts = Ts
        else:
            self.Ts =  digital_control.T
        self.eta2 = eta2
        self.control_signal = control_signal_sequence

        if (downsample != 1):
            raise NotImplementedError("Downsampling currently not implemented for ParallelEstimator")

        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        self._estimate_pointer = self.K1
        
        # For transfer functions
        self.eta2Matrix = np.eye(self.analog_system.CT.shape[0]) * self.eta2

        self._stop_iteration = False

        # Initalize filters
        self._N = self.analog_system.N
        self._M = self.analog_system.M
        self._L = self.analog_system.L
        self._compute_filter_coefficients(analog_system, digital_control, eta2)
        self._allocate_memory_buffers()
    
    def _compute_filter_coefficients(self, AnalogSystem analog_system, DigitalControl digital_control, double eta2):
        # Compute filter coefficients
        cdef int row, col
        A = np.array(analog_system.A).transpose()
        B = np.array(analog_system.CT).transpose()
        Q = np.dot(np.array(analog_system.B), np.array(analog_system.B).transpose())
        R = eta2 * np.eye(analog_system.N_tilde)
        # Solve care
        cdef double[:,:] Vf = care(A, B, Q, R)
        cdef double[:,:] Vb = care(-A, B, Q, R)
        cdef double[:,:] Vf_plus_Vb = np.zeros((self._N, self._N))
        for row in range(self._N):
            for col in range(self._N):
                Vf_plus_Vb[row, col] = Vb[row, col] + Vf[row, col]        
        cdef double T = self.Ts
        CCT = np.dot(np.array(analog_system.CT).transpose(), np.array(analog_system.CT))
        tempAf = analog_system.A - np.dot(Vf, CCT) / eta2
        tempAb = analog_system.A + np.dot(Vb, CCT) / eta2
        self.Af = expm(tempAf * T)
        self.Ab = expm(-tempAb * T)
        Gamma = np.array(analog_system.Gamma)
        # Solve IVPs
        self.Bf = np.zeros((self._N, self._M))
        self.Bb = np.zeros((self._N, self._M))

        atol = 1e-200
        rtol = 1e-10
        max_step = T/1000.0
        for m in range(self._M):
            derivative = lambda t, x: np.dot(tempAf, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBf = solve_ivp(derivative, (0, T), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            derivative = lambda t, x: - np.dot(tempAb, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBb = -solve_ivp(derivative, (0, T), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            for n in range (self._N):
                self.Bf[n, m] = solBf[n]
                self.Bb[n, m] = solBb[n]
        self.WT = solve(Vf_plus_Vb, analog_system.B).transpose()

        temp, Q_f = np.linalg.eig(self.Af)
        self.forward_a = np.array(temp, dtype=np.complex128)
        Q_f_inv = np.linalg.pinv(Q_f, rcond=1e-20)
        temp, Q_b = np.linalg.eig(self.Ab)
        self.backward_a = np.array(temp, dtype=np.complex128)
        Q_b_inv = np.linalg.pinv(Q_b, rcond=1e-20)

        self.forward_b  = np.array(np.dot(Q_f_inv, self.Bf), dtype=np.complex128)
        self.backward_b = np.array(np.dot(Q_b_inv, self.Bb), dtype=np.complex128)

        self.forward_w = -np.array(np.dot(self.WT, Q_f), dtype=np.complex128)
        self.backward_w = np.array(np.dot(self.WT, Q_b), dtype=np.complex128)


    def _allocate_memory_buffers(self):
        # Allocate memory buffers
        self._control_signal = np.zeros((self.K3, self._M), dtype=np.int8)
        self._estimate = np.zeros((self.K1, self._L), dtype=np.double)
        self._control_signal_in_buffer = 0
        self._mean = np.zeros((self._N), dtype=np.complex128)

    def _compute_batch(self):
        cdef int k1, k2, k3, n
        cdef int N = self._N
        cdef int M = self._M
        cdef int L = self._L
        cdef int K1 = self.K1
        cdef int K3 = self.K3
        cdef complex mean = 0
        # check if ready to compute buffer
        if (self._control_signal_in_buffer < self.K3):
            raise BaseException("Control signal buffer not full")

        self._estimate = np.zeros((self.K1, self._L), dtype=np.double)

        for n in range(N):
            mean = self._mean[n]
            for k1 in range(K1):
                for l in range(L):                    
                    self._estimate[k1, l] += np.real(self.forward_w[l, n] * mean)
                mean = self.forward_a[n] * mean
                for m in range(M):
                    if(self._control_signal[k1, m]):
                        mean += self.forward_b[n, m]
                    else:
                        mean -= self.forward_b[n, m]
            self._mean[n] = mean
            mean = 0.0
            for k3 in range(K3-1, -1, -1):
                mean = self.backward_a[n] * mean
                for m in range(M):
                    if(self._control_signal[k3, m]):
                        mean += self.backward_b[n, m]
                    else:
                        mean -= self.backward_b[n, m]
                if (k3 < K1):
                    for l in range(L):
                        self._estimate[k3, l] += np.real(self.backward_w[l, n] * mean)
        self._control_signal = np.roll(self._control_signal, -self.K1, axis=0)
        self._control_signal_in_buffer -= self.K1
        
    

    def _input(self, char [:] s):
        cdef int m
        if (self._control_signal_in_buffer == (self.K3)):
            raise BaseException("Input buffer full. You must compute batch before adding more control signals")
        for m in range(self._M):
            self._control_signal[self._control_signal_in_buffer, m] = s[m]
        self._control_signal_in_buffer += 1
        return self._control_signal_in_buffer > (self.K3 - 1)    
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Check if the end of prespecified size
        if(self.number_of_iterations < self._iteration ):
            raise StopIteration
        self._iteration += 1

        # Check if there are estimates in the estimate buffer
        if(self._estimate_pointer < self.K1):
            temp = np.array(self._estimate[self._estimate_pointer, :], dtype=np.double) 
            self._estimate_pointer += 1
            return temp
        # Check if stop iteration has been raised in previous batch
        if (self._stop_iteration):
            print("Warning: StopIteration recived by estimator.")
            raise StopIteration
        # Otherwise start reciving control signals
        full = False

        # Fill up batch with new control signals.
        while (not full):
            # next(self.control_signal) calls the control signal
            # generator and thus recives new control 
            # signal samples
            try:
                control_signal_sample = next(self.control_signal)
            except StopIteration:
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

    def noise_transfer_function(self, double [:] omega):
        """Compute the noise transfer function (NTF) at the angular frequencies of the omega array.

        Specifically, computes

        :math:`\\text{NTF}( \omega) = \mathbf{G}( \omega)^\mathsf{H} \\left( \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N \\right)^{-1}`

        for each angular frequency in omega where where :math:`\mathbf{G}(\omega)\in\mathbb{R}^{N \\times L}` is the ATF matrix of the analog system
        and :math:`\mathbf{I}_N` represents a square identity matrix.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for evaluation.

        Returns
        -------
        `array_like`, shape=(L, N_tilde, K)
            return NTF evaluated at K different angular frequencies.
        """
        result = np.zeros((self.analog_system.L, self.analog_system.N, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o]))
            G = G.reshape((self.analog_system.N, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, :, index] = np.abs(np.dot(GH, linalg.inv(GGH + self.eta2Matrix)))
        return result


    def signal_transfer_function(self, double [:] omega):
        """Compute the signal transfer function (STF) at the angular frequencies of the omega array.

        Specifically, computes

        :math:`\\text{STF}( \omega) = \mathbf{G}( \omega)^\mathsf{H} \\left( \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N \\right)^{-1} \mathbf{G}( \omega)`

        for each angular frequency in omega where where :math:`\mathbf{G}(\omega)\in\mathbb{R}^{N \\times L}` is the ATF matrix of the analog system
        and :math:`\mathbf{I}_N` represents a square identity matrix.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for evaluation.

        Returns
        -------
        `array_like`, shape=(L, K)
            return STF evaluated at K different angular frequencies.
        """
        result = np.zeros((self.analog_system.L, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o]))
            G = G.reshape((self.analog_system.N_tilde, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, index] = np.abs(np.dot(GH, np.dot(linalg.inv(GGH + self.eta2Matrix), G)))
        return result

    def __str__(self):
        return f"Parallel estimator is parameterized as \neta2 = {self.eta2:.2f}, {20 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},\nand\nnumber_of_iterations = {self.number_of_iterations}\nResulting in the filter coefficients\nf_a = \n{self.forward_a},\nb_a = \n{self.backward_b},\nf_b = \n{self.forward_b},\nb_b = \n{self.backward_b},\nf_w = \n{self.forward_w},\nand b_w = \n{self.backward_w}."




class IIRFilter():
    """IIR filter implementation of the digital estimator.
    
    Specifically, the IIR filter estimator estimates a filtered version :math:`\hat{\mathbf{u}}(t)` (shaped by 
    :py:func:`signal_transfer_function`) of the 
    input signal :math:`\mathbf{u}(t)` from a sequence of control signals :math:`\mathbf{s}[k]`.

    Specifically, the estimate is of the form
    
    :math:`\hat{\mathbf{u}}(k T) = - \mathbf{W}^{\mathsf{T}} \overrightarrow{\mathbf{m}}_k + \sum_{\ell=0}^{K_2} \mathbf{h}[\ell] \mathbf{s}[k - \ell]`

    where 

    :math:`\mathbf{h}[\ell]=\\begin{cases}\mathbf{W}^{\mathsf{T}} \mathbf{A}_b^\ell \mathbf{B}_b & \mathrm{if} \, \ell \geq 0 \\\  -\mathbf{W}^{\mathsf{T}} \mathbf{A}_f^{-\ell + 1} \mathbf{B}_f & \mathrm{else} \\end{cases}`

    :math:`\overrightarrow{\mathbf{m}}_k = \mathbf{A}_f \mathbf{m}_{k-1} + \mathbf{B}_f \mathbf{s}[k-1]`

    and :math:`\mathbf{W}^{\mathsf{T}}`, :math:`\mathbf{A}_b`, 
    :math:`\mathbf{B}_b`, :math:`\mathbf{A}_f`, and :math:`\mathbf{B}_f`
    are computed based on the analog system, the sample period :math:`T_s`, and the
    digital control's DAC waveform as desribed in 
    `control-bounded converters <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf?sequence=1&isAllowed=y#page=67/>`_.

    Parameters
    ----------
    control_signal : iterator
        a generator which outputs a sequence of control signals.
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system (necessary to compute the estimators filter coefficients).
    digital_control : :py:class:`cbadc.digital_control.DigitalControl`
        a digital control (necessary to determine the corresponding DAC waveform).
    eta2 : `float`
        the :math:`\eta^2` parameter determines the bandwidth of the estimator.
    K2 : `int`, `optional`
        lookahead size, defaults to 0.
    stop_after_number_of_iterations : `int`
        determine a max number of iterations by the generator, defaults to  :math:`2^{63}`.
    estimator_type : `str`, `optional`
        determine which filtering method :code:`'quadratic', 'parallel', 'mid-point'` 
        to use, defaults to :code:`'quadratic'`.
    Ts: `float`, `optional`
        the sampling time, defaults to the time period of the digital control.
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
        a generator suppling control signals as :py:class:`cbadc.digital_control.DigitalControl`.
    number_of_iterations : `int`
        number of iterations until generator raises :py:class:`StopIteration`.
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
    h : `array_like`, shape=(K2, L, M)
        filter impulse response.
    downsample: `int`
        down sampling rate in relation to the control period :math:`T`.

    Yields
    ------
    `array_like`, shape=(L,)
        an input estimate sample :math:`\hat{\mathbf{u}}(t)`

    """

    def __init__(self, 
        control_signal_sequence, 
        AnalogSystem analog_system, 
        DigitalControl digital_control, 
        double eta2, 
        int K2, 
        stop_after_number_of_iterations=(1 << 63), 
        Ts=None,
        downsample = 1
        ):
        """Initializes filter coefficients
        """
        cdef int row, col
        if (K2 < 0):
            raise "K2 must be non negative integer."
        self.K2 = K2
        self.analog_system = analog_system
        if(eta2 < 0):
            raise "eta2 must be non negative."
        self.eta2 = eta2
        self.control_signal = control_signal_sequence
        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0

        self.downsample = int(downsample)
        
        # For transfer functions
        self.eta2Matrix = np.eye(self.analog_system.CT.shape[0]) * self.eta2
        self._M = analog_system.M
        self._N = analog_system.N
        self._L = analog_system.L
        # Compute filter coefficients
        A = np.array(analog_system.A).transpose()
        B = np.array(analog_system.CT).transpose()
        Q = np.dot(np.array(analog_system.B), np.array(analog_system.B).transpose())
        R = eta2 * np.eye(analog_system.N_tilde)
        # Solve care
        cdef double[:,:] Vf = care(A, B, Q, R)
        cdef double[:,:] Vb = care(-A, B, Q, R)
        cdef double[:,:] Vf_plus_Vb = np.zeros((self._N, self._N))
        for row in range(self._N):
            for col in range(self._N):
                Vf_plus_Vb[row, col] = Vf[row, col] + Vb[row, col]
        if Ts:
            self.Ts = Ts
        else: 
            self.Ts = digital_control.T
        CCT = np.dot(np.array(analog_system.CT).transpose(), np.array(analog_system.CT))
        tempAf = analog_system.A - np.dot(Vf, CCT) / eta2
        tempAb = analog_system.A + np.dot(Vb, CCT) / eta2
        self.Af = expm(tempAf * self.Ts)
        self.Ab = expm(-tempAb * self.Ts)
        Gamma = np.array(analog_system.Gamma)
        # Solve IVPs
        self.Bf = np.zeros((self._N, self._M), dtype=np.double)
        self.Bb = np.zeros((self._N, self._M), dtype=np.double)
        atol = 1e-200
        rtol = 1e-12
        max_step = self.Ts * 1e-4
        for m in range(self._M):
            derivative = lambda t, x: np.dot(tempAf, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBf = solve_ivp(derivative, (0, self.Ts), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            derivative = lambda t, x: - np.dot(tempAb, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBb = -solve_ivp(derivative, (0, self.Ts), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            for n in range (self._N):
                self.Bf[n, m] = solBf[n]
                self.Bb[n, m] = solBb[n]
        self.WT = solve(Vf_plus_Vb, analog_system.B).transpose()
        # print(f"Af : {np.array(Af)}")
        # print(Ab)
        # print(f"Bf : {np.array(Bf)}")
        # print(np.array(Bb))

        # Initalize filter
        self.h = np.zeros((self.K2, self._L, self._M), dtype=np.double)
        # Compute lookback
        temp2 = np.copy(self.Bb)
        for k2 in range(self.K2):
            self.h[k2, :, :] = np.dot(self.WT, temp2)
            temp2 = np.dot(self.Ab, temp2)
        self._control_signal_valued = np.zeros((self.K2, self._M), dtype=np.int8)
        self._mean = np.zeros(self._N, dtype=np.double)
    
    def __iter__(self):
        return self

    def __next__(self):
        # Check if the end of prespecified size
        self._iteration += 1
        if(self.number_of_iterations and self.number_of_iterations < self._iteration ):
            raise StopIteration

        # Rotate control_signal vector
        self._control_signal_valued = np.roll(self._control_signal_valued, -1, axis=0)

        # insert new control signal
        try:
            temp = self.control_signal.__next__()
        except StopIteration:
            print("Warning estimator recived Stop Iteration")
            raise StopIteration

        for m in range(self._M):    
            self._control_signal_valued[-1, m] = 2 * temp[m] - 1

        # self._control_signal_valued.shape -> (K2, M)
        # self.h.shape -> (K2, L, M)
        result = - np.dot(self.WT, self._mean)
        self._mean = np.dot(self.Af, self._mean) + np.dot(self.Bf, self._control_signal_valued[0, :])
        if (((self._iteration - 1) % self.downsample) == 0):
            return np.einsum('ijk,ik', self.h, self._control_signal_valued) + result
        return self.__next__()

    def lookahead(self):
        """Return lookahead size :math:`K2`

        Returns
        -------
        int
            lookahead size.
        """
        return self.K2

    def noise_transfer_function(self, double [:] omega):
        """Compute the noise transfer function (NTF) at the angular frequencies of the omega array.

        Specifically, computes

        :math:`\\text{NTF}( \omega) = \mathbf{G}( \omega)^\mathsf{H} \\left( \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N \\right)^{-1}`

        for each angular frequency in omega where where :math:`\mathbf{G}(\omega)\in\mathbb{R}^{N \\times L}` is the ATF matrix of the analog system
        and :math:`\mathbf{I}_N` represents a square identity matrix.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for evaluation.

        Returns
        -------
        `array_like`, shape=(L, N_tilde, K)
            return NTF evaluated at K different angular frequencies.
        """
        result = np.zeros((self.analog_system.L, self.analog_system.N, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o]))
            G = G.reshape((self.analog_system.N, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, :, index] = np.abs(np.dot(GH, linalg.inv(GGH + self.eta2Matrix)))
        return result


    def signal_transfer_function(self, double [:] omega):
        """Compute the signal transfer function (STF) at the angular frequencies of the omega array.

        Specifically, computes

        :math:`\\text{STF}( \omega) = \mathbf{G}( \omega)^\mathsf{H} \\left( \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N \\right)^{-1} \mathbf{G}( \omega)`

        for each angular frequency in omega where where :math:`\mathbf{G}(\omega)\in\mathbb{R}^{N \\times L}` is the ATF matrix of the analog system
        and :math:`\mathbf{I}_N` represents a square identity matrix.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for evaluation.

        Returns
        -------
        `array_like`, shape=(L, K)
            return STF evaluated at K different angular frequencies.
        """
        result = np.zeros((self.analog_system.L, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o]))
            G = G.reshape((self.analog_system.N_tilde, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, index] = np.abs(np.dot(GH, np.dot(linalg.inv(GGH + self.eta2Matrix), G)))
        return result

    def __str__(self):
        return f"IIR estimator is parameterized as \neta2 = {self.eta2:.2f}, {20 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK2 = {self.K2},\nand\nnumber_of_iterations = {self.number_of_iterations}.\nResulting in the filter coefficients\nAf = \n{self.Af},\nBf = \n{self.Bf},WT = \n{self.WT},\n and h = \n{self.h}."


class FIRFilter():
    """FIR filter implementation of the digital estimator.
    
    Specifically, the FIR filter estimator estimates a filtered version :math:`\hat{\mathbf{u}}(t)` (shaped by 
    :py:func:`signal_transfer_function`) of the 
    input signal :math:`\mathbf{u}(t)` from a sequence of control signals :math:`\mathbf{s}[k]`.

    Specifically, the estimate is of the form
    
    :math:`\hat{\mathbf{u}}(k T) = \sum_{\ell=-K_1}^{K_2} \mathbf{h}[\ell] \mathbf{s}[k - \ell]`

    where 

    :math:`\mathbf{h}[\ell]=\\begin{cases}\mathbf{W}^{\mathsf{T}} \mathbf{A}_b^\ell \mathbf{B}_b & \mathrm{if} \, \ell \geq 0 \\\  -\mathbf{W}^{\mathsf{T}} \mathbf{A}_f^{-\ell + 1} \mathbf{B}_f & \mathrm{else} \\end{cases}`

    and :math:`\mathbf{W}^{\mathsf{T}}`, :math:`\mathbf{A}_b`, 
    :math:`\mathbf{B}_b`, :math:`\mathbf{A}_f`, and :math:`\mathbf{B}_f`
    are computed based on the analog system, the sample period :math:`T_s`, and the
    digital control's DAC waveform as desribed in 
    `control-bounded converters <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf?sequence=1&isAllowed=y#page=67/>`_.

    Parameters
    ----------
    control_signal : iterator
        a generator which outputs a sequence of control signals.
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system (necessary to compute the estimators filter coefficients).
    digital_control : :py:class:`cbadc.digital_control.DigitalControl`
        a digital control (necessary to determine the corresponding DAC waveform).
    eta2 : `float`
        the :math:`\eta^2` parameter determines the bandwidth of the estimator.
    K1 : `int`
        The lookback size
    K2 : `int`, `optional`
        lookahead size, defaults to 0.
    stop_after_number_of_iterations : `int`
        determine a max number of iterations by the generator, defaults to  :math:`2^{63}`.
    estimator_type : `str`, `optional`
        determine which filtering method :code:`'quadratic', 'parallel', 'mid-point'` 
        to use, defaults to :code:`'quadratic'`.
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
    eta2 : float
        eta2, or equivalently :math:`\eta^2`, sets the bandwidth of the estimator.
    control_signal : :py:class:`cbadc.digital_control.DigitalControl`
        a generator suppling control signals as :py:class:`cbadc.digital_control.DigitalControl`.
    number_of_iterations : `int`
        number of iterations until generator raises :py:class:`StopIteration`.
    K1 : `int`
        number of samples, prior to estimate, used in estimate
    K2 : `int`
        number of lookahead samples per computed batch.
    Ts : `float`
        spacing between samples in seconds.
    mid_point: `bool`
        estimated samples shifted in between control updates, i.e., :math:`\hat{u}(kT + T/2)`.
    downsample: `int`, `optional`
        down sampling rate in relation to the control period :math:`T`.
    Af : `array_like`, shape=(N, N)
        The Af matrix
    Ab : `array_like`, shape=(N, N)
        The Ab matrix
    Bf : `array_like`, shape=(N, M)
        The Bf matrix
    Bb : `array_like`, shape=(N, M)
        The Bb matrix
    WT : `array_like`, shape=(L, N)
        The W matrix transposed
    h : `array_like`, shape=(K3, L, M)
        filter impulse response

    Yields
    ------
    `array_like`, shape=(L,)
        an input estimate sample :math:`\hat{\mathbf{u}}(t)`
    """

    def __init__(self, control_signal_sequence, AnalogSystem analog_system, DigitalControl digital_control, double eta2, int K1, int K2, stop_after_number_of_iterations=(1 << 63), Ts=None, mid_point=False, downsample=1):
        """Initializes filter coefficients
        """
        cdef int row, col
        if (K1 < 0):
            raise BaseException("K1 must be non negative integer.")
        self.K1 = K1
        if (K2 < 1):
            raise BaseException("K2 must be a positive integer.")
        self.K2 = K2
        self.K3 = K1 + K2
        self.analog_system = analog_system
        self.digital_control = digital_control
        if(eta2 < 0):
            raise BaseException("eta2 must be non negative.")
        self.eta2 = eta2
        self.control_signal = control_signal_sequence
        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        
        self.mid_point = mid_point
        self.downsample = int(downsample)

        # For transfer functions
        self.eta2Matrix = np.eye(self.analog_system.CT.shape[0]) * self.eta2
        self._M = analog_system.M
        self._N = analog_system.N
        self._L = analog_system.L
        # Compute filter coefficients
        A = np.array(analog_system.A).transpose()
        B = np.array(analog_system.CT).transpose()
        Q = np.dot(np.array(analog_system.B), np.array(analog_system.B).transpose())
        R = eta2 * np.eye(analog_system.N_tilde)
        # Solve care
        cdef double[:,:] Vf = care(A, B, Q, R)
        cdef double[:,:] Vb = care(-A, B, Q, R)
        cdef double[:,:] Vf_plus_Vb = np.zeros((self._N, self._N))
        for row in range(self._N):
            for col in range(self._N):
                Vf_plus_Vb[row, col] = Vb[row, col] + Vf[row, col]
        if Ts:
            self.Ts = Ts
        else: 
            self.Ts = digital_control.T
        CCT = np.dot(np.array(analog_system.CT).transpose(), np.array(analog_system.CT))
        tempAf = analog_system.A - np.dot(Vf, CCT) / eta2
        tempAb = analog_system.A + np.dot(Vb, CCT) / eta2
        self.Af = expm(tempAf * self.Ts)
        self.Ab = expm(-tempAb * self.Ts)
        Gamma = np.array(analog_system.Gamma)
        # Solve IVPs
        self.Bf = np.zeros((self._N, self._M), dtype=np.double)
        self.Bb = np.zeros((self._N, self._M), dtype=np.double)
        atol = 1e-200
        rtol = 1e-12
        max_step = self.Ts * 1e-4

        if (self.mid_point):
            for m in range(self._M):
                derivative = lambda t, x: np.dot(tempAf, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
                solBf = solve_ivp(derivative, (0, self.Ts/2.0), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
                derivative = lambda t, x: - np.dot(tempAb, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
                solBb = -solve_ivp(derivative, (0, self.Ts/2.0), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
                for n in range (self._N):
                    self.Bf[n, m] = solBf[n]
                    self.Bb[n, m] = solBb[n]
            self.Bf = np.dot(np.eye(self._N) + expm(tempAf * self.Ts / 2.0), self.Bf)
            self.Bb = np.dot(np.eye(self._N) + expm(tempAb * self.Ts / 2.0), self.Bb)

        else:
            for m in range(self._M):
                derivative = lambda t, x: np.dot(tempAf, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
                solBf = solve_ivp(derivative, (0, self.Ts), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
                derivative = lambda t, x: - np.dot(tempAb, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
                solBb = -solve_ivp(derivative, (0, self.Ts), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
                for n in range (self._N):
                    self.Bf[n, m] = solBf[n]
                    self.Bb[n, m] = solBb[n]
        self.WT = solve(Vf_plus_Vb, analog_system.B).transpose()

        # Initalize filter.
        self.h = np.zeros((self.K3, self._L, self._M), dtype=np.double)
        # Compute lookback.
        temp1 = np.copy(self.Bf)
        cdef double [:, :] tempRes
        for k1 in range(self.K1 - 1, -1, -1):
            self.h[k1, :, :] = - np.dot(self.WT, temp1)
            temp1 = np.dot(self.Af, temp1)
        # Compute lookahead.
        temp2 = np.copy(self.Bb)
        for k2 in range(self.K1, self.K3):
            self.h[k2, :, :] = np.dot(self.WT, temp2)
            temp2 = np.dot(self.Ab, temp2)
        self._control_signal_valued = np.zeros((self.K3, self._M), dtype=np.int8)
    
    def __iter__(self):
        return self

    def __next__(self):
        # Check if the end of prespecified size
        self._iteration += 1
        if(self.number_of_iterations and self.number_of_iterations < self._iteration ):
            raise StopIteration

        # Rotate control_signal vector
        self._control_signal_valued = np.roll(self._control_signal_valued, -1, axis=0)

        # insert new control signal
        try:
            temp = self.control_signal.__next__()
        except StopIteration:
            print("Warning estimator recived Stop Iteration")
            raise StopIteration

        for m in range(self._M):    
            self._control_signal_valued[self.K3 - 1, m] = 2 * temp[m] - 1

        # Check for down sampling
        if (((self._iteration - 1) % self.downsample) == 0):
            # self._control_signal_valued.shape -> (K1 + K2, M)
            # self.h.shape -> (K1 + K2, L, M)
            return np.einsum('ijk,ik', self.h, self._control_signal_valued)
            # the Einstein summation results in:
            # result = np.zeros(self._L)
            # for l in range(self._L):
            #    for k in range(self.K1 + self.K2):
            #        for m in range(self._M):
            #            result[l] += self.h[k, l, m] * self._control_signal_valued[k, m]
            # return result
        
        # if not, recusively call self
        return self.__next__()


    def lookback(self):
        """Return lookback size :math:`K1`.

        Returns
        -------
        int
            lookback size.
        """
        return self.K1

    def noise_transfer_function(self, double [:] omega):
        """Compute the noise transfer function (NTF) at the angular frequencies of the omega array.

        Specifically, computes

        :math:`\\text{NTF}( \omega) = \mathbf{G}( \omega)^\mathsf{H} \\left( \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N \\right)^{-1}`

        for each angular frequency in omega where where :math:`\mathbf{G}(\omega)\in\mathbb{R}^{N \\times L}` is the ATF matrix of the analog system
        and :math:`\mathbf{I}_N` represents a square identity matrix.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for evaluation.

        Returns
        -------
        `array_like`, shape=(L, N_tilde, K)
            return NTF evaluated at K different angular frequencies.
        """
        result = np.zeros((self.analog_system.L, self.analog_system.N, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o]))
            G = G.reshape((self.analog_system.N, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, :, index] = np.abs(np.dot(GH, linalg.inv(GGH + self.eta2Matrix)))
        return result


    def signal_transfer_function(self, double [:] omega):
        """Compute the signal transfer function (STF) at the angular frequencies of the omega array.

        Specifically, computes

        :math:`\\text{STF}( \omega) = \mathbf{G}( \omega)^\mathsf{H} \\left( \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N \\right)^{-1} \mathbf{G}( \omega)`

        for each angular frequency in omega where where :math:`\mathbf{G}(\omega)\in\mathbb{R}^{N \\times L}` is the ATF matrix of the analog system
        and :math:`\mathbf{I}_N` represents a square identity matrix.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for evaluation.

        Returns
        -------
        `array_like`, shape=(L, K)
            return STF evaluated at K different angular frequencies.
        """
        result = np.zeros((self.analog_system.L, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function_matrix(np.array([o]))
            G = G.reshape((self.analog_system.N_tilde, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, index] = np.abs(np.dot(GH, np.dot(linalg.inv(GGH + self.eta2Matrix), G)))
        return result

    
    def __str__(self):
        return f"FIR estimator is parameterized as \neta2 = {self.eta2:.2f}, {20 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},\nand\nnumber_of_iterations = {self.number_of_iterations}.\nResulting in the filter coefficients\nh = \n{self.h}."
