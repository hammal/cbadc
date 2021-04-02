""" Digital estimator

This module provides alternative implementations of the digital
estimator.
"""
#cython: language_level=3
from cbadc.linear_filter cimport LinearFilter
from cbadc.filter_mid_point cimport MidPointFilter
from cbadc.offline_computations import care
from scipy.linalg import expm, solve
from scipy.integrate import solve_ivp
import numpy.linalg as linalg
cimport numpy as np
import numpy as np
import math

class DigitalEstimator:
    """The digital estimator estimates a filtered version :math:`\hat{\mathbf{u}}(t)` (shaped by 
    :py:func:`signal_transfer_function`) of the 
    input signal :math:`\mathbf{u}(t)` from a sequence of control signals :math:`\mathbf{s}[k]`.

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
    estimator_type : `str`, `optional`
        determine which filtering method :code:`'quadratic', 'parallel', 'mid-point'` 
        to use, defaults to :code:`'quadratic'`.
    Ts: `float`, `optional`
        the sampling time, defaults to the time period of the digital control.

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
        number of samples per estimate batch.
    K2 : `int`
        number of lookahead samples per computed batch.
    estimator_type : `str`
        describing type of estimator implemenation.
    Ts : `float`
        spacing between samples in seconds.
    

    Yields
    ------
    `array_like`, shape=(L,)
        an input estimate sample :math:`\hat{\mathbf{u}}(t)`

    Examples
    --------
    TODO:
    """

    def __init__(self, 
        control_signal_sequence, 
        analog_system, 
        digital_control, 
        eta2, 
        K1, 
        K2 = 0, 
        stop_after_number_of_iterations = (1 << 63), 
        estimator_type = 'quadratic',
        Ts=None):
        # Check inputs
        if (K1 < 1):
            raise "K1 must be a positive integer."
        self.K1 = K1
        if (K2 < 0):
            raise "K2 must be a non negative integer."
        self.K2 = K2
        self.analog_system = analog_system
        if(eta2 < 0):
            raise "eta2 must be non negative."
        if Ts:
            self.Ts = Ts
        else:
            self.Ts =  digital_control.T
        self.eta2 = eta2
        self.control_signal = control_signal_sequence

        estimation_filter_implementations = {
            'quadratic': lambda : LinearFilter(analog_system, digital_control, eta2, K1, K2 = K2, Ts = self.Ts),
            'mid-point': lambda : MidPointFilter(analog_system, digital_control, eta2, K1, K2 = K2, Ts = self.Ts)
        }
        def not_implemented():
            raise f"{estimator_type} is not a implemented estimator algorithm, currently choose from {estimation_filter_implementations.keys()}"
        self._filter = estimation_filter_implementations.get(estimator_type, not_implemented)()
        self.estimator_type = estimator_type

        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        self._estimate_pointer = self._filter.batch_size()
        
        # For transfer functions
        self.eta2Matrix = np.eye(self.analog_system.CT.shape[0]) * self.eta2

        self._stop_iteration = False
    
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
            G = self.analog_system.transfer_function(np.array([o]))
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
            G = self.analog_system.transfer_function(np.array([o]))
            G = G.reshape((self.analog_system.N_tilde, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = np.dot(G, GH)
            result[:, index] = np.abs(np.dot(GH, np.dot(linalg.inv(GGH + self.eta2Matrix), G)))
        return result


    def __iter__(self):
        return self
    
    def __next__(self):
        # Check if the end of prespecified size
        if(self.number_of_iterations < self._iteration ):
            raise StopIteration
        self._iteration += 1

        # Check if there are estimates in the estimate buffer
        if(self._estimate_pointer < self._filter.batch_size()):
            self._estimate_pointer += 1
            return self._filter.output(self._estimate_pointer - 1)

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
            full = self._filter.input(control_signal_sample)
            
        # Compute new batch of K1 estimates
        self._filter.compute_batch()

        # adjust pointer to indicate that estimate buffer
        # is non empty
        self._estimate_pointer -= self._filter.batch_size()
        
        # recursively call itself to return new estimate
        return self.__next__()
    
    def __str__(self):
        return f"Digital estimator is parameterized as \neta2 = {self.eta2:.2f}, {10 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},\nestimator_type = {self.estimator_type},\nand\nnumber_of_iterations = {self.number_of_iterations}."


class FIRFilter(DigitalEstimator):
    """The FIR filter estimator estimates a filtered version :math:`\hat{\mathbf{u}}(t)` (shaped by 
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

    Yields
    ------
    `array_like`, shape=(L,)
        an input estimate sample :math:`\hat{\mathbf{u}}(t)`

    Examples
    --------
    TODO:
    """

    def __init__(self, control_signal_sequence, analog_system, digital_control, double eta2, int K1, int K2, stop_after_number_of_iterations=(1 << 63), Ts=None):
        """Initializes filter coefficients
        """
        if (K1 < 0):
            raise "K1 must be non negative integer."
        self.K1 = K1
        if (K2 < 1):
            raise "K2 must be a positive integer."
        self.K2 = K2
        self.analog_system = analog_system
        if(eta2 < 0):
            raise "eta2 must be non negative."
        self.eta2 = eta2
        self.control_signal = control_signal_sequence
        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        
        # For transfer functions
        self.eta2Matrix = np.eye(self.analog_system.CT.shape[0]) * self.eta2
        self._M = analog_system.M
        self._N = analog_system.N
        self._L = analog_system.L
        self._K1 = K1
        self._K2 = K2
        # Compute filter coefficients
        A = np.array(analog_system.A).transpose()
        B = np.array(analog_system.CT).transpose()
        Q = np.dot(np.array(analog_system.B), np.array(analog_system.B).transpose())
        R = eta2 * np.eye(analog_system.N_tilde)
        # Solve care
        Vf, Vb = care(A, B, Q, R)
        if Ts:
            self.Ts = Ts
        else: 
            self.Ts = digital_control.T
        CCT = np.dot(np.array(analog_system.CT).transpose(), np.array(analog_system.CT))
        tempAf = analog_system.A - np.dot(Vf, CCT) / eta2
        tempAb = analog_system.A + np.dot(Vb, CCT) / eta2
        Af = expm(tempAf * self.Ts)
        Ab = expm(-tempAb * self.Ts)
        Gamma = np.array(analog_system.Gamma)
        # Solve IVPs
        Bf = np.zeros((self._N, self._M))
        Bb = np.zeros((self._N, self._M))

        atol = 1e-200
        rtol = 1e-10
        max_step = self.Ts/1000.0
        for m in range(self._M):
            derivative = lambda t, x: np.dot(tempAf, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBf = solve_ivp(derivative, (0, self.Ts), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            derivative = lambda t, x: - np.dot(tempAb, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBb = -solve_ivp(derivative, (0, self.Ts), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            for n in range (self._N):
                Bf[n, m] = solBf[n]
                Bb[n, m] = solBb[n]
        WT = solve(Vf + Vb, analog_system.B).transpose()

        # Initalize filter
        self.h = np.zeros(( self._K1 + self._K2, self._L, self._M), dtype=np.double)
        # Compute lookback
        # Aprod = Af
        Aprod = np.eye(self._N)
        for k1 in range(1, self._K1 + 1):
            self.h[self.K1 - k1, :, :] = -np.dot(WT, np.dot(Aprod, Bf))
            Aprod = np.dot(Aprod, Af)
        
        Aprod = np.eye(self._N)
        for k2 in range(self._K2):
            self.h[self.K1 + k2, :, :] = np.dot(WT, np.dot(Aprod, Bb))
            Aprod = np.dot(Aprod, Ab)
        self._control_signal_valued = np.zeros((self._K1 + self._K2, self._M), dtype=np.int8)
        
    def __next__(self):
        # Check if the end of prespecified size
        if(self.number_of_iterations and self.number_of_iterations < self._iteration ):
            raise StopIteration
        self._iteration += 1

        # Rotate control_signal vector
        self._control_signal_valued = np.roll(self._control_signal_valued, 1, axis=0)

        # insert new control signal
        try:
            temp = self.control_signal.__next__()
        except StopIteration:
            print("Warning estimator recived Stop Iteration")
            raise StopIteration

        for m in range(self._M):    
            self._control_signal_valued[0, m] = 2 * temp[m] - 1

        # self._control_signal_valued.shape -> (K1 + K2, M)
        # self.h.shape -> (K1 + K2, L, M)
        return np.einsum('ijk,ik', self.h, self._control_signal_valued)

    def lookback(self):
        """Return lookback size :math:`K1`.

        Returns
        -------
        int
            lookback size.
        """
        return self._K1

    def lookahead(self):
        """Return lookahead size :math:`K2`

        Returns
        -------
        int
            lookahead size.
        """
        return self._K2
    
    def __str__(self):
        return f"FIR estimator is parameterized as \neta2 = {self.eta2:.2f}, {10 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},\nand\nnumber_of_iterations = {self.number_of_iterations}."


    