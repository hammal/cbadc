# distutils: language = c++
# cython: language_level=3
"""Digital estimators.

This module provides alternative implementations for the control-bounded A/D conterter's
digital estimator.
"""

from cbadc.linear_filter cimport LinearFilter
from cbadc.filter_mid_point cimport MidPointFilter
from cbadc.parallel_filter cimport ParallelFilter
from cbadc.offline_computations import care
from scipy.linalg import expm, solve
from scipy.integrate import solve_ivp
import numpy.linalg as linalg
cimport numpy as np
import numpy as np
import math

class DigitalEstimator:
    """Computes batches of estimates from control signals.
    
    Specifically, the digital estimator estimates a filtered version :math:`\hat{\mathbf{u}}(t)` (shaped by 
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
            'mid-point': lambda : MidPointFilter(analog_system, digital_control, eta2, K1, K2 = K2, Ts = self.Ts),
            'parallel': lambda : ParallelFilter(analog_system, digital_control, eta2, K1, K2 = K2, Ts = self.Ts)
        }
        def not_implemented():
            raise NotImplementedError(f"{estimator_type} is not a implemented estimator algorithm, currently choose from {estimation_filter_implementations.keys()}")
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
            temp = self._filter.output(self._estimate_pointer)
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
            full = self._filter.input(control_signal_sample)
        
        # Compute new batch of K1 estimates
        self._filter.compute_batch()
        # adjust pointer to indicate that estimate buffer
        # is non empty
        self._estimate_pointer -= self._filter.batch_size()
        
        # recursively call itself to return new estimate
        return self.__next__()
    
    def __str__(self):
        return f"Digital estimator is parameterized as \neta2 = {self.eta2:.2f}, {20 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},\nestimator_type = {self.estimator_type},\nand\nnumber_of_iterations = {self.number_of_iterations}."

class IIRFilter(DigitalEstimator):
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
    K2 : `int`
        number of lookahead samples per computed batch.
    Ts : `float`
        spacing between samples in seconds.

    Yields
    ------
    `array_like`, shape=(L,)
        an input estimate sample :math:`\hat{\mathbf{u}}(t)`

    """

    def __init__(self, control_signal_sequence, analog_system, digital_control, double eta2, int K2, stop_after_number_of_iterations=(1 << 63), Ts=None):
        """Initializes filter coefficients
        """
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
        Vf, Vb = care(A, B, Q, R)
        if Ts:
            self.Ts = Ts
        else: 
            self.Ts = digital_control.T
        CCT = np.dot(np.array(analog_system.CT).transpose(), np.array(analog_system.CT))
        tempAf = analog_system.A - np.dot(Vf, CCT) / eta2
        tempAb = analog_system.A + np.dot(Vb, CCT) / eta2
        self._Af = expm(tempAf * self.Ts)
        Ab = expm(-tempAb * self.Ts)
        Gamma = np.array(analog_system.Gamma)
        # Solve IVPs
        self._Bf = np.zeros((self._N, self._M), dtype=np.double)
        Bb = np.zeros((self._N, self._M), dtype=np.double)
        atol = 1e-200
        rtol = 1e-12
        max_step = self.Ts * 1e-4
        for m in range(self._M):
            derivative = lambda t, x: np.dot(tempAf, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBf = solve_ivp(derivative, (0, self.Ts), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            derivative = lambda t, x: - np.dot(tempAb, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
            solBb = -solve_ivp(derivative, (0, self.Ts), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
            for n in range (self._N):
                self._Bf[n, m] = solBf[n]
                Bb[n, m] = solBb[n]
        self._WT = solve(Vf + Vb, analog_system.B).transpose()
        # print(f"Af : {np.array(Af)}")
        # print(Ab)
        # print(f"Bf : {np.array(Bf)}")
        # print(np.array(Bb))

        # Initalize filter
        self.h = np.zeros((self.K2, self._L, self._M), dtype=np.double)
        # Compute lookback
        temp2 = np.copy(Bb)
        for k2 in range(self.K2):
            self.h[k2, :, :] = np.dot(self._WT, temp2)
            temp2 = np.dot(Ab, temp2)
        self._control_signal_valued = np.zeros((self.K2, self._M), dtype=np.int8)
        self._mean = np.zeros(self._N, dtype=np.double)
        
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
        result = np.einsum('ijk,ik', self.h, self._control_signal_valued) - np.dot(self._WT, self._mean)
        self._mean = np.dot(self._Af, self._mean) + np.dot(self._Bf, self._control_signal_valued[0, :])
        return result

    def lookahead(self):
        """Return lookahead size :math:`K2`

        Returns
        -------
        int
            lookahead size.
        """
        return self.K2

class FIRFilter(IIRFilter):
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
    down_sample: `int`, `optional`
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
    down_sample: `int`, `optional`
        down sampling rate in relation to the control period :math:`T`.

    Yields
    ------
    `array_like`, shape=(L,)
        an input estimate sample :math:`\hat{\mathbf{u}}(t)`
    """

    def __init__(self, control_signal_sequence, analog_system, digital_control, double eta2, int K1, int K2, stop_after_number_of_iterations=(1 << 63), Ts=None, mid_point=False, down_sample=1):
        """Initializes filter coefficients
        """
        if (K1 < 0):
            raise "K1 must be non negative integer."
        self.K1 = K1
        if (K2 < 1):
            raise "K2 must be a positive integer."
        self.K2 = K2
        self._K3 = K1 + K2
        self.analog_system = analog_system
        if(eta2 < 0):
            raise "eta2 must be non negative."
        self.eta2 = eta2
        self.control_signal = control_signal_sequence
        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        
        self.mid_point = mid_point
        self.down_sample = down_sample

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
        Bf = np.zeros((self._N, self._M), dtype=np.double)
        Bb = np.zeros((self._N, self._M), dtype=np.double)
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
                    Bf[n, m] = solBf[n]
                    Bb[n, m] = solBb[n]
            Bf = np.dot(np.eye(self._N) + expm(tempAf * self.Ts / 2.0), Bf)
            Bb = np.dot(np.eye(self._N) + expm(tempAb * self.Ts / 2.0), Bb)

        else:
            for m in range(self._M):
                derivative = lambda t, x: np.dot(tempAf, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
                solBf = solve_ivp(derivative, (0, self.Ts), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
                derivative = lambda t, x: - np.dot(tempAb, x) + np.dot(Gamma, digital_control.impulse_response(m, t))
                solBb = -solve_ivp(derivative, (0, self.Ts), np.zeros(self._N), atol=atol, rtol=rtol, max_step=max_step).y[:,-1]
                for n in range (self._N):
                    Bf[n, m] = solBf[n]
                    Bb[n, m] = solBb[n]
        WT = solve(Vf + Vb, analog_system.B).transpose()

        # Initalize filter.
        self.h = np.zeros((self._K3, self._L, self._M), dtype=np.double)
        # Compute lookback.
        temp1 = np.copy(Bf)
        for k1 in range(self.K1 - 1, -1, -1):
            self.h[k1, :, :] = - np.dot(WT, temp1)
            temp1 = np.dot(Af, temp1)
        # Compute lookahead.
        temp2 = np.copy(Bb)
        for k2 in range(self.K1, self._K3):
            self.h[k2, :, :] = np.dot(WT, temp2)
            temp2 = np.dot(Ab, temp2)
        self._control_signal_valued = np.zeros((self._K3, self._M), dtype=np.int8)
        
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
            self._control_signal_valued[self._K3 - 1, m] = 2 * temp[m] - 1

        # self._control_signal_valued.shape -> (K1 + K2, M)
        # self.h.shape -> (K1 + K2, L, M)
        res = np.einsum('ijk,ik', self.h, self._control_signal_valued)
        # the Einstein summation results in:
        # result = np.zeros(self._L)
        # for l in range(self._L):
        #    for k in range(self.K1 + self.K2):
        #        for m in range(self._M):
        #            result[l] += self.h[k, l, m] * self._control_signal_valued[k, m]
        # return result
        
        # Check for down sampling
        if (((self._iteration - 1) % self.down_sample) == 0):
            return res
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

    
    
    def __str__(self):
        return f"FIR estimator is parameterized as \neta2 = {self.eta2:.2f}, {20 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},\nand\nnumber_of_iterations = {self.number_of_iterations}."
