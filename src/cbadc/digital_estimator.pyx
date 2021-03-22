"""
The digital estimator.
"""
#cython: language_level=3
from cbadc.linear_filter cimport LinearFilter
from cbadc.filter_mid_point cimport MidPointFilter
from numpy import dot, zeros, eye, array, abs
from numpy.linalg import inv


class DigitalEstimator:
    """The digital estimator estimates a filtered version :math:`\hat{\mathbf{u}}(t)` (shaped by 
    :py:func:`signal_transfer_function`) of the 
    input signal :math:`\mathbf{u}(t)` from a sequence of control signals :math:`\mathbf{s}[k]`.

    Parameters
    ----------
    controlSignalSequence : iterator
        a generator which outputs a sequence of control signals.
    analogSystem : :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system (necessary to compute the estimators filter coefficients).
    digitalControl : :py:class:`cbadc.digital_control.DigitalControl`
        a digital control (necessary to determine the corresponding DAC waveform).
    eta2 : `float`
        the :math:`\eta^2` parameter determines the bandwidth of the estimator.
    K1 : `int`
        batch size.
    K2 : `int`, `optional`
        lookahead size, defaults to 0.
    stop_after_number_of_iterations : `int`
        determine a max number of iterations by the generator, defaults to :py:obj:`math.inf`.
    estimator_type : `str`, `optional`
        determine which filtering method :code:`'quadratic', 'parallel', 'mid-point'` 
        to use, defaults to :code:`'quadratic'`.

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

    Yields
    ------
    `array_like`, shape=(L,)
        an input estimate sample :math:`\hat{\mathbf{u}}(t)`

    Examples
    --------
    TODO:
    """


    def __init__(self, 
        controlSignalSequence, 
        analogSystem, digitalControl, 
        eta2, 
        K1, 
        K2 = 0, 
        stop_after_number_of_iterations = 0, 
        estimator_type = 'quadratic'):
        # Check inputs
        if (K1 < 1):
            raise "K1 must be a positive integer."
        self.K1 = K1
        if (K2 < 0):
            raise "K2 must be a non negative integer."
        self.K2 = K2
        self.analog_system = analogSystem
        if(eta2 < 0):
            raise "eta2 must be non negative."
        self.eta2 = eta2
        self.control_signal = controlSignalSequence

        estimation_filter_implementations = {
            'quadratic': lambda : LinearFilter(analogSystem, digitalControl, eta2, K1, K2 = K2),
            'mid-point': lambda : MidPointFilter(analogSystem, digitalControl, eta2, K1, K2 = K2)
        }
        def not_implemented():
            raise f"{estimator_type} is not a implemented estimator algorithm, currently choose from {estimation_filter_implementations.keys()}"
        self._filter = estimation_filter_implementations.get(estimator_type, not_implemented)()
        self.estimator_type = estimator_type

        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        self._estimate_pointer = self._filter.batch_size()
        
        # For transfer functions
        self.eta2Matrix = eye(self.analog_system.CT.shape[0]) * self.eta2
    
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
        result = zeros((self.analog_system.L, self.analog_system.N, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function(array([o]))
            G = G.reshape((self.analog_system.N, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = dot(G, GH)
            result[:, :, index] = abs(dot(GH, dot(inv(GGH + self.eta2Matrix))))
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
        result = zeros((self.analog_system.L, omega.size))
        for index, o in enumerate(omega):
            G = self.analog_system.transfer_function(array([o]))
            G = G.reshape((self.analog_system.N_tilde, self.analog_system.L))
            GH = G.transpose().conjugate()
            GGH = dot(G, GH)
            result[:, index] = abs(dot(GH, dot(inv(GGH + self.eta2Matrix), G)))
        return result


    def __iter__(self):
        return self
    
    def __next__(self):
        # Check if the end of prespecified size
        if(self.number_of_iterations and self.number_of_iterations < self._iteration ):
            raise StopIteration
        self._iteration += 1

        # Check if there are estimates in the estimate buffer
        if(self._estimate_pointer < self._filter.batch_size()):
            self._estimate_pointer += 1
            return self._filter.output(self._estimate_pointer - 1)

        # If not.
        full = False

        # Fill up batch with new control signals.
        while (not full):
            # next(self.control_signal) calls the control signal
            # generator and thus recives new control 
            # signal samples
            full = self._filter.input(next(self.control_signal))

        # Compute new batch of K1 estimates
        self._filter.compute_batch()

        # adjust pointer to indicate that estimate buffer
        # is non empty
        self._estimate_pointer -= self._filter.batch_size()
        
        # recursively call itself to return new estimate
        return self.__next__()

    