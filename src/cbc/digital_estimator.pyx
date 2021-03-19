"""
The digital estimator.
"""
#cython: language_level=3
from cbc.linear_filter cimport LinearFilter
from cbc.filter_mid_point cimport MidPointFilter
from numpy import dot, zeros, eye, array, abs
from numpy.linalg import inv

class DigitalEstimator:
    """The digital estimator estimates a input signal :math:`\mathbf{u}(t)` from a sequence of control
    signals :math:`\mathbf{s}[k]`.

    Parameters
    ----------
    controlSignalSequence : iterator
        a generator which outputs a sequence of control signals.
    analogSystem : :py:class:`cbc.analog_system.AnalogSystem`
        an analog system (necessary to compute the estimators filter coefficients).
    digitalControl : :py:class:`cbc.digital_control.DigitalControl`
        a digital control (necessary to determine the corresponding DAC waveform).
    eta2 : `float`
        the :math:`\eta^2` parameter determines the bandwidth of the estimator.
    K1 : `int`
        batch size.
    K2 : `int`, `optional`
        lookahead size, defaults to 0.
    stop_after_number_of_iterations : `int`
        determine a max number of iterations by the generator, defaults to :py:`math.inf`.
    midPoint : `bool`
        determine which filtering method to use, defaults to False.
    
    Yields
    ------
    `array_like`, shape=(L,)
        an input estimate sample

    Examples
    --------
    TODO:
    """


    def __init__(self, controlSignalSequence, analogSystem, digitalControl, eta2, K1, K2 = 0, stop_after_number_of_iterations = 0, midPoint = False):
        # Check inputs
        if (K1 < 1):
            raise "K1 must be a positive integer"
        if (K2 < 0):
            raise "K2 must be a non negative integer"
        self._analog_system = analogSystem
        self._eta2 = eta2
        self._s = controlSignalSequence
        if midPoint:
            self._filter = MidPointFilter(analogSystem, digitalControl, eta2, K1, K2 = K2)
        else:
            self._filter = LinearFilter(analogSystem, digitalControl, eta2, K1, K2 = K2)
        self._size = stop_after_number_of_iterations
        self._iteration = 0
        self._estimate_pointer = self._filter.batch_size()
    
    def transfer_function(self, double [:] omega):
        """Compute tranfer function for estimator.

        Parameters
        ----------
        omega : `array_like`, shape=(:,)

        Returns
        -------
        `array_like`, shape=(L, K)
            return transfer function evaluated at all K angular frequencies of
            the omega vector.
        """
        result = zeros((self._analog_system.L(), omega.size))
        eta2Matrix = eye(self._analog_system.C().shape[0]) * self._eta2
        for index, o in enumerate(omega):
            G = self._analog_system.transfer_function(array([o]))
            G = G.reshape((self._analog_system.N_tilde(), self._analog_system.L()))
            GH = G.transpose().conjugate()
            GGH = dot(G, GH)
            result[:, index] = abs(dot(GH, dot(inv(GGH + eta2Matrix), G)))
            # print(f'G {G}, GGH {GGH}, result: {result[:,index]}')
        return result


    def __iter__(self):
        return self
    
    def __next__(self):
        if(self._size and self._size < self._iteration ):
            raise StopIteration
        self._iteration += 1
        if(self._estimate_pointer < self._filter.batch_size()):
            # print(self._estimate_pointer)
            self._estimate_pointer += 1
            return self._filter.output(self._estimate_pointer - 1)
        # Fill up batch with new control signals
        # s = iter(self._s)
        full = False
        # count = 0
        while (not full):
            # print(count)
            full = self._filter.input(next(self._s))
            # count += 1
        # compute new batch of K1 estimates
        self._filter.compute_batch()
        # print("Called batch compute")
        self._estimate_pointer -= self._filter.batch_size()
        # print("new estimate pointer, ", self._estimate_pointer)
        return self.__next__()

    