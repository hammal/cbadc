from cbc.analog_system cimport AnalogSystem
from cbc.digital_control cimport DigitalControl
from cbc.analog_signal cimport AnalogSignal
import numpy as np
from scipy.integrate import solve_ivp
import math


cdef class StateSpaceSimulator(object):
    """Simulate the analog system and digital control interactions
    in the precense on analog signals.

    Parameters
    ----------
    analogSystem : :py:class:`cbc.analog_system.AnalogSystem`
        the analog system
    digitalControl: :py:class:`cbc.digital_control.DigitalControl`
        the digital control
    inputSignal : :py:class:`cbc.analog_signal.AnalogSignal`
        the analog signal (or a derived class)
    Ts : `float`, optional
        specify a sampling rate at which we want to evaluate the systems
        , defaults to digitalContro.Ts. Note that this Ts must be smaller 
        than digitalControl.Ts. 
    t_stop : `float`, optional
        determines a stop time, defaults to :py:`math.inf`



    Attributes
    ----------
    options : :obj:
        simulation settings

    See also
    --------
    :py:class:`cbc.analog_signal.AnalogSignal`
    :py:class:`cbc.analog_signal.Sinusodial`
    :py:class:`cbc.analog_system.AnalogSystem`
    :py:class:`cbc.digital_control.DigitalControl`

    Examples
    --------

    See also
    --------

    Yields
    ------
    `array_like`, shape=(M,), dtype=numpy.int8
    
    """
    cdef AnalogSystem _as
    cdef DigitalControl _dc
    cdef dict __dict__
    cdef double _t, _Ts, _t_stop
    cdef double [:] _state_vector
    cdef double [:] _input_vector, _control_vector, _temp_state_vector
    cdef double _atol, _rtol, _max_step
        
    cdef _ordinary_differentail_function(self, t, y):
        cdef int l,
        for l in range(self._as.L):
            self._input_vector[l] = self._is[l].evaluate(t)
        self._temp_state_vector = np.dot(self._as.Gamma_tildeT, y)
        self._control_vector = self._dc.evaluate(t, self._temp_state_vector)
        return np.asarray(self._as.derivative(self._temp_state_vector, t, self._input_vector,self. _control_vector)).flatten()

    def __init__(self, 
            AnalogSystem analogSystem, 
            DigitalControl digitalControl, 
            inputSignal, 
            Ts=None,
            t_stop=math.inf
        ):
        if analogSystem.L != len(inputSignal):
            raise """The analog system does not have as many inputs as in input
            list"""
        self._as = analogSystem
        self._dc = digitalControl
        self._is = inputSignal
        self._t_stop = t_stop
        if Ts:
            self._Ts = Ts
        else:
            self._Ts = self._dc.T
        if self._Ts > self._dc.T:
            raise f"Simulating with a sample period ${self._Ts} that exceeds the control period of the digital control ${self._dc.T}"
        self._state_vector = np.zeros(self._as.N, dtype=np.double)
        self._temp_state_vector = np.zeros(self._as.N, dtype=np.double)
        self._input_vector = np.zeros(self._as.L, dtype=np.double)
        self._control_vector = np.zeros(self._as.M, dtype=np.double)
        self._atol = 1e-6
        self._rtol = 1e-6
        self._max_step = self._Ts / 10.

    def __iter__(self):
        """Use simulator as an iterator
        """
        return self
    
    def __next__(self):
        """Computes the next control signal :math:`\mathbf{s}[k]`
        """
        cdef double t_end = self._t + self._Ts
        cdef double[2] t_span = (self._t, t_end)
        cdef double[1] t_eval = (t_end,)
        if t_end >= self._t_stop:
            raise StopIteration
        def f(t, x):
            return self._ordinary_differentail_function(t, x)
        sol = solve_ivp(f, t_span, self._state_vector, atol=self._atol, rtol=self._rtol, max_step=self._max_step)
        self._state_vector = sol.y[:,-1]
        self._t = t_end
        return self._dc.control_signal()

