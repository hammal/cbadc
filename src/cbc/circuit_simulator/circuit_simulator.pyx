from cbc.analog_system.analog_system cimport AnalogSystem
from cbc.digital_control.digital_control cimport DigitalControl
from cbc.analog_signal.analog_signal cimport AnalogSignal
import numpy as np
from scipy.integrate import solve_ivp
import math


cdef class CircuitSimulator(object):
    cdef AnalogSystem _as
    cdef DigitalControl _dc
    cdef dict __dict__
    cdef double _t, _Ts, _t_stop
    cdef double [:] _state_vector
    cdef double [:] _input_vector, _control_vector, _temp_state_vector
        
    cdef _ordinary_differentail_function(self, t, y):
        cdef int l,
        for l in range(self._as._L):
            self._input_vector[l] = self._is[l].evaluate(t)
        self._temp_state_vector = y
        self._control_vector = self._dc.evaluate(t, self._temp_state_vector)
        return np.asarray(self._as.derivative(self._temp_state_vector, t, self._input_vector,self. _control_vector)).flatten()

    def __init__(self, 
            AnalogSystem analogSystem, 
            DigitalControl digitalControl, 
            inputSignal, 
            Ts=None,
            t_stop=math.inf
        ):
        if analogSystem._L != len(inputSignal):
            raise """The analog system does not have as many inputs as in input
            list"""
        self._as = analogSystem
        self._dc = digitalControl
        self._is = inputSignal
        self._t_stop = t_stop
        if Ts:
            self._Ts = Ts
        else:
            self._Ts = self._dc._Ts
        if self._Ts > self._dc._Ts:
            raise f"Simulating with a sample period ${self._Ts} that exceeds the control period of the digital control ${self._dc._Ts}"
        self._state_vector = np.zeros(self._as._N, dtype=np.double)
        self._temp_state_vector = np.zeros(self._as._N, dtype=np.double)
        self._input_vector = np.zeros(self._as._L, dtype=np.double)
        self._control_vector = np.zeros(self._as._M, dtype=np.double)

    def __iter__(self):
        return self
    
    def __next__(self):
        cdef double t_end = self._t + self._Ts
        cdef double[2] t_span = (self._t, t_end)
        cdef double[1] t_eval = (t_end,)
        if t_end >= self._t_stop:
            raise StopIteration
        def f(t, x):
            return self._ordinary_differentail_function(t, x)
        sol = solve_ivp(f, t_span, self._state_vector, t_eval=t_eval, dense_output=False)
        self._state_vector = sol.y[:,0]
        self._t = t_end
        return self._dc.control_signal()

