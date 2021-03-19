"""
The digital control is responsible for stabilizing the analog system.
"""
#cython: language_level=3
import numpy as np

cdef class DigitalControl:

    def __init__(self, Ts, M, t0 = 0.):
        self._Ts = Ts
        self._order = M
        self._t_next = t0 + self._Ts
        self._s = np.zeros(self._order, dtype=np.int8)
        self._dac_values = np.zeros(self._order, dtype=np.double)

    cpdef double [:] evaluate(self, double t, double [:] x):
        # sample and digitize
        cdef int m
        if t >= self._t_next:
            for m in range(self._order):
                self._s[m] = x[m] > 0
                if self._s[m]:
                    self._dac_values[m] = 1 
                else: 
                    self._dac_values[m] = -1
            self._t_next += self._Ts
        # DAC
        return self._dac_values
        
    cpdef char [:] control_signal(self):
        return self._s

    cpdef double Ts(self):
        return self._Ts

    cdef double [:] impulse_response(self, m, t):
        temp = np.zeros(self._order, dtype=np.double)
        temp[m] = 1
        return temp
