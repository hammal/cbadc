cdef class DigitalControl:
    cdef double _t_next, _Ts
    cdef double [:] _dac_values
    cdef char [:] _s
    cdef int _order

    cpdef double [:] evaluate(self, double t, double [:] x)
    cpdef char [:] control_signal(self)
    cpdef double Ts(self)
    cpdef double [:] impulse_response(self, m, t)
