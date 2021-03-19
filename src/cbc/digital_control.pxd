cdef class DigitalControl:
    cdef double _t_next
    cpdef double T
    cdef double [:] _dac_values
    cdef char [:] _s
    cdef int M, M_tilde
    cpdef double [:] evaluate(self, double t, double [:] x)
    cpdef char [:] control_signal(self)
    cdef double [:] impulse_response(self, m, t)
