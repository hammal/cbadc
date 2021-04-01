cdef class DigitalControl:
    cdef double _t_next
    cdef readonly double T
    cdef double [:] _dac_values
    cdef char [:] _s
    cdef int M, M_tilde
    cpdef double [:] control_contribution(self, double t, double [:] x)
    cpdef char [:] control_signal(self)
