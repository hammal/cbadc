
cdef class AnalogSystem:
    cdef double [:,:] _A
    cdef double [:,:] _B
    cdef double [:,:] _C
    cdef double [:,:] _Gamma
    cdef double [:,:] _Gamma_tilde
    cdef double [:] temp_derivative
    cdef double [:] temp_y
    cdef double [:] temp_s_tilde
    cdef int _M, _N, _L, _N_tilde, _M_tilde
    cpdef double [:] derivative(self, double [:] x, double t, double [:] u, double [:] s)
    cpdef double [:] signal_output(self, double [:] x)
    cpdef double [:] control_output(self, double [:] x)
    cdef complex [:,:] _atf(self, double _omega)
