
cdef class AnalogSystem:
    cdef readonly double [:,:] A
    cdef readonly double [:,:] B    
    cdef readonly double [:,:] Gamma   
    cdef readonly double [:,:] CT   
    cdef readonly double [:,:] Gamma_tildeT   
    cdef double [:] temp_derivative
    cdef double [:] temp_y
    cdef double [:] temp_s_tilde
    cdef readonly int M  
    cdef readonly int N   
    cdef readonly int L   
    cdef readonly int N_tilde    
    cdef readonly int M_tilde  
    cpdef double [:] derivative(self, double [:] x, double t, double [:] u, double [:] s)
    cpdef double [:] signal_observation(self, double [:] x)
    cpdef double [:] control_observation(self, double [:] x)
    cdef complex [:,:] _atf(self, double _omega)
