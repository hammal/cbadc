cdef class Filter:
    cdef double [:,:] _Af
    cdef double [:,:] _Ab
    cdef double [:,:] _Bf
    cdef double [:,:] _Bb
    cdef double [:,:] _WT
    cdef char [:,:]  _control_signal
    cdef double [:,:] _mean
    cdef double [:,:] _estimate
    cdef int _K1, _K2, _K3, _control_signal_in_buffer
    cdef int input(self, char [:] s)
    cdef double [:] output(self, int index)
    cdef void compute_batch(self)
    cpdef int batch_size(self)
    cpdef int lookahead(self)
    cpdef int size(self)
