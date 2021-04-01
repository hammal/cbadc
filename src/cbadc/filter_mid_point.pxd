from cbadc.digital_control cimport DigitalControl
from cbadc.analog_system cimport AnalogSystem
cdef class MidPointFilter:
    cdef double [:,:] _Af
    cdef double [:,:] _Ab
    cdef double [:,:] _Bf
    cdef double [:,:] _Bb
    cdef double [:,:] _WT
    cdef char [:,:]  _control_signal
    cdef double [:,:] _estimate
    cdef double Ts
    cdef double [:] temp_forward_mean
    cdef double [:] temp_backward_mean
    cdef int _K1, _K2, _K3, _control_signal_in_buffer
    cdef int _N, _M, _L
    # cdef int input(self, char [:] s)
    # cpdef double [:] output(self, int index)
    cpdef void compute_batch(self)
    cpdef int batch_size(self)
    cpdef int lookahead(self)
    cpdef int size(self)
    cdef void allocate_memory_buffers(self)
    cdef void compute_filter_coefficients(self, AnalogSystem analogSystem, DigitalControl digitalControl, double eta2)
