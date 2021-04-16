from cbadc.digital_control cimport DigitalControl
from cbadc.analog_system cimport AnalogSystem
from cbadc.parallel_digital_estimator cimport ParallelDigitalEstimator

cdef class ParallelFilter:
    cdef double [:,:] _Af
    cdef double [:,:] _Ab
    cdef double [:,:] _Bf
    cdef double [:,:] _Bb
    cdef double [:,:] _WT
    cdef double [:,:] _estimates
    cdef double _Ts
    cdef double complex [:] forward_a, 
    cdef double complex [:] backward_a,  
    cdef double complex [:] forward_b, 
    cdef double complex [:] backward_b, 
    cdef double complex [:] forward_w, 
    cdef double complex [:] backward_w, 
    cdef int _K1, _K2, _K3, _control_signal_in_buffer
    cdef int _N, _M, _L
    cdef ParallelDigitalEstimator*_filter
    cpdef void compute_batch(self)
    cpdef int batch_size(self)
    cpdef int lookahead(self)
    cpdef int size(self)
    cdef void compute_filter_coefficients(self, AnalogSystem analog_system, DigitalControl digital_control, double eta2)
