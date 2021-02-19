from cbc.digital_control.digital_control cimport DigitalControl
from cbc.analog_system.analog_system cimport AnalogSystem
from cbc.parallel_digital_estimator.parallel_estimator cimport ParallelDigitalEstimator

cdef class C_Digital_Estimator:
    cdef double [:,:] _Af
    cdef double [:,:] _Ab
    cdef double [:,:] _Bf
    cdef double [:,:] _Bb
    cdef double [:,:] _WT
    cdef double complex [:] forward_a, 
    cdef double complex [:] backward_a,  
    cdef double complex [:] forward_b, 
    cdef double complex [:] backward_b, 
    cdef double complex [:] forward_w, 
    cdef double complex [:] backward_w, 
    cdef int _K1, _K2, _K3
    cdef int _N, _M, _L
    cdef double [:] estimate
    cdef ParallelDigitalEstimator*_filter
    #// cpdef void compute_batch(self)
    #// cpdef int batch_size(self)
    #// cpdef int lookahead(self)
    #// cpdef int size(self)
    #// cdef void allocate_memory(self)
    cdef void compute_filter_coefficients(self, AnalogSystem analogSystem, DigitalControl digitalControl, double eta2)

