from libc.stdint cimport int8_t


cdef extern from "parallel_digital_estimator_filter.cpp":
    pass


# Declare the class with cdef
cdef extern from "parallel_digital_estimator_filter.h" namespace "CBC":
    cdef cppclass ParallelDigitalEstimator:
        ParallelDigitalEstimator(complex *, complex *, complex *, complex *, complex *, complex *, int, int, int, int, int) except +
        void compute_new_batch()
        int number_of_controls()
        int number_of_states()
        int number_of_inputs()
        int number_of_estimates_in_batch()
        bint empty_batch()
        int number_of_control_signals()
        bint full_batch()
        int batch_size()
        int lookahead()
        int size()
        void input(int *s)
        void output(double *estimate)