"""
The digital estimator.
"""
#cython: language_level=3
from cbc.digital_estimator.linear_filter cimport LinearFilter

class DigitalEstimator:
    # cdef int _estimate_pointer, _size, _iteration
    # cdef Filter _filter


    def __init__(self, controlSignalSequence, analogSystem, digitalControl, eta2, K1, K2 = 0, stop_after_number_of_iterations = None):
        # Check inputs
        if (K1 < 1):
            raise "K1 must be a positive integer"
        if (K2 < 0):
            raise "K2 must be a non negative integer"
        self._s = controlSignalSequence
        self._filter = LinearFilter(analogSystem, digitalControl, eta2, K1, K2 = K2)
        self._size = stop_after_number_of_iterations
        self._iteration = 0
        self._estimate_pointer = self._filter.batch_size()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if(self._size and self._size < self._iteration ):
            raise StopIteration
        self._iteration += 1
        if(self._estimate_pointer < self._filter.batch_size()):
            # print(self._estimate_pointer)
            self._estimate_pointer += 1
            return self._filter.output(self._estimate_pointer - 1)
        # Fill up batch with new control signals
        # s = iter(self._s)
        full = False
        # count = 0
        while (not full):
            # print(count)
            full = self._filter.input(next(self._s))
            # count += 1
        # compute new batch of K1 estimates
        self._filter.compute_batch()
        # print("Called batch compute")
        self._estimate_pointer -= self._filter.batch_size()
        # print("new estimate pointer, ", self._estimate_pointer)
        return self.__next__()

    