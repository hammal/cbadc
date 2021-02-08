"""
The digital estimator.
"""
#cython: language_level=3
# from cbc.digital_estimator.filter cimport Filter

class DigitalEstimator:

    # cdef int _estimate_pointer
    # cdef Filter _filter

    def __init__(self, filter, controlSignalSequence):
        self._s = controlSignalSequence
        self._filter = filter

    def __iter__(self):
        return self
    
    def __next__(self):
        if(self._estimate_pointer < self._filter.batch_size() - 1):
            self._estimate_pointer += 1
            return self._filter.output(self._estimate_pointer)

        # Fill up batch with new control signals
        s = iter(self._s)
        full = False
        while (not full):
            full = self._filter.input(next(s))
        # compute new batch of K1 estimates
        self._filter.compute_batch()
        self._estimate_pointer -= self._filter.batch_size()
        return self.__next__()

    