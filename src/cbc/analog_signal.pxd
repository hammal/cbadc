cdef class AnalogSignal:
    cdef double _offset
    cpdef double evaluate(self, double t)

cdef class Sinusodial(AnalogSignal):
    cdef double _amplitude
    cdef double _frequency 
    cdef double _angluarFrequency
    cdef double _phase 
    