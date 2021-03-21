cdef class AnalogSignal:
    cdef readonly double offset
    cpdef double evaluate(self, double t)

cdef class Sinusodial(AnalogSignal):
    cdef readonly double amplitude
    cdef readonly double frequency 
    cdef readonly double angluarFrequency
    cdef readonly double phase  
    