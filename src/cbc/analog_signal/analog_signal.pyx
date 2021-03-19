import numpy as np
from libc.math cimport sin

cdef class AnalogSignal:
    """
    The analog signal
    """

    def __init__(self, offset = 0):
        """
        initalize the signal
        """
        self._offset = offset

    cpdef double evaluate(self, double t):
        """evaluate the signal at a given time

        :param doublet: [description]
        :type doublet: [type]
        :return: [description]
        :rtype: [type]
        """        
        return self._offset
    
    @property
    def offset(self):
        return self._offset

cdef class Sinusodial(AnalogSignal):
    """Sinusodial input signal this is class is 
    a helper to quickly generate sinusodial input signals.

    :param AnalogSignal: Inherits from AnalogSignal
    :type AnalogSignal: AnalogSignal
    """    
    def __init__(self, amplitude, frequency, phase=0, offset=0):
        self._amplitude = amplitude
        self._frequency = frequency
        self._angluarFrequency = 2 * 3.14159265358979323846 * self._frequency
        self._phase = phase
        self._offset = offset

    cpdef double evaluate(self, double t):
        return (
            self._amplitude * sin(self._angluarFrequency * t + self._phase)
            + self._offset
        )

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def frequency(self):
        return self._frequency

    @property
    def phase(self):
        return self._phase
