"""This module contains various analog signal classes

Specifically, we represent analog signals, i.e., mappings from the time :math:`t`
to a signal value :math:`u(t)`. Typically we are used to handeling discrete time
signals for various signal processing tasks. However, since the control-bounded A/D
converters are converting continuous-time signals we need to define signals that 
can be evaluated over their whole continuous domain.
"""

import numpy as np
from libc.math cimport sin

cdef class AnalogSignal:
    """A constant continuous-time analog signal.
    
    Parameters
    -----------
    offset : `float`, `optional`
         Determines the offset or DC bias of the analog signal, defaults to 0.


    Attributes
    ----------
    offset : `float`
        The offset value.


    See also
    ---------
    :py:class:`cbc.analog_signal.Sinusodial`
    :py:class:`cbc.state_space_simulator.StateSpaceSimulator`

    Examples
    ---------
    >>> import cbc
    >>> u = cbc.AnalogSignal(1.0) 
    >>> print(u.evaluate(42))
    1.0

    """

    def __init__(self, offset = 0.0):
        """Create a constant analog signal.
        """
        self.offset = offset

    cpdef double evaluate(self, double t):
        """Evaluate the signal at time :math:`t`.

        Parameters
        ----------
        t : `float`
            the time instance for evaluation.

        Returns
        -------
        float
            The analog signal value
        """        
        return self.offset

cdef class Sinusodial(AnalogSignal):
    """An analog continuous-time sinusodial signal.

    Parameters
    ----------
    amplitude : `float`
        The amplitude of the sinusodial.
    frequency : `float`
        Frequency of the oscillation in [Hz].
    phase : `float`, optional 
        The phase, defaults to 0.
    offset : `float`
        The offset value.

    Attributes
    ----------
    amplitude : `float`
        The amplitude.
    frequency : `float`
        The frequency in [Hz].
    angularFrequency : `float`
        The frequency in [radians/second].
    phase : `float`
        The phase.
    offset : `float`, `optional`
        The offset

    See also
    --------
    cbc.analog_signal.AnalogSignal

    Example
    -------
    >>> import cbc
    >>> import numpy as np
    >>> u = cbc.Sinudodial(1, 123, np.pi/2, 0)
    >>> u(0)
    0
    >>> u(1/123.)
    1

    """    
    def __init__(self, amplitude, frequency, phase=0.0, offset=0.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.angluarFrequency = 2 * np.pi * self.frequency
        self.phase = phase
        self.offset = offset

    cpdef double evaluate(self, double t):
        """Evaluate the signal at time :math:`t`.

        Parameters
        ----------
        t : `float`
            the time instance for evaluation.

        Returns
        -------
        float
            The analog signal value
        """
        return (
            self.amplitude * sin(self.angluarFrequency * t + self.phase)
            + self.offset
        )