"""Predefined common analog signals

This module focuses on representing analog signals, i.e., mappings from the
time :math:`t` to a signal value :math:`u(t)`. Typically for signal processing
algorithms, we are used to handeling discrete-time signals, i.e. samples of
signals. However, since the control-bounded A/D converters are converting
continuous-time signals we need tools to define signals that can be evaluated
over their whole continuous domain.
"""
import numpy as np


class AnalogSignal:
    """A default continuous-time analog signal.
    """

    def __init__(self):
        pass

    def evaluate(self, t: float) -> float:
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
        return 0.0

    def __str__(self):
        return "Analog signal returns constant 0, i.e., maps t |-> 0."


class ConstantSignal(AnalogSignal):
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
    :py:class:`cbadc.analog_signal.Sinusodial`
    :py:class:`cbadc.simulator.StateSpaceSimulator`

    Examples
    ---------
    >>> from cbadc.analog_signal import ConstantSignal
    >>> u = ConstantSignal(1.0)
    >>> print(u.evaluate(42))
    1.0

    """

    def __init__(self, offset: float = 0.0):
        """Create a constant analog signal.
        """
        self.offset: float = offset

    def evaluate(self, t: float) -> float:
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

    def __str__(self):
        return f"ConstantSignal has an offset = {self.offset}."


class Sinusodial(AnalogSignal):
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
    cbadc.analog_signal.AnalogSignal

    Example
    -------
    >>> from cbadc.analog_signal import Sinusodial
    >>> import numpy as np
    >>> u = Sinusodial(1, 123, np.pi/2, 0)
    >>> print(u.evaluate(0))
    1.0

    """

    def __init__(self, amplitude: float, frequency: float, phase: float = 0.0,
                 offset: float = 0.0):
        self.amplitude: float = amplitude
        self.frequency: float = frequency
        self.angluarFrequency: float = 2 * np.pi * self.frequency
        self.phase: float = phase
        self.offset: float = offset

    def __str__(self):
        return f"""Sinusodial parameterized as: \namplitude = {self.amplitude}, \n
        frequency = {self.frequency}, \nphase = {self.phase},
        and\noffset = {self.offset}"""

    def evaluate(self, t: float) -> float:
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
            self.amplitude *
            np.sin(self.angluarFrequency * t + self.phase) + self.offset
        )
