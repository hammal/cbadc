"""Analog constant signals."""

import numpy as np
from ._analog_signal import _AnalogSignal
from sympy import Float


class ConstantSignal(_AnalogSignal):
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
    :py:class:`cbadc.analog_signal.Sinusoidal`
    :py:class:`cbadc.simulator.StateSpaceSimulator`

    Examples
    ---------
    >>> from cbadc.analog_signal import ConstantSignal
    >>> u = ConstantSignal(1.0)
    >>> print(u.evaluate(42))
    1.0

    """

    def __init__(self, offset: float = 0.0):
        """Create a constant analog signal."""
        super().__init__()
        self.offset: float = offset
        self.piecewise_constant = True

    def symbolic(self) -> Float:
        """Returns as symbolic exression

        Returns
        -------
        : :py:class:`sympy.Float`
            a constant value c
        """
        return Float(self.offset)

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
