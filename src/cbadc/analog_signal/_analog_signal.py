"""
"""
import logging
from typing import Union
from sympy import Symbol
from mpmath import mp

logger = logging.getLogger(__name__)


class _AnalogSignal:
    """A default continuous-time analog signal."""

    def __init__(self):
        self.t = Symbol('t', real=True)
        self.sym_phase = Symbol('\u03C6', real=True)
        self.t0 = 0.0

    def symbolic(self) -> Symbol:
        """Returns as symbolic exression

        Returns
        -------
        : :py:class:`sympy.Symbol`
            the resulting function
        """
        raise NotImplementedError

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

    def _mpmath(self, t: Union[mp.mpf, float]):
        return mp.mpmathify(self.evaluate(t))

    def __call__(self, t: float):
        return self.evaluate(t)

    def __str__(self):
        return "Analog signal returns constant 0, i.e., maps t |-> 0."
