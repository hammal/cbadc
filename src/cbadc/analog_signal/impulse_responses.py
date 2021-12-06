from typing import Union
import numpy as np
import sympy as sp
import mpmath as mp
from ._analog_signal import _AnalogSignal


class _ImpulseResponse(_AnalogSignal):
    def __init__(self):
        super().__init__()
        self.t0 = 0.0


class StepResponse(_ImpulseResponse):
    def __init__(self, t0: float = 0) -> None:
        super().__init__()
        self.t0 = t0

    def evaluate(self, t: float) -> float:
        """Returns the step response function

        :math:`x(t) = \\begin{cases} 1 & \\mathrm{if} \\,\\, t > 0  \\\ 0 & \\mathrm{otherwise} \\end{cases}`

        Parameters
        ----------
        t: `float`
            evaluated at time
        t0: `float`, `optional`
            starting time

        Returns
        -------
        `float`:
            the step response evaluated at t.
        """
        return 1.0 if t >= self.t0 else 0.0

    def _mpmath(self, t: Union[mp.mpf, float]):
        return mp.mpf('1.0')

    def symbolic(self):
        """Returns as symbolic exression

        Returns
        -------
        : :py:class:`sympy.Symbol`
            the resulting function
        """
        # return sp.Piecewise((0, self.t < self.t0), (1, True))
        return sp.Integer(1)


class RCImpulseResponse(_ImpulseResponse):
    def __init__(self, tau: float, t0: float = 0):
        super().__init__()
        self.tau = tau
        self.t0 = t0
        self._tau_mpmath = mp.mpmathify(tau)
        self._t0_mpmath = mp.mpmathify(t0)

    def evaluate(self, t: float) -> float:
        """Returns the impulse response of a RC circuit.

        Specifically, solves

        :math:`\\dot{x}(t) = -\\frac{1}{\\tau} x(t)`

        :math:`x(0) = 1`

        at time :math:`t`.

        Parameters
        ----------
        t: `float`
            Evaluation at time t.
        tau: `float`
            time constant.
        t0: `float`, `optional`
            starting time

        Returns
        --------
        `float`:
            the impulse response evaluated at t.
        """
        if t < self.t0:
            return 0.0
        else:
            return np.exp((self.t0 - t) / self.tau)

    def _mpmath(self, t: Union[mp.mpf, float]):
        t = mp.mpmathify(t)
        return mp.exp((self._t0_mpmath - t) / self._tau_mpmath)

    def symbolic(self):
        """Returns as symbolic exression

        Returns
        -------
        : :py:class:`sympy.Symbol`
            the resulting function
        """
        return sp.exp((self.t0 - self.t) / self.tau)
        # return sp.Piecewise(
        #     (0, self.t < self.t0),
        #     (sp.exp((self.t0 - self.t) / self.tau), True)
        # )
