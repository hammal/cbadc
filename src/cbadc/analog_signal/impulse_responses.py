"""Analog impulse response signals from common linear systems."""

from typing import Union
import numpy as np
import sympy as sp
import mpmath as mp
from ._analog_signal import _AnalogSignal


class _ImpulseResponse(_AnalogSignal):
    def __init__(self):
        super().__init__()
        self.t0 = 0.0
        self.piecewise_constant = True


class StepResponse(_ImpulseResponse):
    def __init__(self, t0: float = 0.0, amplitude=1.0) -> None:
        super().__init__()
        self.t0 = t0
        self.amplitude = amplitude
        self._t0_mpmath = mp.mpmathify(t0)
        self._amplitude_mpmath = mp.mpmathify(amplitude)

    def evaluate(self, t: float) -> float:
        """Returns the step response function

        :math:`x(t) = \\begin{cases} 1 & \\mathrm{if} \\,\\, t > 0  \\\ 0 & \\mathrm{otherwise} \\end{cases}`

        Parameters
        ----------
        t: `float`
            evaluated at time
        t0: `float`, `optional`
            starting time
        amplitude: `float`, `optional`
            the amplitude of the impulse response

        Returns
        -------
        `float`:
            the step response evaluated at t.
        """
        return self.amplitude if t >= self.t0 else 0.0

    def _mpmath(self, t: Union[mp.mpf, float]):
        return mp.mpf(self._amplitude_mpmath)

    def symbolic(self):
        """Returns as symbolic exression

        Returns
        -------
        : :py:class:`sympy.Symbol`
            the resulting function
        """
        # return sp.Piecewise((0, self.t < self.t0), (1, True))
        return sp.Float(self.amplitude)


class NonReturnToZeroDAC(StepResponse):
    def __init__(self, t_start: float, t_end: float, amplitude=1.0):
        super().__init__(t0=t_start, amplitude=amplitude)
        self.t_end = t_end

    def evaluate(self, t: float) -> float:
        """Returns the step response function

        :math:`x(t) = \\begin{cases} 1 & \\mathrm{if} \\,\\, t > 0  \\\ 0 & \\mathrm{otherwise} \\end{cases}`

        Parameters
        ----------
        t: `float`
            evaluated at time
        t0: `float`, `optional`
            starting time
        amplitude: `float`, `optional`
            the amplitude of the impulse response

        Returns
        -------
        `float`:
            the step response evaluated at t.
        """
        return super().evaluate(t) if t < self.t_end else 0.0


class RCImpulseResponse(_ImpulseResponse):
    def __init__(self, tau: float, t0: float = 0, amplitude=1.0):
        super().__init__()
        self.tau = tau
        self.t0 = t0
        self.amplitude = amplitude
        self._tau_mpmath = mp.mpmathify(tau)
        self._t0_mpmath = mp.mpmathify(t0)
        self._amplitude_mpmath = mp.mpmathify(amplitude)

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
        amplitude: `float`, `optional`
            the amplitude of the impulse response

        Returns
        --------
        `float`:
            the impulse response evaluated at t.
        """
        if t < self.t0:
            return 0.0
        else:
            return self.amplitude * np.exp((self.t0 - t) / self.tau)

    def _mpmath(self, t: Union[mp.mpf, float]):
        t = mp.mpmathify(t)
        return self._amplitude_mpmath * mp.exp((self._t0_mpmath - t) / self._tau_mpmath)

    def symbolic(self):
        """Returns as symbolic exression

        Returns
        -------
        : :py:class:`sympy.Symbol`
            the resulting function
        """
        return sp.Float(self.amplitude) * sp.exp((self.t0 - self.t) / self.tau)
        # return sp.Piecewise(
        #     (0, self.t < self.t0),
        #     (sp.exp((self.t0 - self.t) / self.tau), True)
        # )
