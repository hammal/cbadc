from ._analog_signal import _AnalogSignal
from sympy import Piecewise
import numpy as np


class Clock(_AnalogSignal):
    """An analog computer clock signal.

    Specifically, based on a clock period T a
    clock represents an analog clock edge signal.

    Parameters
    ----------
    T: `float`
        the clock period
    tt: `float`
        transition time from one clock edge to the other, defaults
        to 1ps.
    td: `float`
        a global delay for the clock transition, defaults to 0.
    duty_cycle: `float`
        a number (0, 1] representing how long the signal
        is high (1) relatively low (-1), defaults to 0.5.
    max_swing: `float`
        largest permissible swing of the clock signal

    Attributes
    ----------
    T: `float`
        the clock period.
    tt: `float`
        the transition time.
    td: `float`
        any global time delay.
    duty_cycle: `float`
        the duty cycle
    max_value: `float`
        the largest value of the clock signal. Similarly,
        the smallest value is -max_value.
    """

    def __init__(
        self,
        T: float,
        tt: float = 1e-12,
        td: float = 0.0,
        duty_cycle: float = 0.5,
        max_swing: float = 2.0,
    ) -> None:
        super().__init__()
        self.T = T
        self.tt = tt
        self._tt_2 = self.tt / 2.0
        self.td = td
        self.duty_cycle = duty_cycle
        self._neq_pulse_time = self.duty_cycle * self.T
        self.max_value = max_swing / 2.0
        if duty_cycle > 1.0 or duty_cycle <= 0.0:
            raise Exception(
                f"duty_cycle must be a number between 0 and up to 1. Not {duty_cycle}"
            )
        if tt > self.max_step():
            raise Exception(
                "transition time tt can't be longer than smallest clock period"
            )
        if td > T:
            raise Exception(
                "Does not make sense to have longer global delay than time period."
            )

    def max_step(self):
        return self.T * min(self.duty_cycle, 1 - self.duty_cycle)

    def _pos_edge(self, t):
        tmp = (t - self._tt_2) / self._tt_2
        if tmp > self.max_value:
            return tmp / tmp
        elif tmp < -self.max_value:
            return -tmp / tmp
        return tmp

    def clock_edge(self, t):
        t_ = (t - self.td + self._tt_2) % self.T
        print(t_)
        if t_ > self._neq_pulse_time:
            return -self._pos_edge(t_ - self._neq_pulse_time)
        else:
            return self._pos_edge(t_)

    def next_tick(self, t):
        """Return time of next positive
        clock edge relative to time t

        Parameters
        ----------
        t: `float`
            current time

        Returns
        -------
        : `float`
            the time of next positive clock edge.
        """
        t_ = (t - self.td) % self.T
        if np.allclose(t_, 0):
            return t
        return t + self.T - t_

    def symbolic(self) -> Piecewise:
        """Returns as symbolic exression

        Returns
        -------
        : :py:class:`sympy.Symbol`
            the resulting function
        """
        t_ = (self.t - self.td + self._tt_2) % self.T
        return Piecewise(
            (-self._pos_edge(t_), t_ > self.duty_cycle * self.T),
            (self._pos_edge(t_), True),
        )

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
        return self.clock_edge(t)


def delay_clock_by_duty_cycle(clock: Clock, delay: float = 0):
    """Create a delayed clock version of
    an already existing clock singal.

    Parameters
    ----------
    clock: :py:class:`cbadc.analog_signal.Clock`
        the clock from which the new clock is derived
    delay: `float`
        the amount of delay, defaults to the duty cycle
        which can be interpreted as if the positive edge
        of the new clock overlaps with the negative edge
        of the former.

    """
    if not delay:
        phase_delay_to_readout = clock.duty_cycle * clock.T
    else:
        phase_delay_to_readout = delay
    return Clock(
        T=clock.T,
        tt=clock.tt,
        td=clock.td + phase_delay_to_readout,
        max_swing=clock.max_value * 2,
    )
