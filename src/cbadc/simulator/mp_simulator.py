"""This module implements analytical solvers for
simulating the analog_system digital control interactions.
"""

import logging
import math
from typing import List, Tuple


import cbadc.analog_system
import cbadc.digital_control
import cbadc.analog_signal
from ..ode_solver.mpmath import invariant_system_solver
import numpy as np
import sympy as sp
from .base_simulator import _BaseSimulator
from mpmath import mp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MPSimulator(_BaseSimulator):
    """Simulate the analog system and digital control interactions
    in the presence on analog signals.

    Parameters
    ----------
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        the analog system
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        the digital control
    input_signals : [:py:class:`cbadc.analog_signal.AnalogSignal`]
        a python list of analog signals (or a derived class)
    clock: :py:class:`cbadc.simulator.clock`, `optional`
        a clock to syncronize simulator output against, defaults to
        a phase delayed version of the digital_control clock.
    t_stop : `float`, optional
        determines a stop time, defaults to :py:obj:`math.inf`
    initial_state_vector: `array_like`, shape=(N), `optional`
        initial state vector.

    Attributes
    ----------
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        the analog system being simulated.
    digital_control : :py:class:`cbadc.digital_control.DigitalControl`
        the digital control being simulated.
    t : `float`
        current time of simulator.
    clock: `float`
        a clock to which the outputs of the simulator are synchronized.
    t_stop : `float`
        end time at which the generator raises :py:class:`StopIteration`.
    initial_state_vector: `array_like`, shape=(N), `optional`
        initial state vector.


    Yields
    ------
    `array_like`, shape=(M,), dtype=numpy.int8
    """

    def __init__(
        self,
        analog_system: cbadc.analog_system._valid_analog_system_types,
        digital_control: cbadc.digital_control._valid_digital_control_types,
        input_signal: List[cbadc.analog_signal._AnalogSignal],
        clock: cbadc.analog_signal._valid_clock_types = None,
        t_stop: float = math.inf,
        initial_state_vector=None,
        tol: float = 1e-12,
    ):
        super().__init__(
            analog_system,
            digital_control,
            input_signal,
            clock,
            t_stop,
            initial_state_vector,
        )
        mp.dps = 20
        A = mp.matrix(analog_system._A_s)
        B = mp.matrix(analog_system._B_s)
        Gamma = mp.matrix(analog_system._Gamma_s)
        tmp_Af, tmp_Gamma_f = invariant_system_solver(
            A,
            Gamma,
            [s for s in digital_control._impulse_response],
            (0, mp.mpf(self.clock.T)),
            tol=tol,
        )

        self.A = A
        self.B = B
        self.Af = tmp_Af
        self.Gamma_f = tmp_Gamma_f
        self._state_vector = mp.matrix(self._state_vector)
        self.tol = tol

    def _ode_solver(self, t_span: Tuple[float, float]):
        _, input_contributions = invariant_system_solver(
            self.A, self.B, self.input_signals, t_span, homogeneous=False, tol=self.tol
        )
        res = mp.matrix(self.analog_system.N, 1)

        for n in range(self.analog_system.N):
            for l in range(self.analog_system.L):
                res[n] += input_contributions[n, l]
        return res

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""

        t_end: float = self.t + self.clock.T
        t_span = np.array((self.t, t_end))
        if t_end >= self.t_stop:
            raise StopIteration
        # Compute

        # State transition
        self._state_vector = self.Af * self._state_vector
        # Input Signals
        for l in range(self.analog_system.L):
            self._state_vector += self._ode_solver(t_span)

        # Control signals
        self._state_vector += self.Gamma_f * mp.matrix(
            mp.mpf('2') * mp.matrix(self.digital_control._s) - mp.mpf('1')
        )

        # Update controls for next period if necessary
        self.digital_control.control_update(
            t_span[1], np.dot(self.analog_system.Gamma_tildeT, self._state_vector)
        )
        self.t = t_end
        return self.digital_control.control_signal()

    def __str__(self):
        return f"""{super().__str__()}

        {80 * '-'}

        Af:
        {self.Af}

        B:
        {self.B}

        Gamma:
        {self.Gamma_f}

        {80 * '='}
        """
