"""Analytical solvers."""
import logging
import math
from typing import List
import cbadc.analog_system
import cbadc.digital_control
import cbadc.analog_signal
from ..ode_solver.sympy import invariant_system_solver
import numpy as np
import sympy as sp
import mpmath as mp
from ._base_simulator import _BaseSimulator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnalyticalSimulator(_BaseSimulator):
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
    `array_like`, shape=(M,)
    """

    def __init__(
        self,
        analog_system: cbadc.analog_system._valid_analog_system_types,
        digital_control: cbadc.digital_control._valid_digital_control_types,
        input_signal: List[cbadc.analog_signal._AnalogSignal],
        clock: cbadc.analog_signal._valid_clock_types = None,
        t_stop: float = math.inf,
        initial_state_vector=None,
    ):
        super().__init__(
            analog_system,
            digital_control,
            input_signal,
            clock,
            t_stop,
            initial_state_vector,
        )
        mp.dps = 30
        self._state_vector = mp.matrix(self._state_vector)
        signals = [
            *[s.symbolic() for s in self.input_signals],
            *[s.symbolic() for s in self.digital_control._impulse_response],
        ]
        initial_conditions_times = [
            *[s.t0 for s in self.input_signals],
            *[s.t0 for s in self.digital_control._impulse_response],
        ]
        tmp_Bf = sp.Matrix([[self.analog_system._B_s, self.analog_system._Gamma_s]])
        tmp_Af, tmp_Bf, t = invariant_system_solver(
            self.analog_system._A_s, tmp_Bf, signals, initial_conditions_times
        )

        # replace t and extract rhs of expression
        tmp_Bf = [[s.subs(t, self.clock.T).rhs for s in ss] for ss in tmp_Bf]

        # self.Af = np.array(tmp_Af.evalf(subs={t: self.clock.T})).astype(np.float64)
        self.Af = mp.matrix(tmp_Af.evalf(subs={t: self.clock.T}))

        self.Bf = [
            [None for _ in range(self.analog_system.L)]
            for _ in range(self.analog_system.N)
        ]
        # self.Gamma = np.zeros(
        #     (self.analog_system.N, self.analog_system.M), dtype=np.float64
        # )
        self.Gamma = mp.matrix(self.analog_system.N, self.analog_system.M)

        self.Gamma_tildeT = mp.matrix(self.analog_system.Gamma_tildeT)

        for n in range(self.analog_system.N):
            for l in range(self.analog_system.L):
                self.Bf[n][l] = sp.lambdify(
                    (self.input_signals[l].t, self.input_signals[l].sym_phase),
                    sp.re(tmp_Bf[l][n]),
                )
            for m in range(self.analog_system.M):
                self.Gamma[n, m] = tmp_Bf[m + self.analog_system.L][n].doit().evalf()

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""

        t_end: float = self.t + self.clock.T
        t_span = np.array((self.t, t_end))
        if t_end >= self.t_stop:
            raise StopIteration
        # Compute

        # State transition
        # self._state_vector = np.dot(self.Af, self._state_vector)
        self._state_vector = self.Af * self._state_vector
        # Input Signals
        for n in range(self.analog_system.N):
            for l in range(self.analog_system.L):
                if isinstance(self.input_signals[l], cbadc.analog_signal.Sinusoidal):
                    artifical_phase = (
                        self.input_signals[l].angularFrequency * self.t
                    ) % (2 * np.pi)

                else:
                    artifical_phase = 0
                self._state_vector[n] += self.Bf[n][l](t_end - self.t, artifical_phase)

        # Control signals
        # self._state_vector += np.dot(
        #     self.Gamma, np.asarray(2 * self.digital_control._s - 1, dtype=np.double)
        # ).flatten()

        self._state_vector += self.Gamma * (2 * mp.matrix(self.digital_control._s) - 1)

        # Update controls for next period if necessary
        control_observation = np.array(
            (self.Gamma_tildeT * self._state_vector).tolist(), dtype=np.float64
        ).flatten()
        self.digital_control.control_update(t_span[1], control_observation)
        self.t = t_end
        return self.digital_control.control_signal()

    def __str__(self):
        return f"""{super().__str__()}

        {80 * '-'}

        Af:
        {self.Af}

        Bf:
        {self.Bf}

        Gamma:
        {self.Gamma}

        {80 * '='}
        """
