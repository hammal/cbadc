"""Extended numerical precision solvers."""
import logging
import math
from typing import List, Tuple


import cbadc.analog_system
import cbadc.digital_control
import cbadc.analog_signal
from ..ode_solver.mpmath import invariant_system_solver
import numpy as np
import sympy as sp
from ._base_simulator import _BaseSimulator
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
    decimal_places: `int`, optional
        number of decimal places used in simulation

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
        tol: float = 1e-20,
        decimal_places=20,
    ):
        super().__init__(
            analog_system,
            digital_control,
            input_signal,
            clock,
            t_stop,
            initial_state_vector,
        )
        # Fix decimal places
        self.dps = decimal_places
        self.tol = tol
        A = mp.matrix(analog_system._A_s)
        B = mp.matrix(analog_system._B_s)
        Gamma = mp.matrix(analog_system._Gamma_s)
        tmp_dps = mp.dps
        mp.dps = self.dps
        tmp_Af, tmp_Gamma_f = invariant_system_solver(
            A,
            Gamma,
            [s for s in digital_control._impulse_response],
            (0, mp.mpf(self.clock.T)),
            tol=self.tol,
        )

        self.A = A
        self.B = B
        self.Af = tmp_Af
        self.Gamma_f = tmp_Gamma_f
        self.Gamma_tilde_f = mp.matrix(analog_system.Gamma_tildeT)
        self.D_tilde = mp.matrix(analog_system.B_tilde)
        self._state_vector = mp.matrix(self._state_vector)
        mp.dps = tmp_dps

    def _ode_solver_1(self, t_span: Tuple[float, float]):
        def diff_equation(x, y):
            res = mp.matrix(self.analog_system.N, 1)
            for n in range(self.analog_system.N):
                for nn in range(self.analog_system.N):
                    res[n] += self.A[n, nn] * y[nn]
            for l in range(self.analog_system.L):
                res += self.B[:, l] * self.input_signals[l]._mpmath(x)
            return res

        f = mp.odefun(diff_equation, t_span[0], self._state_vector, tol=self.tol)
        return mp.matrix(f(t_span[1]))

    def _ode_solver_2(self, t_span: Tuple[float, float]):
        tmp_dps = mp.dps
        mp.dps = self.dps

        def diff_equation(x, y):
            res = mp.matrix(self.analog_system.N, 1)
            for n in range(self.analog_system.N):
                for nn in range(self.analog_system.N):
                    res[n] += self.A[n, nn] * y[nn]
            for l in range(self.analog_system.L):
                res += self.B[:, l] * self.input_signals[l]._mpmath(x)
            return res

        f = mp.odefun(
            diff_equation,
            t_span[0],
            [0 for _ in range(self.analog_system.N)],
            tol=self.tol,
        )
        res = mp.matrix(f(t_span[1]))
        res += mp.matrix(self.Af * self._state_vector)
        mp.dps = tmp_dps
        return res

    def _input_signal(self, t):
        u = np.zeros(self.analog_system.L)
        for l in range(self.analog_system.L):
            u[l] = np.array(self.input_signals[l]._mpmath(t))
        return u

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""

        t_end: float = self.t + self.clock.T
        t_span = np.array((self.t, t_end))
        if t_end >= self.t_stop:
            raise StopIteration
        # Compute

        # ODE solver
        # self._state_vector = self._ode_solver_1(t_span)
        self._state_vector = self._ode_solver_2(t_span)

        # Control signals
        self._state_vector += self.Gamma_f * mp.matrix(
            mp.mpf('2') * mp.matrix(self.digital_control.control_signal()) - mp.mpf('1')
        )

        # Update controls for next period if necessary
        temp = self.D_tilde * self.input_signals[0]._mpmath(t_end)
        for l in range(1, self.analog_system.L):
            temp += self.D_tilde * self.input_signals[l]._mpmath(t_end)

        control_observation = self.analog_system.control_observation(
            np.array(self._state_vector),
            self._input_signal(t_end),
            self.digital_control.control_signal(),
        )

        self.digital_control.control_update(t_span[1], control_observation)
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
