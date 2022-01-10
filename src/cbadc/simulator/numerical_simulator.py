"""
This module implements numeric solvers for
simulating the analog_system digital control interactions.

"""

import logging
import cbadc.analog_system
import cbadc.digital_control
import cbadc.analog_signal
import numpy as np
import scipy.integrate
import scipy.linalg
import math
from typing import List
from .base_simulator import _BaseSimulator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FullSimulator(_BaseSimulator):
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
    rtol, atol : `float`, `optional`
        Relative and absolute tolerances. The solver keeps the local error estimates less
        than atol + rtol * abs(y). Effects the underlying solver as described in
        :py:func:`scipy.integrate.solve_ivp`. Default to 1e-3 for rtol and 1e-6 for atol.
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
        atol: float = 1e-12,
        rtol: float = 1e-8,
    ):
        super().__init__(
            analog_system,
            digital_control,
            input_signal,
            clock,
            t_stop,
            initial_state_vector,
        )
        self.atol = atol
        self.rtol = rtol

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""

        t_end: float = self.t + self.clock.T
        t_span = np.array((self.t, t_end))
        if t_end >= self.t_stop:
            raise StopIteration
        # Solve full diff equation.
        self._state_vector = self._full_ordinary_differential_solution(t_span)
        self.t = t_end
        return self.digital_control.control_signal()

    def _analog_system_matrix_exponential(self, t: float) -> np.ndarray:
        return np.asarray(scipy.linalg.expm(np.asarray(self.analog_system.A) * t))

    def _full_ordinary_differential_solution(self, t_span: np.ndarray) -> np.ndarray:
        def f(t: float, y: np.ndarray):
            """Solve the differential computational problem
            of the analog system and digital control interaction

            Parameters
            ----------
            t : `float`
                the time for evaluation
            y : array_lik, shape=(N,)
                state vector

            Returns
            -------
            array_like, shape=(N,)
                vector of derivatives evaluated at time t.
            """
            input_vector = np.zeros(self.analog_system.L)
            for _l in range(self.analog_system.L):
                input_vector[_l] = self.input_signals[_l].evaluate(t)

            control_vector = self.digital_control.control_contribution(t)

            delta = self.analog_system.derivative(y, t, input_vector, control_vector)
            return delta

        # terminate in case of control update
        def control_update(t, x):
            return t - self.digital_control._t_next

        control_update.terminal = True
        control_update.direction = 1.0
        t0_impulse_response = [
            lambda t, x: t
            - self.digital_control._t_last_update[m]
            - self.digital_control._impulse_response[m].t0
            for m in range(self.digital_control.M)
        ]
        for i_resp in t0_impulse_response:
            i_resp.direction = 1.0

        # Default Case
        t = t_span[0]
        y_new = self._state_vector[:]
        atol_clock = self.digital_control.clock.T * 1e-4
        while not np.allclose(t, t_span[1], atol=atol_clock):
            res = scipy.integrate.solve_ivp(
                f,
                (t, t_span[1]),
                y_new,
                atol=self.atol,
                rtol=self.rtol,
                # method="Radau",
                # jac=self.analog_system.A,
                # method="DOP853",
                events=(
                    control_update,
                    *t0_impulse_response,
                ),
            )
            if res.status == -1:
                logger.critical(f"IVP solver failed, See:\n\n{res}")
            # In case of control update event
            t = res.t[-1]
            y_new = res.y[:, -1]
            if res.status == 1 or t == t_span[1]:
                self.digital_control.control_update(
                    t, np.dot(self.analog_system.Gamma_tildeT, y_new)
                )
        return y_new

    def __str__(self):
        return f"""{super().__str__()}

{80 * '-'}

atol, rtol:
{self.atol}, {self.rtol}

{80 * '='}
        """


class PreComputedControlSignalsSimulator(_BaseSimulator):
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
    rtol, atol : `float`, `optional`
        Relative and absolute tolerances. The solver keeps the local error estimates less
        than atol + rtol * abs(y). Effects the underlying solver as described in
        :py:func:`scipy.integrate.solve_ivp`. Default to 1e-3 for rtol and 1e-6 for atol.


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
    rtol, atol : `float`, `optional`
        Relative and absolute tolerances. The solver keeps the local error estimates less
        than atol + rtol * abs(y). Effects the underlying solver as described in
        :py:func:`scipy.integrate.solve_ivp`. Default to 1e-3 for rtol and 1e-6 for atol.

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
        atol: float = 1e-12,
        rtol: float = 1e-8,
    ):
        _BaseSimulator.__init__(
            self,
            analog_system,
            digital_control,
            input_signal,
            clock,
            t_stop,
            initial_state_vector,
        )

        self._temp_state_vector = np.zeros(self.analog_system.N, dtype=np.double)
        self.atol = atol
        self.rtol = rtol
        if self.clock.T != self.digital_control.clock.T:
            raise Exception(
                "For this simulator, both simulation clock and digital control clock must have same clock period."
            )
        self._pre_computations()

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""

        t_end: float = self.t + self.clock.T
        t_span = np.array((self.t, t_end))
        if t_end >= self.t_stop:
            raise StopIteration
        self._state_vector = self._ordinary_differential_solution(t_span)
        self.t = t_end
        return self.digital_control.control_signal()

    def _analog_system_matrix_exponential(self, t: float) -> np.ndarray:
        return np.asarray(scipy.linalg.expm(np.asarray(self.analog_system.A) * t))

    def _pre_computations(self):
        """Precomputes quantities for quick evaluation of state transition and control
        contribution.

        Specifically,

        :math:`\exp\\left(\mathbf{A} T_s \\right)`

        and

        :math:`\mathbf{A}_c = \int_{0}^{T_s} \exp\\left(\mathbf{A} (T_s - \tau)\\right) \mathbf{\Gamma} \mathbf{d}(\tau) \mathrm{d} \tau`

        are computed where the formed describes the state transition and the latter
        the control contributions. Furthermore, :math:`\mathbf{d}(\tau)` is the DAC waveform
        (or impulse response) of the digital control.
        """
        logger.info("Executing precomputations.")
        # expm(A T_s)
        self._pre_computed_state_transition_matrix = (
            self._analog_system_matrix_exponential(self.clock.T)
        )

        self._pre_computed_control_matrix = np.zeros(
            (self.analog_system.N, self.analog_system.M)
        )

        for m in range(self.analog_system.M):

            def derivative(t, x):
                dac_waveform = self.digital_control.impulse_response(m, t)
                return np.dot(self.analog_system.A, x) + np.dot(
                    self.analog_system.Gamma, dac_waveform
                )

            def impulse_start(t, x):
                return t - self.digital_control._impulse_response[m].t0

            # impulse_start.terminate = True
            impulse_start.direction = 1.0

            tspan = np.array([0, self.digital_control.clock.T])

            sol = scipy.integrate.solve_ivp(
                derivative,
                (tspan[0], tspan[1]),
                np.zeros((self.analog_system.N)),
                atol=self.atol,
                rtol=self.rtol,
                # method="RK45",
                method="Radau",
                # method="DOP853",
                jac=self.analog_system.A,
                events=(impulse_start,),
            )
            self._pre_computed_control_matrix[:, m] = sol.y[:, -1]

    def _ordinary_differential_solution(self, t_span: np.ndarray) -> np.ndarray:
        """Computes system ivp in three parts:

        First solve input signal contribution by computing

        :math:`\mathbf{u}_{c} = \int_{t_1}^{t_2} \mathbf{A} x(t) + \mathbf{B} \mathbf{u}(t) \mathrm{d} t`

        where :math:`\mathbf{x}(t_1) = \begin{pmatrix} 0, & \dots, & 0 \end{pmatrix}^{\mathsf{T}}`.

        Secondly advance the previous state as

        :math:`\mathbf{x}_c = \mathbf{x}(t_2) = \exp\\left( \mathbf{A} T_s \\right) \mathbf{x}(t_1)`

        Thirdly, compute the control contribution by

        :math:`\mathbf{s}_c = \mathbf{A}_c \mathbf{s}[k]`

        where :math:`\mathbf{A}_c = \int_{0}^{T_s} \exp\\left(\mathbf{A} (T_s - \tau)\\right) \mathbf{\Gamma} \mathbf{d}(\tau) \mathrm{d} \tau`
        and :math:`\mathbf{d}(\tau)` is the DAC waveform (or impulse response) of the digital control.

        Finally, all contributions are added and returned as

        :math:`\mathbf{u}_{c} + \mathbf{x}_c + \mathbf{s}_c`.

        Parameters
        ----------
        t_span : (float, float)
            the initial time :math:`t_1` and end time :math:`t_2` of the
            simulation.

        Returns
        -------
        array_like, shape=(N,)
            computed state vector.
        """

        # Compute signal contribution
        def f(t, x):
            res = np.dot(self.analog_system.A, x)
            for _l in range(self.analog_system.L):
                res += np.dot(
                    self.analog_system.B[:, _l], self.input_signals[_l].evaluate(t)
                )
            return res.flatten()

        sol = scipy.integrate.solve_ivp(
            f,
            (t_span[0], t_span[1]),
            np.zeros(self.analog_system.N),
            atol=self.atol,
            rtol=self.rtol,
            method="RK45",
        )

        if sol.status == -1:
            logger.critical(f"IVP solver failed, See:\n\n{sol}")

        self._temp_state_vector = sol.y[:, -1]

        self._temp_state_vector += np.dot(
            self._pre_computed_state_transition_matrix, self._state_vector
        ).flatten()

        self._temp_state_vector += np.dot(
            self._pre_computed_control_matrix,
            np.asarray(2 * self.digital_control._s - 1, dtype=np.double),
        ).flatten()

        # Update controls for next period if necessary
        self.digital_control.control_update(
            t_span[1], np.dot(self.analog_system.Gamma_tildeT, self._temp_state_vector)
        )

        return self._temp_state_vector

    def __str__(self):
        return f"""{super().__str__()}

        {80 * '='}

atol, rtol:
{self.atol}, {self.rtol}

Pre computed transition matrix:
{self._pre_computed_state_transition_matrix}

Pre-computed control matrix
{self._pre_computed_control_matrix}

        """
