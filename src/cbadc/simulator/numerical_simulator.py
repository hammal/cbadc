"""Numerical solvers."""
import logging
import cbadc.analog_system
import cbadc.digital_control
import cbadc.analog_signal
import numpy as np
import scipy.integrate
import scipy.linalg
import math
from typing import Dict, List
from cbadc.simulator._base_simulator import _BaseSimulator
from scipy.integrate._ivp.ivp import OdeResult

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
    initial_state_vector: `array_like`, shape=(N), `optional`
        initial state vector.
    state_noise_covariance_matrix: `array_like`, shape=(N, N), `optional`
        the covariance matrix of white noise contributions at the state vector
        during each control step.
    atol, rtol : `float`, `optional`
        Relative and absolute tolerances. The solver keeps the local error
        estimates less.

    Attributes
    ----------
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        the analog system being simulated.
    digital_control : :py:class:`cbadc.digital_control.DigitalControl`
        the digital control being simulated.
    t : `float`
        current time of simulator.
    rtol, atol : `float`, `optional`
        Relative and absolute tolerances. The solver keeps the local error estimates less
        than atol + rtol * abs(y). Effects the underlying solver as described in
        :py:func:`scipy.integrate.solve_ivp`. Default to 1e-3 for rtol and 1e-6 for atol.

    Yields
    ------
    `array_like`, shape=(M,)
    """

    res: OdeResult

    def __init__(
        self,
        analog_system: cbadc.analog_system._valid_analog_system_types,
        digital_control: cbadc.digital_control._valid_digital_control_types,
        input_signal: List[cbadc.analog_signal._AnalogSignal],
        initial_state_vector: np.array = None,
        state_noise_covariance_matrix: np.ndarray = None,
        atol: float = 1e-15,
        rtol: float = 1e-10,
    ):
        super().__init__(
            analog_system,
            digital_control,
            input_signal,
            initial_state_vector=initial_state_vector,
            state_noise_covariance_matrix=state_noise_covariance_matrix,
        )
        self.atol = atol
        self.rtol = rtol
        self.res: OdeResult = None
        self._u = np.zeros(self.analog_system.L)
        self._input_signal = len(self.input_signals) > 0

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""
        self._full_ordinary_differential_solution()
        return self.digital_control.control_signal()

    def _full_ordinary_differential_solution(self):
        def derivative(t: float, y: np.ndarray):
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
            if len(self.input_signals) > 0:
                for _l in range(self.analog_system.L):
                    input_vector[_l] = self.input_signals[_l].evaluate(t + self.t)

            control_vector = self.digital_control.control_contribution(t)
            return self.analog_system.derivative(
                y, t + self.t, input_vector, control_vector
            )

        # while not np.allclose(self.t_rel, self.clock.T, atol=atol_clock):
        # Compute input at time t
        for t_old, t in zip(
            self.digital_control.clock_pulses[:-1],
            self.digital_control.clock_pulses[1:],
        ):
            if self._input_signal:
                for ell in range(self.analog_system.L):
                    self._u[ell] = self.input_signals[ell].evaluate(t_old + self.t)
            # update controls

            self.digital_control.control_update(
                t_old,
                self.analog_system.control_observation(
                    t_old + self.t,
                    self.state_vector(),
                    self._u,
                    self.digital_control.control_contribution(t_old),
                ),
            )

            self.res: OdeResult = scipy.integrate.solve_ivp(
                derivative,
                (t_old, t),
                self.state_vector(),
                atol=self.atol,
                rtol=self.rtol,
                # method="Radau",
                # jac=self.analog_system.A,
                method="DOP853",
                # method='LSODA',
                # method="BDF",
            )
            self._state_vector = self.res.y[:, -1]

        # if thermal noise
        if self.noisy:
            self._state_vector += np.dot(
                self._cov_deviation, np.random.randn(self.analog_system.N)
            )

        self.t += self.digital_control.clock.T

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
    `array_like`, shape=(M,)

    """

    res: OdeResult

    def __init__(
        self,
        analog_system: cbadc.analog_system._valid_analog_system_types,
        digital_control: cbadc.digital_control._valid_digital_control_types,
        input_signal: List[cbadc.analog_signal._AnalogSignal],
        initial_state_vector: np.array = None,
        state_noise_covariance_matrix: np.ndarray = None,
        atol: float = 1e-15,
        rtol: float = 1e-10,
    ):
        super().__init__(
            analog_system,
            digital_control,
            input_signal,
            initial_state_vector=initial_state_vector,
            state_noise_covariance_matrix=state_noise_covariance_matrix,
        )

        if not analog_system.pre_computable:
            raise ValueError(
                "The analog system must be pre-computable to use this simulator."
            )

        self.atol = atol
        self.rtol = rtol
        self._u = np.zeros(self.analog_system.L)
        self._input_signal = len(self.input_signals) > 0

        self._pre_computations()

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""
        self._ordinary_differential_solution()
        return self.digital_control.control_signal()

    def _analog_system_matrix_exponential(self, t: float) -> np.ndarray:
        return np.asarray(scipy.linalg.expm(np.asarray(self.analog_system.A) * t))

    def _analog_system_matrix_exponential_with_ivp(self, T: float) -> np.ndarray:
        a_exp = np.zeros_like(self.analog_system.A, dtype=float)

        for n in range(self.analog_system.N):

            def derivative(t, x):
                return np.dot(self.analog_system.A, x)

            unit_vector = np.zeros(self.analog_system.N)
            unit_vector[n] = 1.0

            self.res = scipy.integrate.solve_ivp(
                derivative,
                (0.0, T),
                unit_vector,
                atol=self.atol,
                rtol=self.rtol,
                # method="RK45",
                # method="Radau",
                method="DOP853",
                # method="BDF",
                # method="LSODA",
                # jac=self.analog_system.A,
            )
            a_exp[:, n] = self.res.y[:, -1]

        return a_exp

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
        logger.info("Executing pre-computations.")

        self._pre_computed_state_transition_matrices = []
        self._pre_computed_control_matrices = []

        for t_old, t in zip(
            self.digital_control.clock_pulses[:-1],
            self.digital_control.clock_pulses[1:],
        ):
            # expm(A T_s)
            self._pre_computed_state_transition_matrices.append(
                self._analog_system_matrix_exponential(t - t_old)
            )
            # self._pre_computed_state_transition_matrices.append(
            #     self._analog_system_matrix_exponential_with_ivp(t - t_old)
            # )

            temp = np.zeros((self.analog_system.N, self.analog_system.M), dtype=float)

            for m in range(self.analog_system.M):
                if t > self.digital_control._impulse_response[m].t0:

                    def dac_waveform(m, t):
                        return self.digital_control.impulse_response(m, t)

                else:

                    def dac_waveform(m, t):
                        return self.digital_control.impulse_response(
                            m, t + self.digital_control.clock.T
                        )

                def derivative(t, x):
                    return np.dot(self.analog_system.A, x) + np.dot(
                        self.analog_system.Gamma, dac_waveform(m, t)
                    )

                self.res = scipy.integrate.solve_ivp(
                    derivative,
                    (t_old, t),
                    np.zeros((self.analog_system.N)),
                    atol=self.atol,
                    rtol=self.rtol,
                    # method="RK45",
                    # method="Radau",
                    method="DOP853",
                    # method="BDF",
                    # method="LSODA",
                    # jac=self.analog_system.A,
                )
                temp[:, m] = self.res.y[:, -1]

            self._pre_computed_control_matrices.append(temp)

    def _ordinary_differential_solution(self):
        # Compute signal contribution

        if self.digital_control._mulit_phase:
            raise NotImplementedError("Not yet implemented for multi-phase control.")

        for i, (t_old, t) in enumerate(
            zip(
                self.digital_control.clock_pulses[:-1],
                self.digital_control.clock_pulses[1:],
            )
        ):
            if self._input_signal:
                for ell in range(self.analog_system.L):
                    self._u[ell] = self.input_signals[ell].evaluate(t_old + self.t)

            # update controls
            self.digital_control.control_update(
                t_old,
                self.analog_system.control_observation(
                    t_old + self.t,
                    self.state_vector(),
                    self._u,
                    self.digital_control.control_contribution(t_old),
                ),
            )

            # Homogenious solution
            self._state_vector = np.dot(
                self._pre_computed_state_transition_matrices[i], self.state_vector()
            ).flatten()

            if self._input_signal:

                def f(_t, x):
                    res = np.dot(self.analog_system.A, x)
                    for _l in range(self.analog_system.L):
                        res += np.dot(
                            self.analog_system.B[:, _l],
                            self.input_signals[_l].evaluate(_t + self.t),
                        )
                    return res.flatten()

                self.res = scipy.integrate.solve_ivp(
                    f,
                    (t_old, t),
                    np.zeros(self.analog_system.N),
                    atol=self.atol,
                    rtol=self.rtol,
                    # method="RK45",
                    # method="Radau",
                    method="DOP853",
                    # method="BDF",
                    # method="LSODA",
                    # jac=self.analog_system.A,
                )
                self._state_vector += self.res.y[:, -1]

            # Partial control solution
            self._state_vector += np.dot(
                self._pre_computed_control_matrices[i],
                np.asarray(2 * self.digital_control._s - 1, dtype=np.double),
            ).flatten()

        # if thermal noise
        if self.noisy:
            self._state_vector += np.dot(
                self._cov_deviation, np.random.randn(self.analog_system.N)
            )

        self.t += self.digital_control.clock.T

    def __str__(self):
        return f"""{super().__str__()}

        {80 * '='}

atol, rtol:
{self.atol}, {self.rtol}

Pre computed transition matrices:
{self._pre_computed_state_transition_matrices}

Pre-computed control matrices
{self._pre_computed_control_matrices}

        """
