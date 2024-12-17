"""Numerical solvers."""

import logging
import cbadc.analog_filter
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
    analog_filter : :py:class:`cbadc.analog_filter.AnalogSystem`
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
    analog_filter : :py:class:`cbadc.analog_filter.AnalogSystem`
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
        analog_filter: cbadc.analog_filter._valid_analog_filter_types,
        digital_control: cbadc.digital_control._valid_digital_control_types,
        input_signal: List[cbadc.analog_signal._AnalogSignal],
        initial_state_vector: np.array = None,
        state_noise_covariance_matrix: np.ndarray = None,
        atol: float = 1e-15,
        rtol: float = 1e-10,
        seed: int = 4212312513432239834528672,
    ):
        super().__init__(
            analog_filter,
            digital_control,
            input_signal,
            initial_state_vector=initial_state_vector,
            state_noise_covariance_matrix=state_noise_covariance_matrix,
            seed=seed,
        )
        self.atol = atol
        self.rtol = rtol
        self.res: OdeResult = None
        self._u = np.zeros(self.analog_filter.L)
        self._input_signal = len(self.input_signals) > 0

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""
        self._full_ordinary_differential_solution()
        return self.digital_control.control_signal()

    def _full_ordinary_differential_solution(self):
        # while not np.allclose(self.t_rel, self.clock.T, atol=atol_clock):
        # Compute input at time t
        for t_old, t in zip(
            self.digital_control.clock_pulses[:-1],
            self.digital_control.clock_pulses[1:],
        ):
            if self._input_signal:
                for ell in range(self.analog_filter.L):
                    self._u[ell] = self.input_signals[ell].evaluate(t_old + self.t)
                    # help constant signals to progress at the clock edge
                    self.input_signals[ell].tick()

            # update controls
            self.digital_control.control_update(
                t_old,
                self.analog_filter.control_observation(
                    t_old + self.t,
                    self.state_vector(),
                    self._u,
                    self.digital_control.control_contribution(t_old),
                ),
            )

            if self._input_signal:

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
                    input_vector = np.zeros(self.analog_filter.L, dtype=np.float64)
                    if len(self.input_signals) > 0:
                        for _l in range(self.analog_filter.L):
                            input_vector[_l] = self.input_signals[_l].evaluate(
                                t + self.t
                            )
                    # print(input_vector, t + self.t, self.digital_control.clock.T)
                    control_vector = self.digital_control.control_contribution(t)
                    return self.analog_filter.derivative(
                        y, t + self.t, input_vector, control_vector
                    )

            else:

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
                    return np.dot(self.analog_filter.A, self.state_vector()) + np.dot(
                        self.analog_filter.Gamma,
                        self.digital_control.control_contribution(t),
                    )

            self.res: OdeResult = scipy.integrate.solve_ivp(
                derivative,
                (t_old, t),
                self.state_vector(),
                atol=self.atol,
                rtol=self.rtol,
                # method="Radau",
                # jac=self.analog_filter.A,
                method="DOP853",
                # method='LSODA',
                # method="BDF",
            )
            # print(self.res.y[:, -1], t)
            self._state_vector[:] = self.res.y[:, -1]

        # if thermal noise
        if self.noisy:
            self._state_vector += (
                np.dot(self._cov_deviation, self.rng.normal(size=self.analog_filter.N))
                * self.digital_control.clock.T
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
    analog_filter : :py:class:`cbadc.analog_filter.AnalogSystem`
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
    analog_filter : :py:class:`cbadc.analog_filter.AnalogSystem`
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
        analog_filter: cbadc.analog_filter._valid_analog_filter_types,
        digital_control: cbadc.digital_control._valid_digital_control_types,
        input_signal: List[cbadc.analog_signal.Sinusoidal],
        initial_state_vector: np.array = None,
        state_noise_covariance_matrix: np.ndarray = None,
        atol: float = 1e-15,
        rtol: float = 1e-10,
        seed: int = 4212312513432239834528672,
    ):
        super().__init__(
            analog_filter,
            digital_control,
            input_signal,
            initial_state_vector=initial_state_vector,
            state_noise_covariance_matrix=state_noise_covariance_matrix,
            seed=seed,
        )

        for analog_signal in self.input_signals:
            if not analog_signal.piecewise_constant and not isinstance(
                analog_signal, cbadc.analog_signal.Sinusoidal
            ):
                raise ValueError(
                    "Only piecewise constant and sinusoidal signals are supported in this simulator."
                )

        if not analog_filter.pre_computable:
            raise ValueError(
                "The analog system must be pre-computable to use this simulator."
            )

        if digital_control._mulit_phase:
            raise NotImplementedError("Not yet implemented for multi-phase control.")

        self.atol = atol
        self.rtol = rtol
        self._u = np.zeros(self.analog_filter.L)
        self._input_signal = len(self.input_signals) > 0

        self._pre_computations()

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""
        self._ordinary_differential_solution()
        return self.digital_control.control_signal()

    def _analog_filter_matrix_exponential(self, t: float) -> np.ndarray:
        return np.asarray(scipy.linalg.expm(np.asarray(self.analog_filter.A) * t))

    def _analog_filter_matrix_exponential_with_ivp(self, T: float) -> np.ndarray:
        a_exp = np.zeros_like(self.analog_filter.A, dtype=np.float64)

        for n in range(self.analog_filter.N):

            def derivative(t, x):
                return np.dot(self.analog_filter.A, x)

            unit_vector = np.zeros(self.analog_filter.N)
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
                # jac=self.analog_filter.A,
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

        # Transfer function in polar coordinates where first (0, n, l) is the magnitude
        # and (1, n, l) is the phase in radians.

        # Figure out the sinusoidal and piecewise constant signals
        self._sinusoidal_signals = []
        self._sinusoidal_signals_index = []
        self._piecewise_constant_signals = []
        self._piecewise_constant_signals_index = []
        for l, analog_signal in enumerate(self.input_signals):
            if isinstance(analog_signal, cbadc.analog_signal.Sinusoidal):
                self._sinusoidal_signals.append(analog_signal)
                self._sinusoidal_signals_index.append(l)
            elif analog_signal.piecewise_constant:
                self._piecewise_constant_signals.append(analog_signal)
                self._piecewise_constant_signals_index.append(l)
            else:
                raise ValueError(
                    "Only piecewise constant and sinusoidal signals are supported."
                )

        # Pre-compute the transfer function and angular frequencies
        number_of_sin = len(self._sinusoidal_signals)
        self._pre_computed_transfer_function = np.zeros(
            (2, self.analog_filter.N, number_of_sin), dtype=np.float64
        )
        self._pre_computed_angular_frequencies = np.zeros((number_of_sin, 1))

        for l, (l_index, analog_signal) in enumerate(
            zip(self._sinusoidal_signals_index, self._sinusoidal_signals)
        ):
            tf = self.analog_filter.transfer_function_matrix(
                np.array([analog_signal.angularFrequency])
            )
            self._pre_computed_transfer_function[0, :, l] = (
                np.abs(tf[:, l_index, 0]) * analog_signal.amplitude
            )
            self._pre_computed_transfer_function[1, :, l] = (
                np.angle(tf[:, l_index, 0]) + analog_signal.phase
            )
            self._pre_computed_angular_frequencies[l] = analog_signal.angularFrequency
            if analog_signal.offset != 0:
                raise NotImplementedError("Offset is not yet supported.")

        # Pre-compute piecewise constant signals
        number_of_piecewise = len(self._piecewise_constant_signals)
        self._precomputed_piecewise_constant_transition = []

        for t_old, t in zip(
            self.digital_control.clock_pulses[:-1],
            self.digital_control.clock_pulses[1:],
        ):
            # expm(A T_s)
            self._pre_computed_state_transition_matrices.append(
                self._analog_filter_matrix_exponential(t - t_old)
            )
            # self._pre_computed_state_transition_matrices.append(
            #     self._analog_filter_matrix_exponential_with_ivp(t - t_old)
            # )

            temp = np.zeros(
                (self.analog_filter.N, self.analog_filter.M), dtype=np.float64
            )

            for m in range(self.analog_filter.M):
                if t > self.digital_control._impulse_response[m].t0:

                    def dac_waveform(m, t):
                        return self.digital_control.impulse_response(m, t)

                else:

                    def dac_waveform(m, t):
                        return self.digital_control.impulse_response(
                            m, t + self.digital_control.clock.T
                        )

                def derivative(t, x):
                    return np.dot(self.analog_filter.A, x) + np.dot(
                        self.analog_filter.Gamma, dac_waveform(m, t)
                    )

                self.res = scipy.integrate.solve_ivp(
                    derivative,
                    (t_old, t),
                    np.zeros((self.analog_filter.N)),
                    atol=self.atol,
                    rtol=self.rtol,
                    # method="RK45",
                    # method="Radau",
                    method="DOP853",
                    # method="BDF",
                    # method="LSODA",
                    # jac=self.analog_filter.A,
                )
                temp[:, m] = self.res.y[:, -1]

            self._pre_computed_control_matrices.append(temp)

            temp_piecewise = np.zeros(
                (self.analog_filter.N, number_of_piecewise), dtype=np.float64
            )
            for l, (l_index, analog_signal) in enumerate(
                zip(
                    self._piecewise_constant_signals_index,
                    self._piecewise_constant_signals,
                )
            ):

                def derivative(t, x):
                    return (
                        np.dot(self.analog_filter.A, x)
                        + self.analog_filter.B[:, l_index]
                    ).flatten()

                self.res = scipy.integrate.solve_ivp(
                    derivative,
                    (t_old, t),
                    np.zeros((self.analog_filter.N)),
                    atol=self.atol,
                    rtol=self.rtol,
                    # method="RK45",
                    # method="Radau",
                    method="DOP853",
                    # method="BDF",
                    # method="LSODA",
                    # jac=self.analog_filter.A,
                )
                temp_piecewise[:, l] = self.res.y[:, -1]
            self._precomputed_piecewise_constant_transition.append(temp_piecewise)

    def _ordinary_differential_solution(self):
        # Compute signal contribution

        for i, (t_old, t) in enumerate(
            zip(
                self.digital_control.clock_pulses[:-1],
                self.digital_control.clock_pulses[1:],
            )
        ):
            if self._input_signal:
                for ell in range(self.analog_filter.L):
                    self._u[ell] = self.input_signals[ell].evaluate(self.t + t_old)

            self.digital_control.control_update(
                t_old,
                self.analog_filter.control_observation(
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

            # Compute the sinusoidal signals
            if self._sinusoidal_signals:

                sine_after = np.sum(
                    self._pre_computed_transfer_function[0, :, :]
                    * np.sin(
                        self._pre_computed_angular_frequencies * (self.t + t)
                        + self._pre_computed_transfer_function[1, :, :]
                    ),
                    axis=1,
                )

                sine_before = np.sum(
                    self._pre_computed_transfer_function[0, :, :]
                    * np.sin(
                        self._pre_computed_angular_frequencies * (self.t + t_old)
                        + self._pre_computed_transfer_function[1, :, :]
                    ),
                    axis=1,
                )
                self._state_vector += sine_after - np.dot(
                    self._pre_computed_state_transition_matrices[i], sine_before
                )

            # Compute the piecewise constant signals
            if self._piecewise_constant_signals:

                # self._state_vector +=
                _u = np.array(
                    [
                        self.input_signals[ell].evaluate(self.t + t)
                        for ell in self._piecewise_constant_signals_index
                    ]
                )
                pre_comp = np.dot(
                    self._precomputed_piecewise_constant_transition[i],
                    # self._u[self._piecewise_constant_signals_index],
                    _u,
                ).flatten()
                self._state_vector += pre_comp

                # def f(_t, x):
                #     res = np.dot(self.analog_filter.A, x)
                #     for _l in self._piecewise_constant_signals_index:
                #         res += np.dot(
                #             self.analog_filter.B[:, _l],
                #             self.input_signals[_l].evaluate(_t + self.t),
                #         )
                #     return res.flatten()

                # self.res = scipy.integrate.solve_ivp(
                #     f,
                #     (t_old, t),
                #     np.zeros(self.analog_filter.N),
                #     atol=self.atol,
                #     rtol=self.rtol,
                #     # method="RK45",
                #     # method="Radau",
                #     method="DOP853",
                #     # method="BDF",
                #     # method="LSODA",
                #     # jac=self.analog_filter.A,
                # )
                # self._state_vector += self.res.y[:, -1]

                # print(pre_comp - self.res.y[:, -1])

            ## The old way of computing the piecewise constant and sinusoidal signals
            # if self._input_signal:
            #     def f(_t, x):
            #         res = np.dot(self.analog_filter.A, x)
            #         for _l in range(self.analog_filter.L):
            #             res += np.dot(
            #                 self.analog_filter.B[:, _l],
            #                 self.input_signals[_l].evaluate(_t + self.t),
            #             )
            #         return res.flatten()

            #     self.res = scipy.integrate.solve_ivp(
            #         f,
            #         (t_old, t),
            #         np.zeros(self.analog_filter.N),
            #         atol=self.atol,
            #         rtol=self.rtol,
            #         # method="RK45",
            #         # method="Radau",
            #         method="DOP853",
            #         # method="BDF",
            #         # method="LSODA",
            #         # jac=self.analog_filter.A,
            #     )
            #     self._state_vector += self.res.y[:, -1]

            # Partial control solution
            self._state_vector += np.dot(
                self._pre_computed_control_matrices[i],
                np.asarray(2 * self.digital_control._s - 1, dtype=np.double),
            ).flatten()

        # if thermal noise
        if self.noisy:
            self._state_vector += (
                np.dot(self._cov_deviation, self.rng.normal(size=self.analog_filter.N))
                * self.digital_control.clock.T
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
