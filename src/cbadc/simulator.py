"""Analog System and Digital Control Simulator

This module provides simulator tools to simulate the hardware
interaction between an analog system and digital control.
These are mainly intended to produce control signals
:math:`\mathbf{s}[k]` and evaluate state vector trajectories
:math:`\mathbf{x}(t)` for various Analog system
:py:class:`cbadc.analog_system.AnalogSystem` and
:py:class:`cbadc.digital_control.DigitalControl` interactions.
"""
import cbadc.analog_system
import cbadc.digital_control
import cbadc.analog_signal
import numpy as np
import scipy.integrate
import scipy.linalg
import math
from typing import Iterator, Generator, List, Dict, Union


class StateSpaceSimulator(Iterator[np.ndarray]):
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
    Ts : `float`, optional
        specify a sampling rate at which we want to evaluate the systems
        , defaults to :py:class:`digitalControl.Ts`. Note that this Ts must be smaller
        than :py:class:`digitalControl.Ts`.
    t_stop : `float`, optional
        determines a stop time, defaults to :py:obj:`math.inf`
    pre_compute_control_interactions: `bool`, `optional`
        determine if precomputed control interactions should be used, defaults to True.



    Attributes
    ----------
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        the analog system being simulated.
    digital_control : :py:class:`cbadc.digital_control.DigitalControl`
        the digital control being simulated.
    t : `float`
        current time of simulator.
    Ts : `float`
        sample rate of simulation.
    t_stop : `float`
        end time at which the generator raises :py:class:`StopIteration`.
    rtol, atol : `float`, `optional`
        Relative and absolute tolerances. The solver keeps the local error estimates less
        than atol + rtol * abs(y). Effects the underlying solver as described in
        :py:func:`scipy.integrate.solve_ivp`. Default to 1e-3 for rtol and 1e-6 for atol.
    max_step : `float`, `optional`
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver. Effects the underlying solver as
        described in :py:func:`scipy.integrate.solve_ivp`. Defaults to :py:obj:`math.inf`.
    initial_state_vector: `array_like`, shape=(N), `optional`
        initial state vector.
    See also
    --------
    :py:class:`cbadc.analog_signal.AnalogSignal`
    :py:class:`cbadc.analog_system.AnalogSystem`
    :py:class:`cbadc.digital_control.DigitalControl`

    Examples
    --------
    >>> from cbadc.simulator import StateSpaceSimulator
    >>> from cbadc.analog_signal import Sinusodial
    >>> from cbadc.analog_system import AnalogSystem
    >>> from cbadc.digital_control import DigitalControl
    >>> import numpy as np
    >>> A = np.array([[0., 0], [6250., 0.]])
    >>> B = np.array([[6250., 0]]).transpose()
    >>> CT = np.array([[1, 0], [0, 1]])
    >>> Gamma = np.array([[-6250, 0], [0, -6250]])
    >>> Gamma_tildeT = CT
    >>> analog_system = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    >>> digital_control = DigitalControl(1e-6, 2)
    >>> input_signal = Sinusodial(1.0, 250)
    >>> simulator = StateSpaceSimulator(analog_system, digital_control, (input_signal,))
    >>> _ = simulator.__next__()
    >>> _ = simulator.__next__()
    >>> print(np.array(simulator.__next__()))
    [0 0]

    See also
    --------

    Yields
    ------
    `array_like`, shape=(M,), dtype=numpy.int8

    Raises
    ------
    str : unknown

    """

    def __init__(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
        input_signal: List[
            Union[
                cbadc.analog_signal.AnalogSignal,
                cbadc.analog_signal.ConstantSignal,
                cbadc.analog_signal.Sinusodial,
                cbadc.analog_signal.Ramp,
                cbadc.analog_signal.SincPulse,
            ]
        ],
        Ts: float = None,
        t_stop: float = math.inf,
        atol: float = 1e-12,
        rtol: float = 1e-12,
        max_step: float = math.inf,
        initial_state_vector=None,
        pre_compute_control_interactions=True,
        t_init: float = 0.0,
    ):
        if analog_system.L != len(input_signal):
            raise BaseException(
                """The analog system does not have as many inputs as in input
            list"""
            )
        if not np.allclose(analog_system.D, np.zeros_like(analog_system.D)):
            raise BaseException(
                """Can't simulate system with non-zero
                D matrix. Consider chaining systems to remove D."""
            )
        self.analog_system = analog_system
        if isinstance(digital_control, cbadc.digital_control.SwitchedCapacitorControl):
            raise Exception(
                """Switched capacitor control is not compatible
            with this simulator. Instead use
            cbadc.simulator.SwitchedCapacitorStateSpaceSimulator."""
            )
        self.digital_control = digital_control
        self.input_signals = input_signal
        self.t: float = t_init
        self.t_stop = t_stop
        if Ts:
            self.Ts = Ts
        else:
            self.Ts = self.digital_control.T
        if self.Ts > self.digital_control.T:
            raise BaseException(
                f"Simulating with a sample period {self.Ts} that exceeds the control period of the digital control {self.digital_control.T}"
            )
        if initial_state_vector is not None:
            self._state_vector = np.array(initial_state_vector, dtype=np.float64)
            print("initial state vector: ", self._state_vector)
            if (
                self._state_vector.size != self.analog_system.N
                or len(self._state_vector.shape) > 1
            ):
                raise BaseException(
                    "initial_state_vector not single dimension of length N"
                )
        else:
            self._state_vector = np.zeros(self.analog_system.N, dtype=np.double)
        self._temp_state_vector = np.zeros(self.analog_system.N, dtype=np.double)
        self._control_observation = np.zeros(
            self.analog_system.M_tilde, dtype=np.double
        )
        self._input_vector = np.zeros(self.analog_system.L, dtype=np.double)
        self._control_vector = np.zeros(self.analog_system.M, dtype=np.double)
        self._res = np.zeros(self.analog_system.N, dtype=np.double)
        self.atol = atol  # 1e-6
        self.rtol = rtol  # 1e-6
        if max_step > self.Ts:
            self.max_step = self.Ts / 1e-1
        else:
            self.max_step = max_step
        if pre_compute_control_interactions:
            self._pre_computations()
        self._pre_compute_control_interactions = pre_compute_control_interactions
        # self.solve_oder = self._ordinary_differential_solution
        # self.solve_oder = self._full_ordinary_differential_solution

    def state_vector(self) -> np.ndarray:
        """return current analog system state vector :math:`\mathbf{x}(t)`
        evaluated at time :math:`t`.

        Examples
        --------
        >>> from cbadc.simulator import StateSpaceSimulator
        >>> from cbadc.analog_signal import Sinusodial
        >>> from cbadc.analog_system import AnalogSystem
        >>> from cbadc.digital_control import DigitalControl
        >>> import numpy as np
        >>> A = np.array([[0., 0], [6250., 0.]])
        >>> B = np.array([[6250., 0]]).transpose()
        >>> CT = np.array([[1, 0], [0, 1]])
        >>> Gamma = np.array([[-6250, 0], [0, -6250]])
        >>> Gamma_tildeT = CT
        >>> analog_system = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
        >>> digital_control = DigitalControl(1e-6, 2)
        >>> input_signal = Sinusodial(1.0, 250)
        >>> simulator = StateSpaceSimulator(analog_system, digital_control, (input_signal,))
        >>> _ = simulator.__next__()
        >>> _ = simulator.__next__()
        >>> print(np.array(simulator.state_vector()))
        [-0.00623036 -0.00626945]

        Returns
        -------
        `array_like`, shape=(N,)
            returns the state vector :math:`\mathbf{x}(t)`
        """
        return self._state_vector[:]

    def __iter__(self):
        """Use simulator as an iterator"""
        return self

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""
        t_end: float = self.t + self.Ts
        t_span = np.array((self.t, t_end))
        if t_end >= self.t_stop:
            raise StopIteration
        if self._pre_compute_control_interactions:
            self._state_vector = self._ordinary_differential_solution(t_span)
        else:
            self._state_vector = self._full_ordinary_differential_solution(t_span)
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

        # expm(A T_s)
        self._pre_computed_state_transition_matrix = (
            self._analog_system_matrix_exponential(self.Ts)
        )

        def derivative(t, x):
            dac_waveform = np.zeros((self.analog_system.M, self.analog_system.M))
            for m in range(self.analog_system.M):
                dac_waveform[:, m] = self.digital_control.impulse_response(m, t)
            return np.dot(
                np.asarray(self._analog_system_matrix_exponential(self.Ts - t)),
                np.dot(np.asarray(self.analog_system.Gamma), dac_waveform),
            ).flatten()

        tspan = np.array([0, self.Ts])
        atol = 1e-12
        rtol = 1e-12
        max_step = self.Ts * 1e-3

        self._pre_computed_control_matrix = (
            scipy.integrate.solve_ivp(
                derivative,
                tspan,
                np.zeros((self.analog_system.N * self.analog_system.M)),
                atol=atol,
                rtol=rtol,
                max_step=max_step,
            )
            .y[:, -1]
            .reshape((self.analog_system.N, self.analog_system.M), order="C")
        )

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
            the initial time :math:`t_1` and end time :math:`t_2` of the simulation.

        Returns
        -------
        array_like, shape=(N,)
            computed state vector.
        """

        # Compute signal contribution
        def f(t, x):
            return self._signal_derivative(t, x)

        self._temp_state_vector = scipy.integrate.solve_ivp(
            f,
            t_span,
            np.zeros(self.analog_system.N),
            atol=self.atol,
            rtol=self.rtol,
            max_step=self.max_step,
        ).y[:, -1]

        self._temp_state_vector += np.dot(
            self._pre_computed_state_transition_matrix, self._state_vector
        ).flatten()

        # Compute control observation s_tilde(t)
        self._control_observation = np.dot(
            self.analog_system.Gamma_tildeT, self._state_vector
        )

        # update control at time t_span[0]
        self._char_control_vector = self.digital_control.control_contribution(
            t_span[0], self._control_observation
        )
        self._temp_state_vector += np.dot(
            self._pre_computed_control_matrix, self._char_control_vector
        ).flatten()
        return self._temp_state_vector

    def _signal_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        res = np.dot(self.analog_system.A, x)
        for _l in range(self.analog_system.L):
            res += np.dot(
                self.analog_system.B[:, _l], self.input_signals[_l].evaluate(t)
            )
        return res.flatten()

    def _full_ordinary_differential_solution(self, t_span: np.ndarray) -> np.ndarray:
        return scipy.integrate.solve_ivp(
            lambda t, x: self._ordinary_differentail_function(t, x),
            t_span,
            self._state_vector,
            atol=self.atol,
            rtol=self.rtol,
            max_step=self.max_step,
        ).y[:, -1]

    def _ordinary_differentail_function(self, t: float, y: np.ndarray):
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
        for _l in range(self.analog_system.L):
            self._input_vector[_l] = self.input_signals[_l].evaluate(t)
        self._temp_state_vector = np.dot(self.analog_system.Gamma_tildeT, y)
        self._control_vector = self.digital_control.control_contribution(
            t, self._temp_state_vector
        )
        return np.asarray(
            self.analog_system.derivative(
                y, t, self._input_vector, self._control_vector
            )
        ).flatten()

    def __str__(self):
        return f"t = {self.t}, (current simulator time)\nTs = {self.Ts},\nt_stop = {self.t_stop},\nrtol = {self.rtol},\natol = {self.atol}, and\nmax_step = {self.max_step}\n"


def extended_simulation_result(
    simulator: StateSpaceSimulator,
) -> Generator[Dict[str, np.ndarray], None, None]:
    """Extended simulation output

    Used to also pass the state vector from a
    simulator generator.

    Parameters
    ----------
    simulator : :py:class:`cbadc.simulator.StateSpaceSimulator`
        a iterable simulator instance.

    Yields
    ------
    { 'control_signal', 'analog_state' } : { (array_like, shape=(M,)), (array_like, shape=(N,)) }
        an extended output including the analog state vector.
    """
    for control_signal in simulator:
        analog_state = simulator.state_vector()
        yield {
            "control_signal": np.array(control_signal),
            "analog_state": np.array(analog_state),
        }


class SwitchedCapacitorStateSpaceSimulator(Iterator[np.ndarray]):
    """Simulate the analog system and digital control interactions
    in the presence on analog signals.

    Parameters
    ----------
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        the analog system
    digital_control: :py:class:`cbadc.digital_control.SwitchedCapacitorControl`
        the digital control
    input_signals : [:py:class:`cbadc.analog_signal.AnalogSignal`]
        a python list of analog signals (or a derived class)
    Ts : `float`, optional
        specify a sampling rate at which we want to evaluate the systems
        , defaults to :py:class:`digitalControl.Ts`. Note that this Ts must be smaller
        than :py:class:`digitalControl.Ts`.
    t_stop : `float`, optional
        determines a stop time, defaults to :py:obj:`math.inf`



    Attributes
    ----------
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        the analog system being simulated.
    digital_control : :py:class:`cbadc.digital_control.SwitchedCapacitorControl`
        the digital control being simulated.
    t : `float`
        current time of simulator.
    Ts : `float`
        sample rate of simulation.
    t_stop : `float`
        end time at which the generator raises :py:class:`StopIteration`.
    rtol, atol : `float`, `optional`
        Relative and absolute tolerances. The solver keeps the local error estimates less
        than atol + rtol * abs(y). Effects the underlying solver as described in
        :py:func:`scipy.integrate.solve_ivp`. Default to 1e-3 for rtol and 1e-6 for atol.
    max_step : `float`, `optional`
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver. Effects the underlying solver as
        described in :py:func:`scipy.integrate.solve_ivp`. Defaults to :py:obj:`math.inf`.
    initial_state_vector: `array_like`, shape=(N), `optional`
        initial state vector.
    t_init: `float`, `optional`
        initial time of simulator, defaults to 0.

    Yields
    ------
    `array_like`, shape=(M,), dtype=numpy.int8

    Raises
    ------
    str : unknown

    """

    def __init__(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.SwitchedCapacitorControl,
        input_signal: List[
            Union[
                cbadc.analog_signal.AnalogSignal,
                cbadc.analog_signal.ConstantSignal,
                cbadc.analog_signal.Sinusodial,
                cbadc.analog_signal.Ramp,
                cbadc.analog_signal.SincPulse,
            ]
        ],
        Ts: float = None,
        t_stop: float = math.inf,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        steps=10,
        initial_state_vector=None,
        t_init: float = 0.0,
    ):
        if analog_system.L != len(input_signal):
            raise BaseException(
                """The analog system does not have as many inputs as in input
            list"""
            )
        if not np.allclose(analog_system.D, np.zeros_like(analog_system.D)):
            raise BaseException(
                """Can't simulate system with non-zero
                D matrix. Consider chaining systems to remove D."""
            )
        self.analog_system = analog_system
        if not isinstance(
            digital_control, cbadc.digital_control.SwitchedCapacitorControl
        ):
            raise Exception(
                """This simulator is only compatible with
                cbadc.digital_control.SwitchedCapacitorControl"""
            )
        self.digital_control = digital_control
        self.input_signals = input_signal
        self.t: float = t_init
        self.t_stop = t_stop
        if Ts:
            self.Ts = Ts
        else:
            self.Ts = self.digital_control.T
        if self.Ts > self.digital_control.T:
            raise BaseException(
                f"Simulating with a sample period {self.Ts} that exceeds the control period of the digital control {self.digital_control.T}"
            )

        self._control_observation = np.zeros(
            self.analog_system.M_tilde, dtype=np.double
        )
        self.atol = atol  # 1e-6
        self.rtol = rtol  # 1e-6
        self.steps = steps

        self.N = self.analog_system.N + self.analog_system.M

        self.A = scipy.linalg.block_diag(self.analog_system.A, self.digital_control.A)

        self.B = np.zeros((self.N, self.analog_system.L))
        self.B[: analog_system.N, :] = self.analog_system.B

        self.CT = np.hstack(
            (
                self.analog_system.CT,
                np.zeros((self.analog_system.N_tilde, self.digital_control.M)),
            )
        )

        self.Gamma_tildeT = np.hstack(
            (
                self.analog_system.Gamma_tildeT,
                np.zeros((self.analog_system.M_tilde, self.digital_control.M)),
            )
        )

        self._state_vector = np.zeros(self.N, dtype=np.double)
        if initial_state_vector is not None:
            self._state_vector[: self.analog_system.N] = np.array(
                initial_state_vector, dtype=np.double
            )
            print("initial state vector: ", self._state_vector)
            if self._state_vector.size != self.N or len(self._state_vector.shape) > 1:
                raise BaseException(
                    "initial_state_vector not single dimension of length N"
                )

        self._input_vector = np.zeros(self.analog_system.L, dtype=np.double)
        self._control_vector = np.zeros(self.analog_system.M, dtype=np.double)
        self._res = np.zeros(self.N, dtype=np.double)

    def state_vector(self) -> np.ndarray:
        """return current analog system state vector :math:`\mathbf{x}(t)`
        evaluated at time :math:`t`.

        Returns
        -------
        `array_like`, shape=(N + M,)
            returns the state vector :math:`\mathbf{x}(t)`
        """
        return self._state_vector[:]

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""
        t_end: float = self.t + self.Ts
        t_span = np.array((self.t, t_end))
        if t_end >= self.t_stop:
            raise StopIteration
        self._state_vector = self._solve_ivp(t_span, self._state_vector)
        self.t = t_end
        return self.digital_control.control_signal()

    def _solve_ivp(self, t_span, x):
        # Define derivative
        _x = x
        _t = t_span[0]
        while _t < t_span[1]:

            next_update = self.digital_control.next_update()
            t_next = min((t_span[1], next_update))
            # If time at next update then update controls
            if _t == next_update:
                s_tilde = np.dot(self.Gamma_tildeT, _x)
                phase, reset, s = self.digital_control.control_update(_t, s_tilde)
                if reset.any():
                    self._control_update(phase, reset, s, _x)
            # Otherwise, continoue solve diff
            else:

                def system_derivative(t: float, x: np.ndarray):
                    res = np.dot(self.A, x)
                    for _l in range(self.analog_system.L):
                        res += np.dot(self.B[:, _l], self.input_signals[_l].evaluate(t))
                    return res

                _x = scipy.integrate.solve_ivp(
                    system_derivative,
                    (_t, t_next),
                    _x,
                    atol=self.atol,
                    rtol=self.rtol,
                    max_step=(t_next - _t) / self.steps,
                    # method="DOP853",
                    method="RK45",
                ).y[:, -1]
                _t = t_next
        return _x

    def _control_update(
        self,
        phase: np.ndarray,
        reset: np.ndarray,
        s: np.ndarray,
        x: np.ndarray,
    ):
        N = self.analog_system.N
        for m in range(self.digital_control.M):
            if reset[m]:
                pos = N + m
                # Phase 1
                if phase[m] == 1:
                    self.A[:N, pos] = np.zeros(N)
                    x[pos] = 0
                # Phase 0
                else:
                    self.A[:N, pos] = self.analog_system.Gamma[:, m]
                    if s[m] == 1:
                        x[pos] = self.digital_control.VCap
                    else:
                        x[pos] = -self.digital_control.VCap
