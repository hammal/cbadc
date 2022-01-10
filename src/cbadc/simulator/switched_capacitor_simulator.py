# from .simulator import _Simulator


# class SwitchedCapacitorStateSpaceSimulator(_Simulator):
#     """Simulate the analog system and digital control interactions
#     in the presence on analog signals.

#     Parameters
#     ----------
#     analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
#         the analog system
#     digital_control: :py:class:`cbadc.digital_control.SwitchedCapacitorControl`
#         the digital control
#     input_signals : [:py:class:`cbadc.analog_signal.AnalogSignal`]
#         a python list of analog signals (or a derived class)
#     Ts : `float`, optional
#         specify a sampling rate at which we want to evaluate the systems
#         , defaults to :py:class:`digitalControl.Ts`. Note that this Ts must be smaller
#         than :py:class:`digitalControl.Ts`.
#     t_stop : `float`, optional
#         determines a stop time, defaults to :py:obj:`math.inf`


#     Attributes
#     ----------
#     analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
#         the analog system being simulated.
#     digital_control : :py:class:`cbadc.digital_control.SwitchedCapacitorControl`
#         the digital control being simulated.
#     t : `float`
#         current time of simulator.
#     Ts : `float`
#         sample rate of simulation.
#     t_stop : `float`
#         end time at which the generator raises :py:class:`StopIteration`.
#     rtol, atol : `float`, `optional`
#         Relative and absolute tolerances. The solver keeps the local error estimates less
#         than atol + rtol * abs(y). Effects the underlying solver as described in
#         :py:func:`scipy.integrate.solve_ivp`. Default to 1e-3 for rtol and 1e-6 for atol.
#     max_step : `float`, `optional`
#         Maximum allowed step size. Default is np.inf, i.e., the step size is not
#         bounded and determined solely by the solver. Effects the underlying solver as
#         described in :py:func:`scipy.integrate.solve_ivp`. Defaults to :py:obj:`math.inf`.
#     initial_state_vector: `array_like`, shape=(N), `optional`
#         initial state vector.
#     t_init: `float`, `optional`
#         initial time of simulator, defaults to 0.

#     Yields
#     ------
#     `array_like`, shape=(M,), dtype=numpy.int8

#     Raises
#     ------
#     str : unknown

#     """

#     def __init__(
#         self,
#         analog_system: cbadc.analog_system.AnalogSystem,
#         digital_control: cbadc.digital_control.SwitchedCapacitorControl,
#         input_signal: List[
#             Union[
#                 cbadc.analog_signal.AnalogSignal,
#                 cbadc.analog_signal.ConstantSignal,
#                 cbadc.analog_signal.Sinusoidal,
#                 cbadc.analog_signal.Ramp,
#                 cbadc.analog_signal.SincPulse,
#             ]
#         ],
#         Ts: float = None,
#         t_stop: float = math.inf,
#         atol: float = 1e-8,
#         rtol: float = 1e-6,
#         steps=10,
#         initial_state_vector=None,
#         t_init: float = 0.0,
#     ):
#         if analog_system.L != len(input_signal):
#             raise Exception(
#                 """The analog system does not have as many inputs as in input
#             list"""
#             )
#         if not np.allclose(analog_system.D, np.zeros_like(analog_system.D)):
#             raise Exception(
#                 """Can't simulate system with non-zero
#                 D matrix. Consider chaining systems to remove D."""
#             )
#         self.analog_system = analog_system
#         if not isinstance(
#             digital_control, cbadc.digital_control.SwitchedCapacitorControl
#         ):
#             raise Exception(
#                 """This simulator is only compatible with
#                 cbadc.digital_control.SwitchedCapacitorControl"""
#             )
#         self.digital_control = digital_control
#         self.input_signals = input_signal
#         self.t: float = t_init
#         self.t_stop = t_stop
#         if Ts:
#             self.Ts = Ts
#         else:
#             self.Ts = self.digital_control.T
#         if self.Ts > self.digital_control.T:
#             raise Exception(
#                 f"Simulating with a sample period {self.Ts} that exceeds the control period of the digital control {self.digital_control.T}"
#             )

#         self._control_observation = np.zeros(
#             self.analog_system.M_tilde, dtype=np.double
#         )
#         self.atol = atol  # 1e-6
#         self.rtol = rtol  # 1e-6
#         self.steps = steps

#         self.N = self.analog_system.N + self.analog_system.M

#         self.A = scipy.linalg.block_diag(
#             self.analog_system.A, self.digital_control.A)

#         self.B = np.zeros((self.N, self.analog_system.L))
#         self.B[: analog_system.N, :] = self.analog_system.B

#         self.CT = np.hstack(
#             (
#                 self.analog_system.CT,
#                 np.zeros((self.analog_system.N_tilde, self.digital_control.M)),
#             )
#         )

#         self.Gamma_tildeT = np.hstack(
#             (
#                 self.analog_system.Gamma_tildeT,
#                 np.zeros((self.analog_system.M_tilde, self.digital_control.M)),
#             )
#         )

#         self._state_vector = np.zeros(self.N, dtype=np.double)
#         if initial_state_vector is not None:
#             self._state_vector[: self.analog_system.N] = np.array(
#                 initial_state_vector, dtype=np.double
#             )
#             print("initial state vector: ", self._state_vector)
#             if self._state_vector.size != self.N or len(self._state_vector.shape) > 1:
#                 raise Exception(
#                     "initial_state_vector not single dimension of length N"
#                 )

#         self._input_vector = np.zeros(self.analog_system.L, dtype=np.double)
#         self._control_vector = np.zeros(self.analog_system.M, dtype=np.double)
#         self._res = np.zeros(self.N, dtype=np.double)

#     def state_vector(self) -> np.ndarray:
#         """return current analog system state vector :math:`\mathbf{x}(t)`
#         evaluated at time :math:`t`.

#         Returns
#         -------
#         `array_like`, shape=(N + M,)
#             returns the state vector :math:`\mathbf{x}(t)`
#         """
#         return self._state_vector[:]

#     def __next__(self) -> np.ndarray:
#         """Computes the next control signal :math:`\mathbf{s}[k]`"""
#         t_end: float = self.t + self.Ts
#         t_span = np.array((self.t, t_end))
#         if t_end >= self.t_stop:
#             raise StopIteration
#         self._state_vector = self._solve_ivp(t_span, self._state_vector)
#         self.t = t_end
#         return self.digital_control.control_signal()

#     def _solve_ivp(self, t_span, x):
#         # Define derivative
#         _x = x
#         _t = t_span[0]
#         while _t < t_span[1]:

#             next_update = self.digital_control.next_update()
#             t_next = min((t_span[1], next_update))
#             # If time at next update then update controls
#             if _t == next_update:
#                 s_tilde = np.dot(self.Gamma_tildeT, _x)
#                 phase, reset, s = self.digital_control.control_update(
#                     _t, s_tilde)
#                 if reset.any():
#                     self._control_update(phase, reset, s, _x)
#             # Otherwise, continoue solve diff
#             else:

#                 def system_derivative(t: float, x: np.ndarray):
#                     res = np.dot(self.A, x)
#                     for _l in range(self.analog_system.L):
#                         res += np.dot(self.B[:, _l],
#                                       self.input_signals[_l].evaluate(t))
#                     return res

#                 _x = scipy.integrate.solve_ivp(
#                     system_derivative,
#                     (_t, t_next),
#                     _x,
#                     atol=self.atol,
#                     rtol=self.rtol,
#                     max_step=(t_next - _t) / self.steps,
#                     # method="DOP853",
#                     method="RK45",
#                 ).y[:, -1]
#                 _t = t_next
#         return _x

#     def _control_update(
#         self,
#         phase: np.ndarray,
#         reset: np.ndarray,
#         s: np.ndarray,
#         x: np.ndarray,
#     ):
#         N = self.analog_system.N
#         for m in range(self.digital_control.M):
#             if reset[m]:
#                 pos = N + m
#                 # Phase 1
#                 if phase[m] == 1:
#                     self.A[:N, pos] = np.zeros(N)
#                     x[pos] = 0
#                 # Phase 0
#                 else:
#                     self.A[:N, pos] = self.analog_system.Gamma[:, m]
#                     if s[m] == 1:
#                         x[pos] = self.digital_control.VCap
#                     else:
#                         x[pos] = -self.digital_control.VCap
