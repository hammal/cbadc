from typing import Union
import cbadc
import logging
import os
import scipy.integrate
import numpy as np
from .digital_estimator import DigitalEstimator
from ._filter_coefficients import FilterComputationBackend

logger = logging.getLogger(__name__)


class FIRFilter(DigitalEstimator):
    """FIR filter implementation of the digital estimator.

    Specifically, the FIR filter estimator estimates a filtered version :math:`\hat{\mathbf{u}}(t)` (shaped by
    :py:func:`signal_transfer_function`) of the
    input signal :math:`\mathbf{u}(t)` from a sequence of control signals :math:`\mathbf{s}[k]`.

    Specifically, the estimate is of the form

    :math:`\hat{\mathbf{u}}(k T) = \hat{\mathbf{u}}_0 + \sum_{\ell=-K_1}^{K_2} \mathbf{h}[\ell] \mathbf{s}[k + \ell]`

    where

    :math:`\mathbf{h}[\ell]=\\begin{cases}\mathbf{W}^{\mathsf{T}} \mathbf{A}_b^\ell \mathbf{B}_b & \mathrm{if} \, \ell \geq 0 \\\  -\mathbf{W}^{\mathsf{T}} \mathbf{A}_f^{-\ell + 1} \mathbf{B}_f & \mathrm{else} \\end{cases}`

    and :math:`\mathbf{W}^{\mathsf{T}}`, :math:`\mathbf{A}_b`,
    :math:`\mathbf{B}_b`, :math:`\mathbf{A}_f`, and :math:`\mathbf{B}_f`
    are computed based on the analog system, the sample period :math:`T_s`, and the
    digital control's DAC waveform as described in
    `control-bounded converters <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf?sequence=1&isAllowed#=ypage=67/>`_.

    Parameters
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system (necessary to compute the estimators filter coefficients).
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        a digital control (necessary to determine the corresponding DAC waveform).
    eta2: `float`
        the :math:`\eta^2` parameter determines the bandwidth of the estimator.
    K1: `int`
        The lookback size
    K2: `int`, `optional`
        lookahead size, defaults to 0.
    stop_after_number_of_iterations: `int`
        determine a max number of iterations by the iterator, defaults to  :math:`2^{63}`.
    Ts: `float`, `optional`
        the sampling time, defaults to the time period of the digital control.
    mid_point: `bool`, `optional`
        set samples in between control updates, i.e., :math:`\hat{u}(kT + T/2)`, defaults to False.
    downsample: `int`, `optional`
        specify down sampling rate in relation to the control period :math:`T`, defaults to 1, i.e.,
        no down sampling.
    offset: `array_like`, shape=(L), `optional`
        the estimate offset :math:`\hat{\mathbf{u}}_0`, defaults to a zero vector.
    fixed_point: :py:class:`cbadc.utilities.FixedPoint`, `optional`
        fixed point arithmetic configuration, defaults to None.
    solver_type: :py:class:`cbadc.digital_estimator._filter_coefficients.FilterComputationBackend`
        determine which solver type to use when computing filter coefficients.

    Attributes
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        analog system as in :py:class:`cbadc.analog_system.AnalogSystem` or from
        derived class.
    eta2: float
        eta2, or equivalently :math:`\eta^2`, sets the bandwidth of the estimator.
    control_signal: :py:class:`cbadc.digital_control.DigitalControl`
        a iterator suppling control signals as :py:class:`cbadc.digital_control.DigitalControl`.
    number_of_iterations: `int`
        number of iterations until iterator raises :py:class:`StopIteration`.
    K1: `int`
        number of samples, prior to estimate, used in estimate
    K2: `int`
        number of lookahead samples per computed batch.
    Ts: `float`
        spacing between samples in seconds.
    mid_point: `bool`
        estimated samples shifted in between control updates, i.e., :math:`\hat{u}(kT + T/2)`.
    downsample: `int`, `optional`
        down sampling rate in relation to the control period :math:`T`.
    Af: `array_like`, shape=(N, N)
        The Af matrix
    Ab: `array_like`, shape=(N, N)
        The Ab matrix
    Bf: `array_like`, shape=(N, M)
        The Bf matrix
    Bb: `array_like`, shape=(N, M)
        The Bb matrix
    WT: `array_like`, shape=(L, N)
        The W matrix transposed
    h: `array_like`, shape=(L, K1 + K2, M)
        filter impulse response
    offset: `array_like`, shape=(L)
        the estimate offset :math:`\hat{\mathbf{u}}_0`.
    fixed_point: `bool`
        using fixed point?
    solver_type: :py:class:`cbadc.digital_estimator._filter_coefficients.FilterComputationBackend`
        The solver used for computing the filter coefficients.


    Yields
    ------
    `array_like`, shape=(L,)
        an input estimate sample :math:`\hat{\mathbf{u}}(t)`
    """

    def __init__(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
        eta2: float,
        K1: int,
        K2: int,
        stop_after_number_of_iterations: int = (1 << 63),
        Ts: float = None,
        mid_point: bool = False,
        downsample: int = 1,
        offset: np.ndarray = None,
        fixed_point: cbadc.utilities.FixedPoint = None,
        solver_type: FilterComputationBackend = FilterComputationBackend.numpy,
    ):
        """Initializes filter coefficients"""
        if K1 < 0:
            raise Exception("K1 must be non negative integer.")
        self.K1 = K1
        if K2 < 1:
            raise Exception("K2 must be a positive integer.")
        self.K2 = K2
        self.K3 = K1 + K2
        self._filter_lag = self.K2 - 2
        self.analog_system = analog_system
        self.digital_control = digital_control
        if eta2 < 0.0:
            raise Exception("eta2 must be non negative.")
        self.eta2 = eta2
        self.control_signal = None
        self.number_of_iterations = stop_after_number_of_iterations
        self._iteration = 0
        if Ts:
            self.Ts = Ts
        else:
            self.Ts = digital_control.clock.T
        if mid_point:
            raise NotImplementedError("Planned for v.0.1.0")
        self.mid_point = mid_point
        self.downsample = int(downsample)
        self._temp_controls = np.zeros(
            (self.downsample, self.analog_system.M), dtype=np.int8
        )

        if offset is not None:
            self.offset = np.array(offset, dtype=np.float64)
            if self.offset.size != self.analog_system.L:
                raise Exception("offset is not of size L")
        else:
            self.offset = np.zeros(self.analog_system.L, dtype=np.float64)

        if fixed_point is not None:
            self.fixed_point = True
            self.__fixed_point = fixed_point
            self.__fixed_to_float = np.vectorize(self.__fixed_point.fixed_to_float)
            self.__float_to_fixed = np.vectorize(self.__fixed_point.float_to_fixed)
        else:
            self.fixed_point = False

        # For transfer functions
        self.eta2Matrix = np.eye(self.analog_system.CT.shape[0]) * self.eta2
        # Compute filter coefficients
        self.solver_type = solver_type
        self._compute_filter_coefficients(analog_system, digital_control, eta2)

        # Initialize filter.
        if self.fixed_point:
            self.h = np.zeros(
                (self.analog_system.L, self.K3, self.analog_system.M), dtype=np.int64
            )
        else:
            self.h = np.zeros(
                (self.analog_system.L, self.K3, self.analog_system.M), dtype=np.double
            )
        # Compute lookback.
        temp1 = np.copy(self.Bf)
        for k1 in range(self.K1 - 1, -1, -1):
            if self.fixed_point:
                self.h[:, k1, :] = self.__float_to_fixed(-np.dot(self.WT, temp1))
            else:
                self.h[:, k1, :] = -np.dot(self.WT, temp1)
            temp1 = np.dot(self.Af, temp1)

        # Compute lookahead.
        temp2 = np.copy(self.Bb)
        for k2 in range(self.K1, self.K3):
            if self.fixed_point:
                self.h[:, k2, :] = self.__float_to_fixed(np.dot(self.WT, temp2))
            else:
                self.h[:, k2, :] = np.dot(self.WT, temp2)
            temp2 = np.dot(self.Ab, temp2)
        self._control_signal_valued = np.zeros(
            (self.K3, self.analog_system.M), dtype=np.int8
        )

    def __iter__(self):
        return self

    # def _compute_filter_coefficients(
    #     self,
    #     analog_system: cbadc.analog_system.AnalogSystem,
    #     digital_control: cbadc.digital_control.DigitalControl,
    #     eta2: float,
    # ):
    #     logger.info("Computing filter coefficients.")
    #     atol = 1e-15
    #     rtol = 2.3e-13
    #     steps = 1000
    #     max_step = self.Ts / steps

    #     # Compute filter coefficients
    #     A = np.array(analog_system.A).transpose()
    #     B = np.array(analog_system.CT).transpose()
    #     Q = np.dot(np.array(analog_system.B),
    #                np.array(analog_system.B).transpose())
    #     R = eta2 * np.eye(analog_system.N_tilde)
    #     # Solve care
    #     Vf = care(A, B, Q, R)
    #     # tau = 1e-6
    #     # rtol = 1e-150
    #     # atol = 1e-350
    #     # Vf = bruteForceCare(A, B, Q, R, tau=tau, rtol=rtol, atol=atol)
    #     Vb = care(-A, B, Q, R)
    #     # Vb = bruteForceCare(-A, B, Q, R, tau=tau, rtol=rtol, atol=atol)

    #     CCT: np.ndarray = np.dot(
    #         np.array(analog_system.CT).transpose(), np.array(analog_system.CT)
    #     )
    #     tempAf: np.ndarray = analog_system.A - np.dot(Vf, CCT) / eta2
    #     tempAb: np.ndarray = analog_system.A + np.dot(Vb, CCT) / eta2
    #     self.Af: np.ndarray = np.asarray(scipy.linalg.expm(tempAf * self.Ts))
    #     self.Ab: np.ndarray = np.asarray(scipy.linalg.expm(-tempAb * self.Ts))
    #     # Solve IVPs
    #     self.Bf: np.ndarray = np.zeros(
    #         (self.analog_system.N, self.analog_system.M))
    #     self.Bb: np.ndarray = np.zeros(
    #         (self.analog_system.N, self.analog_system.M))

    #     self._Bf_2T = np.zeros_like(self.Bf)

    #     if self.mid_point:
    #         for m in range(self.analog_system.M):

    #             def _derivative_forward(t, x):
    #                 return np.dot(tempAf, x) + np.dot(
    #                     self.analog_system.Gamma, digital_control.impulse_response(
    #                         m, t)
    #                 )

    #             solBf = scipy.integrate.solve_ivp(
    #                 _derivative_forward,
    #                 (0, self.Ts / 2.0),
    #                 np.zeros(self.analog_system.N),
    #                 atol=atol,
    #                 rtol=rtol,
    #                 max_step=max_step,
    #                 t_eval=np.linspace(0, self.Ts, steps),
    #                 # method="Radau",
    #             ).y[:, -1]

    #             def _derivative_backward(t, x):
    #                 return -np.dot(tempAb, x) - np.dot(
    #                     self.analog_system.Gamma, digital_control.impulse_response(
    #                         m, t)
    #                 )

    #             solBb = scipy.integrate.solve_ivp(
    #                 _derivative_backward,
    #                 (0, self.Ts / 2.0),
    #                 np.zeros(self.analog_system.N),
    #                 atol=atol,
    #                 rtol=rtol,
    #                 max_step=max_step,
    #                 t_eval=np.linspace(0, self.Ts, steps),
    #                 # method="Radau",
    #             ).y[:, -1]
    #             for n in range(self.analog_system.N):
    #                 self.Bf[n, m] = solBf[n]
    #                 self.Bb[n, m] = solBb[n]
    #         self.Bf = np.dot(
    #             np.eye(self.analog_system.N)
    #             + scipy.linalg.expm(tempAf * self.Ts / 2.0),
    #             self.Bf,
    #         )
    #         self.Bb = np.dot(
    #             np.eye(self.analog_system.N)
    #             + scipy.linalg.expm(tempAb * self.Ts / 2.0),
    #             self.Bb,
    #         )
    #     # elif isinstance(
    #     #     self.digital_control, cbadc.digital_control.SwitchedCapacitorControl
    #     # ):
    #     #     for m in range(self.analog_system.M):
    #     #         N = self.analog_system.N
    #     #         tempStateSpaceMatrix = np.zeros((N + 1, N + 1))

    #     #         # Forward Coefficients
    #     #         tempStateSpaceMatrix[:N, :N] = tempAf
    #     #         tempStateSpaceMatrix[N, N] = self.digital_control.A[m, m]
    #     #         tempStateSpaceMatrix[:N, N] = self.analog_system.Gamma[:, m]

    #     #         def _derivative_forward(t, x):
    #     #             return (
    #     #                 np.dot(tempStateSpaceMatrix, x).flatten()
    #     #                 # + np.random.randn(N + 1) * 1e-3
    #     #             )

    #     #         # print(
    #     #         #     f"m={m}, T1={self.digital_control.clock.T1[m]}, T2={self.digital_control.clock.T2[m]}"
    #     #         # )

    #     #         initial_state = np.zeros(N + 1)
    #     #         initial_state[N] = self.digital_control.VCap
    #     #         # First phase (discarging capacitor)
    #     #         if self.digital_control.clock.T2[m] > self.digital_control.clock.T:
    #     #             T2LT = True
    #     #             t_eval1 = np.array(
    #     #                 [
    #     #                     self.digital_control.clock.T - self.digital_control.clock.T1[m],
    #     #                     self.digital_control.clock.T2[m] - self.digital_control.clock.T1[m],
    #     #                 ]
    #     #             )
    #     #             tspan1 = (0, t_eval1[-1])
    #     #             tspace1 = (
    #     #                 min(
    #     #                     (
    #     #                         (
    #     #                             self.digital_control.clock.T2[m]
    #     #                             - self.digital_control.clock.T1[m]
    #     #                         ),
    #     #                         (self.digital_control.clock.T - self.digital_control.clock.T1[m]),
    #     #                     )
    #     #                 )
    #     #                 / steps
    #     #             )
    #     #             t_eval2 = np.array(
    #     #                 [0, 2 * self.digital_control.clock.T - self.digital_control.clock.T2[m]]
    #     #             )
    #     #             tspan2 = (0, t_eval2[-1])
    #     #             tspace2 = (
    #     #                 2 * self.digital_control.clock.T - self.digital_control.clock.T2[m]
    #     #             ) / steps
    #     #         else:
    #     #             T2LT = False
    #     #             t_eval1 = np.array(
    #     #                 [0, self.digital_control.clock.T2[m] - self.digital_control.clock.T1[m]]
    #     #             )
    #     #             tspan1 = (0, t_eval1[-1])
    #     #             tspace1 = (
    #     #                 self.digital_control.clock.T2[m] - self.digital_control.clock.T1[m]
    #     #             ) / steps
    #     #             t_eval2 = np.array(
    #     #                 [
    #     #                     self.digital_control.clock.T - self.digital_control.clock.T2[m],
    #     #                     2 * self.digital_control.clock.T - self.digital_control.clock.T2[m],
    #     #                 ]
    #     #             )
    #     #             tspan2 = (0, t_eval2[-1])
    #     #             if self.digital_control.clock.T == self.digital_control.clock.T2[m]:
    #     #                 tspace2 = (
    #     #                     2 * self.digital_control.clock.T - self.digital_control.clock.T2[m]
    #     #                 ) / steps
    #     #             else:
    #     #                 tspace2 = (
    #     #                     self.digital_control.clock.T - self.digital_control.clock.T2[m]
    #     #                 ) / steps

    #     #         # print(f"T2 > T = {T2LT}")

    #     #         odeDOP853 = scipy.integrate.DOP853(
    #     #             _derivative_forward,
    #     #             0,
    #     #             initial_state,
    #     #             t_bound=tspan1[1],
    #     #             first_step=tspace1,
    #     #             max_step=tspace1,
    #     #             rtol=rtol,
    #     #             atol=atol,
    #     #             vectorized=False,
    #     #         )

    #     #         sol1 = []
    #     #         for t in t_eval1:
    #     #             while odeDOP853.t < t:
    #     #                 odeDOP853.step()
    #     #             sol1.append(odeDOP853.y)

    #     #         empty_cap_at_T2 = sol1[-1][N]
    #     #         initial_state = sol1[-1][:]
    #     #         initial_state[N] = 0

    #     #         odeDOP853 = scipy.integrate.DOP853(
    #     #             _derivative_forward,
    #     #             0,
    #     #             initial_state,
    #     #             t_bound=tspan2[1],
    #     #             first_step=tspace2,
    #     #             max_step=tspace2,
    #     #             rtol=rtol,
    #     #             atol=atol,
    #     #             vectorized=False,
    #     #         )

    #     #         sol2 = []
    #     #         for t in t_eval2:
    #     #             while odeDOP853.t < t:
    #     #                 odeDOP853.step()
    #     #             sol2.append(odeDOP853.y)

    #     #         # T2 > T
    #     #         if T2LT:
    #     #             self.Bf[:, m] = sol1[0][:N]
    #     #             self._Bf_2T[:, m] = sol2[1][:N]
    #     #         else:
    #     #             self.Bf[:, m] = sol2[0][:N]
    #     #             self._Bf_2T[:, m] = sol2[1][:N]

    #     #         # Backward Coefficients
    #     #         tempStateSpaceMatrix[:N, :N] = -tempAb
    #     #         tempStateSpaceMatrix[N, N] = -self.digital_control.A[m, m]
    #     #         tempStateSpaceMatrix[:N, N] = -self.analog_system.Gamma[:, m]

    #     #         def _derivative_backward(t, x):
    #     #             return (
    #     #                 np.dot(tempStateSpaceMatrix, x).flatten()
    #     #                 # + np.random.randn(N + 1) * 1e-3
    #     #             )

    #     #         # First Phase
    #     #         initial_state = np.zeros(N + 1)
    #     #         initial_state[N] = empty_cap_at_T2
    #     #         # print(
    #     #         #     f"initial_state = {initial_state}, empty_cap_at_T2 = {empty_cap_at_T2}"
    #     #         # )
    #     #         t_eval1 = np.array(
    #     #             [0, self.digital_control.clock.T2[m] - self.digital_control.clock.T1[m]]
    #     #         )
    #     #         max_step = t_eval1[1] / steps

    #     #         odeDOP853 = scipy.integrate.DOP853(
    #     #             _derivative_backward,
    #     #             0,
    #     #             initial_state,
    #     #             t_bound=t_eval1[1],
    #     #             first_step=max_step,
    #     #             max_step=max_step,
    #     #             rtol=rtol,
    #     #             atol=atol,
    #     #             vectorized=False,
    #     #         )

    #     #         sol1 = []
    #     #         for t in t_eval1:
    #     #             while odeDOP853.t < t:
    #     #                 odeDOP853.step()
    #     #             sol1.append(odeDOP853.y)

    #     #         if self.digital_control.clock.T1[m] > 0:
    #     #             initial_state = sol1[1][:]
    #     #             initial_state[N] = 0.0
    #     #             t_eval2 = np.array([0, self.digital_control.clock.T1[m]])
    #     #             max_step = t_eval2[1] / steps

    #     #             odeDOP853 = scipy.integrate.DOP853(
    #     #                 _derivative_backward,
    #     #                 0,
    #     #                 initial_state,
    #     #                 t_bound=t_eval2[1],
    #     #                 first_step=max_step,
    #     #                 max_step=max_step,
    #     #                 rtol=rtol,
    #     #                 atol=atol,
    #     #                 vectorized=False,
    #     #             )

    #     #             sol2 = []
    #     #             for t in t_eval2:
    #     #                 while odeDOP853.t < t:
    #     #                     odeDOP853.step()
    #     #                 sol2.append(odeDOP853.y)
    #     #             self.Bb[:, m] = sol2[1][:N]
    #     #         else:
    #     #             self.Bb[:, m] = sol1[1][:N]
    #     else:
    #         for m in range(self.analog_system.M):

    #             def _derivative_forward_2(t, x):
    #                 return np.dot(tempAf, x) + np.dot(
    #                     self.analog_system.Gamma, digital_control.impulse_response(
    #                         m, t)
    #                 )

    #             def impulse_start(t, x):
    #                 return t - self.digital_control._impulse_response[m].t0

    #             # impulse_start.direction = -1.0

    #             solver = scipy.integrate.solve_ivp(
    #                 _derivative_forward_2,
    #                 (self.digital_control._impulse_response[m].t0, self.Ts),
    #                 np.zeros(self.analog_system.N),
    #                 atol=atol,
    #                 rtol=rtol,
    #                 method="RK45",
    #                 # method="Radau",
    #                 # method="BDF",
    #                 vectorized=False,
    #                 # jac=tempAf,
    #                 max_step=max_step,
    #                 # min_step=max_step,
    #                 # events=(impulse_start,),
    #             )
    #             solBf = solver.y[:, -1]
    #             # print("Bf m=", m)
    #             # print(solver)

    #             def _derivative_backward_2(t, x):
    #                 return np.dot(-tempAb, x) - np.dot(
    #                     self.analog_system.Gamma,
    #                     digital_control.impulse_response(m, self.Ts - t),
    #                 )

    #             def impulse_start(t, x):
    #                 return self.Ts - self.digital_control._impulse_response[m].t0 - t

    #             # impulse_start.terminate = True
    #             # impulse_start.direction = -1.0

    #             solver = scipy.integrate.solve_ivp(
    #                 _derivative_backward_2,
    #                 (0, self.Ts),
    #                 np.zeros(self.analog_system.N),
    #                 atol=atol,
    #                 rtol=rtol,
    #                 method="RK45",
    #                 # method="Radau",
    #                 # method="BDF",
    #                 vectorized=False,
    #                 # jac=-tempAb,
    #                 max_step=max_step,
    #                 # min_step=max_step,
    #                 events=(impulse_start,),
    #             )
    #             solBb = solver.y[:, -1]
    #             # print("Bb m=", m)
    #             # print(solver)

    #             self.Bf[:, m] = solBf
    #             self.Bb[:, m] = solBb
    #     W, _, _, _ = np.linalg.lstsq(
    #         np.array(Vf + Vb, dtype=np.double), analog_system.B, rcond=-1
    #     )
    #     self.WT = W.transpose()

    def __next__(self) -> np.ndarray:
        # Check if control signal iterator is set.
        if self.control_signal is None:
            raise Exception("No iterator set.")

        # Check if the end of prespecified size
        self._iteration += self.downsample
        if self.number_of_iterations and self.number_of_iterations < self._iteration:
            raise StopIteration

        # Rotate control_signal vector
        self._control_signal_valued = np.roll(
            self._control_signal_valued, -self.downsample, axis=0
        )

        # insert new control signal
        try:
            for index in range(self.downsample):
                self._temp_controls[index, :] = 2 * self.control_signal.__next__() - 1
        except RuntimeError:
            logger.warning("Estimator received Stop Iteration")
            raise StopIteration

        self._control_signal_valued[
            self.K3 - self.downsample :, :
        ] = self._temp_controls

        # self._control_signal_valued.shape -> (K3, M)
        # self.h.shape -> (L, K3, M)
        res = (
            np.tensordot(self.h, self._control_signal_valued, axes=((1, 2), (0, 1)))
            + self.offset
        )
        if self.fixed_point:
            return self.__fixed_to_float(res)
        else:
            return res
        # return np.einsum('lkm,km', self.h, self._control_signal_valued)
        # the Einstein summation results in:
        # result = np.zeros(self._L)
        # for l in range(self._L):
        #    for k in range(self.K1 + self.K2):
        #        for m in range(self._M):
        #            result[l] += self.h[l, k, m] * self._control_signal_valued[k, m]
        # return result

    def lookback(self):
        """Return lookback size :math:`K1`.

        Returns
        -------
        int
            lookback size.
        """
        return self.K1

    def __str__(self):
        return f"FIR estimator is parameterized as \neta2 = {self.eta2:.2f}, {10 * np.log10(self.eta2):.0f} [dB],\nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},\nand\nnumber_of_iterations = {self.number_of_iterations}.\nResulting in the filter coefficients\nh = \n{self.h}."

    def filter_lag(self):
        """Return the lag of the filter.

        Returns
        -------
        `int`
            The filter lag.

        """
        return self._filter_lag

    def fir_filter_transfer_function(self, Ts: float = 1.0):
        """Compute the FFT of the system impulse response (FIR filter coefficients).

        Parameters
        ----------
        Ts: `float`
            the sample period of the corresponding impulse response.

        Returns
        -------
        `array_like`, shape=(L, K3 // 2, M)
            the FFT of the corresponding impulse responses.
        """
        frequency_response = np.fft.rfft(self.h, axis=1)
        frequencies = np.fft.rfftfreq(self.K3, d=Ts)
        return (frequencies, frequency_response)

    def convolve(self, filter: np.ndarray):
        """Shape :math:`\mathbf{h}` filter by convolving with filter

        Parameters
        ----------
        filter: `array_like`, shape=(K)
            filter to be applied for each digital control filter
            equivalently.
        """
        for l in range(self.analog_system.L):
            for m in range(self.analog_system.M):
                # self.h.shape -> (L, K3, M)
                temp = np.convolve(self.h[l, :, m], filter, mode="full")
                if temp.size == self.K3:
                    self.h[l, :, m] = temp
                else:
                    mid_point = temp.size // 2
                    half_length = self.K3 // 2
                    self.h[l, :, m] = temp[
                        (mid_point - half_length) : (mid_point + half_length)
                    ]

    def write_C_header(self, filename: str):
        """Write the FIR filter coefficients h into
        a C header file.

        Parameters
        ----------
        filename: `str`
            filename of header file.
        """
        if not filename.endswith(".h"):
            filename += ".h"

        with open(filename, "w") as f:
            f.write(
                "// This file was autogenerated using the src/cbadc/digital_estimator.py FIRFilter class's write_C_header function."
                + os.linesep
            )
            f.write(f"#define L {self.analog_system.L}" + os.linesep)
            f.write(f"#define M {self.analog_system.M}" + os.linesep)
            f.write(f"#define K1 {self.K1}" + os.linesep)
            f.write(f"#define K2 {self.K2}" + os.linesep)
            f.write(f"#define K3 {self.K3}" + os.linesep)
            # self.h.shape -> (L, K3, M)
            if self.fixed_point:
                f.write("int h[L][M][K3] = ")
            else:
                f.write("double h[L][M][K3] = ")
            f.write("{")
            for l in range(self.analog_system.L):
                f.write("{")
                for m in range(self.analog_system.M):
                    f.write("{")
                    if self.fixed_point:
                        for k in range(self.K3 - 1):
                            f.write(f"{self.h[l,k,m]},")
                        f.write(f"{self.h[l,-1,m]}" + "}")
                    else:
                        for k in range(self.K3 - 1):
                            f.write(f"{self.h[l,k,m]:.17E},")
                        f.write(f"{self.h[l,-1,m]:.17E}" + "}")
                    if m < (self.analog_system.M - 1):
                        f.write(",")
                f.write("}")
                if l < (self.analog_system.L - 1):
                    f.write(",")
            f.write("}" + os.linesep)

    def number_of_filter_coefficients(self) -> int:
        """Number of non-zero filter coefficients

        Returns
        -------
        `int`
            total number of non-zero filter coefficients
        """
        return np.sum(self.h > 0)
