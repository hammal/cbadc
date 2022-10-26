"""Filter coefficient computations."""

import cbadc
import logging
import time
import copy
import enum
import scipy.linalg
import scipy.integrate
import numpy as np
import sympy as sp
import mpmath as mp
from mpmath import mp
from numpy.linalg import LinAlgError
from ..ode_solver.sympy import invariant_system_solver as analytical_system_solver
from ..ode_solver.mpmath import invariant_system_solver as mp_system_solver

logger = logging.getLogger(__name__)


# creating enumerations using class
class FilterComputationBackend(enum.Enum):
    numpy = 1
    sympy = 2
    mpmath = 3


class BruteForceCareException(Exception):
    pass


def bruteForceCare(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    tau=1e-12,
    rtol=1e-200,
    atol=1e-300,
) -> np.ndarray:

    # raise BruteForceCareException("Currently not a reliable implementation.")
    timelimit = 10 * 60
    start_time = time.time()

    V = _numpy_care(A, B, Q, R)
    # V = np.eye(A.shape[0])
    V_tmp = np.ones_like(V) * 1e300
    RInv = np.linalg.inv(R)

    shrink = 1.0 - 1e-2

    while not np.allclose(V, V_tmp, rtol=rtol, atol=atol):
        if time.time() - start_time > timelimit:
            raise BruteForceCareException("Brute Force CARE solver ran out of time")
        V_tmp = V[:, :]
        try:
            V = V + tau * (
                np.dot(A.transpose(), V)
                + np.dot(V.transpose(), A)
                + Q
                - np.dot(
                    V, np.dot(B, np.dot(RInv, np.dot(B.transpose(), V.transpose())))
                )
            )
            V = 0.5 * (V + V.transpose())
        except FloatingPointError:
            logger.warning("V_frw:\n{}\n".format(V))
            logger.warning("V_frw.dot(V_frw):\n{}".format(np.dot(V, V)))
            raise FloatingPointError
        # print(np.linalg.norm(V - V_tmp, ord="fro"))
        tau *= shrink
    return np.array(V, dtype=np.double)


def _scipy_care(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> np.ndarray:
    return scipy.linalg.solve_continuous_are(A, B, Q, R, balanced=True)


def care(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    This function solves the forward and backward continuous Riccati equation.
    """
    A = np.array(A, dtype=np.double)
    B = np.array(B, dtype=np.double)
    Q = np.array(Q, dtype=np.double)
    R = np.array(R, dtype=np.double)

    V = np.zeros_like(A)

    try:
        logger.info("Computing CARE analytically")
        V = _analytical_care(A, B, Q, R)
    except (sp.PrecisionExhausted, TypeError):
        logger.warning(
            """PrecisionExhausted in analytical CARE solver, switching to Scipy."""
        )
        try:
            V = _scipy_care(A, B, Q, R)
        except LinAlgError:
            logger.warning(
                """Scipy Cholesky method failed for CARE solver.
            Starting brute force"""
            )
            V = bruteForceCare(A, B, Q, R)
    return np.array((V + V.transpose()) / 2.0, dtype=np.double)


def _analytical_care(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> sp.Matrix:
    A = sp.Matrix(A)
    B = sp.Matrix(B)
    Q = sp.Matrix(Q)
    R = sp.Matrix(R)

    Z = sp.Matrix([[A, -B * R.inv() * B.transpose()], [-Q, -A.transpose()]])

    (P, D) = Z.diagonalize(reals_only=False, sort=True)
    sol = P[A.shape[0] :, : A.shape[1]] * P[: A.shape[0], : A.shape[1]].inv()
    V: sp.Matrix = sp.re(sp.simplify(sol))
    return (V + V.transpose()) / 2


def _numpy_care(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.array:
    Z = np.vstack(
        (
            np.hstack((A, -np.dot(B, np.dot(np.linalg.inv(R), B.T)))),
            np.hstack((-Q, -A.transpose())),
        )
    )
    W, V = np.linalg.eig(Z)
    # Sort
    idx = W.argsort()
    W = W[idx]
    V = V[:, idx]
    sol = np.dot(
        V[A.shape[0] :, : Q.shape[1]], np.linalg.pinv(V[: A.shape[0], : Q.shape[1]])
    )
    return np.real(sol)


def _mpmath_care(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> mp.matrix:
    B = mp.matrix(B)
    A = mp.matrix(A)
    Q = mp.matrix(Q)
    R = mp.matrix(R)

    Z = mp.matrix(A.rows + Q.rows, Q.cols + A.rows)

    Z_11 = A
    Z_12 = -B * R ** (-1) * B.T
    Z_21 = -Q
    Z_22 = -A.T
    Z[: Z_11.rows, : Z_11.cols] = Z_11
    Z[: Z_11.rows, Z_11.cols :] = Z_12
    Z[Z_11.rows :, : Z_21.cols] = Z_21
    Z[Z_11.rows :, Z_21.cols :] = Z_22
    E, Q = mp.eig(Z)
    sol = Q[Z_11.rows :, : Z_11.cols] * Q[: Z_11.rows, : Z_11.cols] ** (-1)
    raise NotImplementedError
    return np.real(np.array(sol.tolist(), dtype=np.complex128)).astype(np.float64)


def compute_filter_coefficients(
    analog_system: cbadc.analog_system.AnalogSystem,
    digital_control: cbadc.digital_control.DigitalControl,
    eta2: float,
    solver_type: FilterComputationBackend = FilterComputationBackend.mpmath,
    mid_point: bool = False,
):
    # Compute filter coefficients
    A = np.array(analog_system.A).transpose()
    B = np.array(analog_system.CT).transpose()
    Q = np.dot(np.array(analog_system.B), np.array(analog_system.B).transpose())
    R = eta2 * np.eye(analog_system.N_tilde)
    # Solve care
    Vf = care(A, B, Q, R)
    Vb = care(-A, B, Q, R)

    if solver_type == FilterComputationBackend.sympy:
        return _analytical_solver(
            analog_system,
            digital_control,
            sp.Matrix(R),
            sp.Matrix(Vf),
            sp.Matrix(Vb),
            mid_point,
        )
    if solver_type == FilterComputationBackend.mpmath:
        return _mp_solver(
            analog_system, digital_control, mp.matrix(R), mp.matrix(Vf), mp.matrix(Vb)
        )
    else:  # FilterComputationBackend.numpy
        return _numerical_solver(analog_system, digital_control, R, Vf, Vb, mid_point)


def _analytical_solver(
    analog_system: cbadc.analog_system.AnalogSystem,
    digital_control: cbadc.digital_control.DigitalControl,
    R: sp.Matrix,
    Vf: sp.Matrix,
    Vb: sp.Matrix,
    mid_point: bool,
):
    if mid_point:
        raise NotImplementedError
    Ts = digital_control.clock.T
    A_sym = sp.Matrix(analog_system.A)
    Vf_sym = sp.Matrix(Vf)
    Vb_sym = sp.Matrix(Vb)
    # eta2_sym = sp.Float(eta2)
    CCT_sym = (
        sp.Matrix(analog_system.CT).T
        * sp.Matrix(R) ** (-1)
        * sp.Matrix(analog_system.CT)
    )
    tempAf = A_sym - Vf_sym * CCT_sym
    tempAb = A_sym + Vb_sym * CCT_sym
    Af, Bf = _ode_solver(tempAf, analog_system._Gamma_s, digital_control, Ts)
    Ab, Bb = _ode_solver(-tempAb, -analog_system._Gamma_s, digital_control, Ts)
    W = (Vf_sym + Vb_sym) ** (-1) * sp.Matrix(analog_system.B)
    return (
        np.array(Af).astype(np.float64),
        np.array(Ab).astype(np.float64),
        np.array(Bf).astype(np.float64),
        np.array(Bb).astype(np.float64),
        np.array(W.T).astype(np.float64),
    )


def _ode_solver(
    tempAf: sp.Matrix,
    tempBf: sp.Matrix,
    digital_control: cbadc.digital_control.DigitalControl,
    Ts: float,
):
    sigs = [fun.symbolic() for fun in digital_control._impulse_response]
    hom, non_hom, t = analytical_system_solver(
        tempAf, tempBf, sigs, [d.t0 for d in digital_control._impulse_response]
    )
    A_sol = np.real(np.array(hom.subs(t, Ts)).astype(np.complex128))
    B_sol = np.zeros_like(tempBf)
    for n in range(B_sol.shape[0]):
        for m in range(B_sol.shape[1]):
            B_sol[n, m] = non_hom[m][n].subs(t, Ts).rhs.doit().evalf()
    return A_sol, B_sol


def _mp_solver(
    analog_system: cbadc.analog_system.AnalogSystem,
    digital_control: cbadc.digital_control.DigitalControl,
    R: mp.matrix,
    Vf: mp.matrix,
    Vb: mp.matrix,
):
    tol: float = 1e-40
    # set number of decimal places.
    dps = 50
    tmp_dps = mp.dps
    mp.dps = dps

    CetaCT = mp.matrix(
        mp.matrix(analog_system.CT).transpose()
        * R ** (-1)
        * mp.matrix(analog_system.CT)
    )
    Ts = mp.mpf(digital_control.clock.T)
    t_span = (mp.mpf('0'), Ts)
    tempAf = mp.matrix(analog_system._A_s) - Vf * CetaCT
    tempAb = mp.matrix(analog_system._A_s) + Vb * CetaCT
    Af, Bf = mp_system_solver(
        tempAf,
        mp.matrix(analog_system.Gamma),
        digital_control._impulse_response,
        t_span,
        tol=tol,
    )
    reversed_signals = list(
        map(lambda s: reverse_signal(s, Ts), digital_control._impulse_response)
    )
    Ab, Bb = mp_system_solver(
        -tempAb, -mp.matrix(analog_system.Gamma), reversed_signals, t_span, tol=tol
    )
    WT = mp.matrix(analog_system.L, analog_system.N)
    VfVb = Vf + Vb
    for l in range(analog_system.L):
        WT[l, :] = mp.qr_solve(VfVb, mp.matrix(analog_system.B[:, l]))[0].transpose()
        # WT[l, :] = mp.lu_solve(VfVb, mp.matrix(analog_system.B))[l].transpose()
    mp.dps = tmp_dps
    return Af, Ab, Bf, Bb, WT


def reverse_signal(signal: cbadc.analog_signal._AnalogSignal, T0: float = 0.0):
    """
    A function to reverse a signal around T0,
    i.e.,

    u(t) -> u(T0 - t)

    Parameters
    ----------
    signal: :py:class:`cbadc.analog_system._AnalogSystem`
        the signal to be reversed
    T0: `float`
        the point around which the signal is reversed.

    Returns
    -------
    : :py:class:`cbadc.analog_system._AnalogSignal`
        a new analog signal.
    """
    new_signal = copy.deepcopy(signal)
    new_signal.evaluate = lambda t: signal.evaluate(T0 - t)
    new_signal._mpmath = lambda t: signal._mpmath(mp.mpf(T0) - t)
    new_signal.t0 = -signal.t0
    new_signal._t0_mpmath = -signal._t0_mpmath
    return new_signal


def _numerical_solver(
    analog_system: cbadc.analog_system.AnalogSystem,
    digital_control: cbadc.digital_control.DigitalControl,
    R: np.ndarray,
    Vf: np.ndarray,
    Vb: np.ndarray,
    mid_point: bool,
):
    CCT: np.ndarray = np.dot(
        np.array(analog_system.CT).transpose(),
        np.dot(np.linalg.inv(R), np.array(analog_system.CT)),
    )
    Ts = digital_control.clock.T
    tempAf: np.ndarray = analog_system.A - np.dot(Vf, CCT)
    tempAb: np.ndarray = analog_system.A + np.dot(Vb, CCT)
    Af: np.ndarray = np.asarray(scipy.linalg.expm(tempAf * Ts))
    Ab: np.ndarray = np.asarray(scipy.linalg.expm(-tempAb * Ts))
    W, _, _, _ = np.linalg.lstsq(Vf + Vb, analog_system.B, rcond=-1)
    WT = W.transpose()
    if mid_point:
        Bf, Bb = _mid_point(analog_system, digital_control, tempAf, tempAb)
    else:
        Bf, Bb = _regular(analog_system, digital_control, tempAf, tempAb)
    return Af, Ab, Bf, Bb, WT


def _regular(
    analog_system: cbadc.analog_system.AnalogSystem,
    digital_control: cbadc.digital_control.DigitalControl,
    tempAf: np.ndarray,
    tempAb: np.ndarray,
):
    Ts = digital_control.clock.T
    Gamma = np.array(analog_system.Gamma)
    # Solve IVPs
    Bf: np.ndarray = np.zeros((analog_system.N, analog_system.M))
    Bb: np.ndarray = np.zeros((analog_system.N, analog_system.M))

    atol = 1e-100
    rtol = 1e-14
    # max_step = Ts * 1e-5
    for m in range(analog_system.M):

        def _derivative_forward_2(t, x):
            return np.dot(tempAf, x) + np.dot(
                Gamma, digital_control.impulse_response(m, t)
            )

        def impulse_start(t, x):
            return t - digital_control._impulse_response[m].t0

        # impulse_start.terminate = True
        impulse_start.direction = 1.0

        solBf = scipy.integrate.solve_ivp(
            _derivative_forward_2,
            (0, Ts),
            # (digital_control._impulse_response[m].t0, Ts),
            np.zeros(analog_system.N),
            atol=atol,
            rtol=rtol,
            # max_step=max_step,
            # method="Radau",
            # jacobian=tempAf,
            events=(impulse_start,),
        ).y[:, -1]

        def _derivative_backward_2(t, x):
            return np.dot(-tempAb, x) - np.dot(
                Gamma, digital_control.impulse_response(m, Ts - t)
            )

        def impulse_stop(t, x):
            return t - Ts + digital_control._impulse_response[m].t0

        # impulse_stop.terminate = True
        impulse_stop.direction = 1.0

        solBb = scipy.integrate.solve_ivp(
            _derivative_backward_2,
            (0, Ts),
            np.zeros(analog_system.N),
            # solBf,
            atol=atol,
            rtol=rtol,
            # max_step=max_step,
            # method="Radau",
            # jacobian=-tempAb,
            events=(impulse_stop,),
        ).y[:, -1]
        Bf[:, m] = solBf
        Bb[:, m] = solBb
    return Bf, Bb


def _mid_point(
    analog_system: cbadc.analog_system.AnalogSystem,
    digital_control: cbadc.digital_control.DigitalControl,
    tempAf: np.ndarray,
    tempAb: np.ndarray,
):
    Ts = digital_control.clock.T
    Gamma = np.array(analog_system.Gamma)
    # Solve IVPs
    Bf: np.ndarray = np.zeros((analog_system.N, analog_system.M))
    Bb: np.ndarray = np.zeros((analog_system.N, analog_system.M))

    atol = 1e-200
    rtol = 1e-10
    max_step = Ts / 1000.0
    for m in range(analog_system.M):

        def _derivative_forward(t, x):
            return np.dot(tempAf, x) + np.dot(
                Gamma, digital_control.impulse_response(m, t)
            )

        solBf = scipy.integrate.solve_ivp(
            _derivative_forward,
            (0, Ts / 2.0),
            np.zeros(analog_system.N),
            atol=atol,
            rtol=rtol,
            max_step=max_step,
            # method="Radau",
        ).y[:, -1]

        def _derivative_backward(t, x):
            return np.dot(-tempAb, x) - np.dot(
                Gamma, digital_control.impulse_response(m, Ts - t)
            )

        solBb = scipy.integrate.solve_ivp(
            _derivative_backward,
            (0, Ts / 2.0),
            np.zeros(analog_system.N),
            atol=atol,
            rtol=rtol,
            max_step=max_step,
            # method="Radau",
        ).y[:, -1]
        for n in range(analog_system.N):
            Bf[n, m] = solBf[n]
            Bb[n, m] = solBb[n]
    Bf = np.dot(np.eye(analog_system.N) + scipy.linalg.expm(tempAf * Ts / 2.0), Bf)
    Bb = np.dot(np.eye(analog_system.N) + scipy.linalg.expm(tempAb * Ts / 2.0), Bb)
    return Bf, Bb
