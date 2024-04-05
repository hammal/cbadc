"""The analog frontend"""

import cbadc
import numpy as np
import scipy.integrate
import scipy.linalg


class AnalogFrontend:
    """Represents an analog frontend.

    Parameters
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system instance
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        a digital control instance

    Attributes
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        the analog frontend's analog system instance
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        the analog frontend's digital control instance
    """

    def __init__(
        self,
        analog_system: cbadc.analog_system._valid_analog_system_types,
        digital_control: cbadc.digital_control._valid_digital_control_types,
    ):
        self.analog_system = analog_system
        self.digital_control = digital_control


def _analog_system_matrix_exponential(A: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(scipy.linalg.expm(np.asarray(A) * t))


def get_global_control(
    analog_system: cbadc.analog_system.AnalogSystem,
    digital_control: cbadc.digital_control.DigitalControl,
    phi_delay: np.ndarray,
    atol=1e-12,
    rtol=1e-12,
) -> AnalogFrontend:
    """Compute the global control for the given analog system and digital control.

    Parameters
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system instance
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        a digital control instance
    phi_delay: np.ndarray
        the delay vector
    atol: float
        absolute tolerance for the ODE solver
    rtol: float
        relative tolerance for the ODE solver

    Returns
    -------
    :py:class:`cbadc.analog_frontend.AnalogFrontend`
        an analog frontend with global control
    """

    def zero_order_hold(l: int, t: float) -> np.ndarray:
        res = np.zeros(analog_system.L)
        if t >= 0 and t < digital_control.clock.T:
            res[l] = 1.0
        return res

    A_tilde = np.zeros((analog_system.M_tilde, analog_system.M))
    B_tilde = np.zeros((analog_system.M_tilde, analog_system.L))
    Gamma_tildeT = np.zeros((analog_system.M_tilde, analog_system.N))
    # analog_system.CT = np.zeros((analog_system.N, analog_system.N))
    # analog_system.CT[-1, -1] = 1.0

    for m_tilde in range(analog_system.M_tilde):
        m_tilde_index = (
            m_tilde + analog_system.M - analog_system.M_tilde + analog_system.L
        )
        index_offset = (analog_system.M + analog_system.L) * analog_system.N

        def derivative_m(t: float, x: np.ndarray) -> np.ndarray:
            res = np.zeros_like(x)

            # extract the relevant control vector

            psi_s_m = np.dot(
                analog_system.CT,
                x[
                    m_tilde_index
                    * analog_system.N : (1 + m_tilde_index)
                    * analog_system.N
                ],
            )

            # compute the input signal psi_u
            for l in range(analog_system.L):
                res[l * analog_system.N : (l + 1) * analog_system.N] = np.dot(
                    analog_system.A, x[l * analog_system.N : (l + 1) * analog_system.N]
                ) + np.dot(analog_system.B, zero_order_hold(l, t - phi_delay[m_tilde]))

            # compute all control signal psi_s, with relevant impulse responses
            for m in range(analog_system.M):
                if m > m_tilde + analog_system.M - analog_system.M_tilde:
                    corrected_impulse_response = digital_control.impulse_response(
                        m, t + digital_control.clock.T
                    )
                else:
                    corrected_impulse_response = digital_control.impulse_response(m, t)

                res[
                    (m + analog_system.L)
                    * analog_system.N : (m + 1 + analog_system.L)
                    * analog_system.N
                ] = np.dot(
                    analog_system.A,
                    x[
                        (m + analog_system.L)
                        * analog_system.N : (m + 1 + analog_system.L)
                        * analog_system.N
                    ],
                ) + np.dot(
                    analog_system.Gamma, corrected_impulse_response
                )

            # compute inner product between homogeneous state vector and control signal
            res[index_offset : index_offset + analog_system.N] = np.dot(
                psi_s_m.transpose(),
                np.dot(
                    analog_system.CT,
                    _analog_system_matrix_exponential(
                        analog_system.A, t - phi_delay[m_tilde]
                    ),
                ),
            )

            # compute inner product between input signal and control vector
            for l in range(analog_system.L):
                res[index_offset + analog_system.N + l] = np.inner(
                    psi_s_m,
                    np.dot(
                        analog_system.CT,
                        x[l * analog_system.N : (l + 1) * analog_system.N],
                    ),
                )

            # compute inner product between control signal and control vector
            for m in range(analog_system.M):
                res[index_offset + analog_system.N + analog_system.L + m] = np.inner(
                    psi_s_m,
                    np.dot(
                        analog_system.CT,
                        x[
                            (m + analog_system.L)
                            * analog_system.N : (m + 1 + analog_system.L)
                            * analog_system.N
                        ],
                    ),
                )

            return res

        events = []

        for m in range(analog_system.M_tilde):

            def event_function(t, x):
                return t - phi_delay[m]

            event_function.terminal = False
            event_function.direction = 0
            events.append(event_function)

        # solve ode
        sol = scipy.integrate.solve_ivp(
            derivative_m,
            (phi_delay[m_tilde], phi_delay[m_tilde] + digital_control.clock.T),
            np.zeros(
                index_offset + analog_system.N + analog_system.L + analog_system.M
            ),
            atol=atol,
            rtol=rtol,
            events=events,
        )

        y = sol.y[:, -1]

        Q = y[index_offset + analog_system.N + m_tilde_index]

        for l in range(analog_system.L):
            B_tilde[m_tilde, l] = -1 / Q * y[index_offset + analog_system.N + l]

        for m in range(analog_system.M):
            if m != m_tilde - analog_system.M_tilde + analog_system.M:
                A_tilde[m_tilde, m] = (
                    -1 / Q * y[index_offset + analog_system.N + analog_system.L + m]
                )

        Gamma_tildeT[m_tilde, :] = (
            -1 / Q * y[index_offset : index_offset + analog_system.N]
        )

    analog_system = cbadc.analog_system.AnalogSystem(
        analog_system.A,
        analog_system.B,
        analog_system.CT,
        analog_system.Gamma,
        Gamma_tildeT,
        A_tilde=A_tilde,
        B_tilde=B_tilde,
    )

    return AnalogFrontend(analog_system, digital_control)


def inverse_state_trajectory(A: np.ndarray, T: float):
    def derivative(t: float, x: np.ndarray) -> np.ndarray:
        return _analog_system_matrix_exponential(A, t).reshape(x.shape)

    sol = scipy.integrate.solve_ivp(
        derivative, (0, T), np.zeros(A.size), atol=1e-10, rtol=1e-10
    )

    res = sol.y[:, -1].reshape(A.shape)
    return res, np.linalg.inv(res)
