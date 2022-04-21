"""Synthesise functions for leap-frog control-bounded ADCs.
"""
from cbadc.fom import snr_from_dB, enob_to_snr
from cbadc.analog_system.leap_frog import LeapFrog
from cbadc.digital_control.digital_control import DigitalControl
from cbadc.analog_signal.impulse_responses import RCImpulseResponse, StepResponse
from cbadc.analog_signal import Clock, ConstantSignal
from cbadc.analog_frontend import AnalogFrontend
from cbadc.simulator import PreComputedControlSignalsSimulator
import sympy as sp
import numpy as np


def g_i(N: int):
    """Compute the integration factor g_i

    Parameters
    ----------
    N: `int`
        the system order
    Returns
    -------
    :  `float`
        the computed integration factor.
    """
    omega, omega_p, gamma = sp.symbols('w w_p, g', real=True, positive=True)
    n = sp.symbols('n', integer=True, positive=True)
    determinant = sp.Product(
        sp.I * (omega + omega_p * sp.cos(n * sp.pi / (N + 1))), (n, 1, N)
    )
    H = determinant / ((gamma * omega_p) ** N)
    H2 = sp.Abs(H) ** 2
    LF_int = sp.integrate(H2, (omega, 0, omega_p))
    g_i = sp.simplify(omega_p / (LF_int * gamma ** (2 * N)))
    return np.float64(g_i.subs(omega_p, 1e0).evalf())


def get_leap_frog(**kwargs) -> AnalogFrontend:
    """Quick parameterize a leap-frog ADC

    Returns a parameterized analog system and
    digital control corresponding to a given
    target specification.

    The following examples demonstrate the
    valid input specifications

    Parameters
    ----------
    ENOB: `float`
        targeted effective number of bits.
    N: `int`
        system order.
    BW: `float`
        target bandwidth
    xi: `float`, `optional`
        a proportionality constant, defaults to 0.022.
    local_feedback: `bool`, `optional`
        include local feedback, defaults to False.
    excess_delay: `float`, `optional`
        delay control actions by an excess delay, defaults to 0.


    Returns
    -------
    : (:py:class:`cbadc.analog_system.LeapFrog`, :py:class:`cbadc.digital_control.DigitalControl`)
        returns an analog system and digital control tuple
    """

    if all(param in kwargs for param in ('ENOB', 'N', 'BW')):
        SNR = enob_to_snr(kwargs['ENOB'])
        snr = snr_from_dB(SNR)
        N = kwargs['N']
        omega_BW = 2.0 * np.pi * kwargs['BW']
        xi = 8e-3
        if 'xi' in kwargs:
            xi = kwargs['xi']
        gamma = (xi / g_i(N) * snr) ** (1.0 / (2.0 * N))
        beta = -(omega_BW / 2.0) * (2 * gamma)
        alpha = (omega_BW / 2.0) / (2 * gamma)
        rho = 0
        if 'local_feedback' in kwargs and kwargs['local_feedback'] is True:
            rho = -(omega_BW / 2.0) / gamma * 1e-2 * 0
        T = 1.0 / np.abs(2.0 * beta)
        kappa = beta
        t0 = 0.0
        if 'excess_delay' in kwargs:
            t0 = kwargs['excess_delay'] * T
        if kwargs.get('digital_control') == 'switch-cap':
            scale = 5e1
            tau = 1.0 / (np.abs(beta) * scale)
            v_cap = 0.5
            kappa = v_cap * beta * scale
            impulse_response = RCImpulseResponse(tau, t0)
            digital_control = DigitalControl(
                Clock(T), N, impulse_response=impulse_response
            )
        else:
            impulse_response = StepResponse(t0)
            digital_control = DigitalControl(
                Clock(T), N, impulse_response=impulse_response
            )
            digital_control = DigitalControl(Clock(T), N)
        if 'opt' in kwargs:
            amplification_scale_vector = np.array([1.01 ** (n) for n in range(N)])
            analog_system = LeapFrog(
                beta * amplification_scale_vector,
                alpha / amplification_scale_vector,
                rho * np.ones(N),
                kappa * np.eye(N),
            )
        else:
            analog_system = LeapFrog(
                beta * np.ones(N),
                alpha * np.ones(N - 1),
                rho * np.ones(N),
                kappa * np.eye(N),
            )
        return AnalogFrontend(analog_system, digital_control)
    elif all(param in kwargs for param in ('OSR', 'N', 'BW')):
        N = kwargs['N']
        BW = kwargs['BW']
        OSR = kwargs['OSR']
        T = 1.0 / (2 * OSR * BW)
        omega_BW = 2.0 * np.pi * BW
        beta = 1 / (2 * T)
        alpha = -((omega_BW / 2) ** 2) / beta
        kappa = beta
        rho = 0
        digital_control = DigitalControl(Clock(T), N)
        if 'opt' in kwargs and kwargs['opt'] == True:
            amplification_scale_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0] + [1] * N)
            control_scale_vector = np.ones(N)
            for m in range(N - 1):
                analog_system = LeapFrog(
                    beta * amplification_scale_vector[:N],
                    alpha / amplification_scale_vector[: N - 1],
                    rho * np.ones(N),
                    kappa * np.diag(control_scale_vector[:N]),
                )
                sim = PreComputedControlSignalsSimulator(
                    analog_system, digital_control, [ConstantSignal(1.0)]
                )
                control_scale_vector[m + 1] = np.linalg.norm(
                    sim._pre_computed_control_matrix[m + 1, :], ord=1
                )
                # amplification_scale_vector[m + 1] = 1.0 / control_scale_vector[m + 1]
            analog_system = LeapFrog(
                beta * amplification_scale_vector[:N],
                alpha / amplification_scale_vector[: N - 1],
                rho * np.ones(N),
                kappa * np.diag(control_scale_vector[:N]),
            )
            print(f"control_scale_vector:\n{control_scale_vector}")
            print(f"amplification_scale_vector:\n{amplification_scale_vector}")
        else:
            analog_system = LeapFrog(
                beta * np.ones(N),
                alpha * np.ones(N - 1),
                rho * np.ones(N),
                kappa * np.eye(N),
            )
        return AnalogFrontend(analog_system, digital_control)
    elif all(param in kwargs for param in ('OSR', 'N', 'T')):
        N = kwargs['N']
        T = kwargs.pop('T')
        OSR = kwargs['OSR']
        BW = 1.0 / (2 * OSR * T)

        return get_leap_frog(N=N, OSR=OSR, BW=BW, **kwargs)
    raise NotImplementedError
