"""Synthesise functions for leap-frog control-bounded ADCs.
"""
from cbadc.fom import snr_from_dB, enob_to_snr, snr_to_enob
from cbadc.analog_system.leap_frog import LeapFrog
from cbadc.digital_control.digital_control import DigitalControl
from cbadc.analog_signal.impulse_responses import RCImpulseResponse, StepResponse
from cbadc.analog_signal import Clock, ConstantSignal
from cbadc.analog_frontend import AnalogFrontend
from cbadc.simulator import PreComputedControlSignalsSimulator
import sympy as sp
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
    omega, omega_p, gamma = sp.symbols("w w_p, g", real=True, positive=True)
    n = sp.symbols("n", integer=True, positive=True)
    determinant = sp.Product(
        sp.I * (omega + omega_p * sp.cos(n * sp.pi / (N + 1))), (n, 1, N)
    )
    H = determinant / ((gamma * omega_p) ** N)
    H2 = sp.Abs(H) ** 2
    LF_int = sp.integrate(H2, (omega, 0, omega_p))
    # g_i = sp.simplify(omega_p / (LF_int * gamma ** (2 * N)))
    g_i = omega_p / (LF_int * gamma ** (2 * N))
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
        a proportionality constant, defaults to 4e-3.
    local_feedback: `bool`, `optional`
        include local feedback, defaults to False.
    excess_delay: `float`, `optional`
        delay control actions by an excess delay, defaults to 0.


    Returns
    -------
    : (:py:class:`cbadc.analog_system.LeapFrog`, :py:class:`cbadc.digital_control.DigitalControl`)
        returns an analog system and digital control tuple
    """

    if all(param in kwargs for param in ("ENOB", "N", "BW")):
        SNR = enob_to_snr(kwargs["ENOB"])
        snr = snr_from_dB(SNR)
        N = kwargs["N"]
        omega_BW = 2.0 * np.pi * kwargs["BW"]
        xi = kwargs.get("xi", 4e-3)
        delta = kwargs.get("delta", 1.0)
        gamma_over_delta = (xi / g_i(N) * snr) ** (1.0 / (2.0 * N))
        gamma = gamma_over_delta * delta
        omega_p = omega_BW / 2.0
        # omega_p /= np.cos(N * np.pi / (N + 1.0))
        beta = -omega_p * (2.0 * gamma)
        alpha = omega_p / (2.0 * gamma)
        rho = 0
        T = 1.0 / np.abs(2.0 * omega_BW * gamma / delta)
        kappa = beta
    elif all(param in kwargs for param in ("OSR", "N", "BW")):
        N = kwargs["N"]
        BW = kwargs["BW"]
        OSR = kwargs["OSR"]
        T = 1.0 / (2 * OSR * BW)
        omega_BW = 2.0 * np.pi * BW
        delta = kwargs.get("delta", 1.0)
        gamma = delta * OSR / (2 * np.pi)
        beta = gamma * omega_BW
        alpha = -((omega_BW / 2) ** 2) / beta
        kappa = beta
        rho = 0
    elif all(param in kwargs for param in ("SNR", "N", "BW")):
        return get_leap_frog(ENOB=snr_to_enob(kwargs["SNR"]), **kwargs)
    elif all(param in kwargs for param in ("OSR", "N", "T")):
        T = kwargs.pop("T")
        BW = 1.0 / (2 * kwargs["OSR"] * T)
        return get_leap_frog(BW=BW, **kwargs)
    else:
        raise NotImplementedError

    analog_system = LeapFrog(
        beta * np.ones(N),
        alpha * np.ones(N - 1),
        rho * np.ones(N),
        np.diag([kappa * delta**n for n in range(N)]),
    )
    return AnalogFrontend(analog_system, DigitalControl(Clock(T), N))
