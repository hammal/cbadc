"""Initialize analog systems and digital control from specifications.
"""
from .fom import snr_from_dB, enob_to_snr
from .analog_system.chain_of_integrators import ChainOfIntegrators
from .analog_system.leap_frog import LeapFrog
from .digital_control.digital_control import DigitalControl
from .analog_signal.impulse_responses import RCImpulseResponse, StepResponse
from .analog_signal.clock import Clock
import numpy as np


def get_chain_of_integrator(**kwargs):
    """Quick parameterize a chain-of-integrator ADC

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
        a proportionality constant, defaults to 0.0016.
    local_feedback: `bool`, `optional`
        include local feedback, defaults to False.
    excess_delay: `float`, `optional`
        delay control actions by an excess delay, defaults to 0.

    Examples
    --------
    >>> import cbadc.specification
    >>> analog_system, digital_control = cbadc.specification.get_chain_of_integrator(ENOB=12, N=5, BW=1e7, excess_delay=0.0)

    Returns
    -------
    : (:py:class:`cbadc.analog_system.ChainOfIntegrators`, :py:class:`cbadc.digital_control.DigitalControl`)
        returns an analog system and digital control tuple
    """

    if all(param in kwargs for param in ("ENOB", "N", "BW")):
        SNR = enob_to_snr(kwargs["ENOB"])
        snr = snr_from_dB(SNR)
        N = kwargs["N"]
        omega_3dB = 2.0 * np.pi * kwargs["BW"]
        # xi = 1e-1 / (np.pi * (2 * N * 0 + 1))
        xi = 5e-3 / np.pi
        if "xi" in kwargs:
            xi = kwargs["xi"]
        gamma = (xi * snr) ** (1.0 / (2.0 * N))
        beta = -gamma * omega_3dB
        if "local_feedback" in kwargs and kwargs["local_feedback"] is True:
            rho = -omega_3dB / gamma
        else:
            rho = 0
        kappa = beta
        T = 1.0 / np.abs(2.0 * beta)
        all_ones = np.ones(N)
        analog_system = ChainOfIntegrators(
            beta * all_ones, rho * all_ones, kappa * np.eye(N)
        )
        t0 = 0.0
        if "excess_delay" in kwargs:
            t0 = kwargs["excess_delay"] * T
            impulse_response = [StepResponse(t0) for _ in range(N)]
            digital_control = DigitalControl(
                Clock(T), N, impulse_response=impulse_response
            )
        else:
            digital_control = DigitalControl(Clock(T), N)

        analog_system = ChainOfIntegrators(
            beta * all_ones, rho * all_ones, kappa * np.eye(N)
        )
        return (analog_system, digital_control)

    # if all(param in kwargs for param in ('ENOB', 'beta', 'BW')):
    #     snr = snr_from_dB(enob_to_snr(kwargs['ENOB']))
    #     omega_3dB = 2.0 * np.pi * kwargs['BW']
    #     beta = kwargs['beta']
    #     N = int(np.ceil(np.log(snr) / (np.log(beta) - np.log(omega_3dB))))
    #     tmp = snr ** (1.0 / N)
    #     beta_adjusted = tmp * omega_3dB
    #     rho = beta_adjusted / tmp
    #     kappa = beta_adjusted
    #     T = 1.0 / (2.0 * beta_adjusted)
    #     all_ones = np.ones((N, 1))
    #     analog_system = ChainOfIntegrators(
    #         beta_adjusted * all_ones, rho * all_ones, kappa * np.eye(N)
    #     )
    #     digital_control = DigitalControl(Clock(T), N)
    #     return (analog_system, digital_control)
    raise NotImplementedError


def get_leap_frog(**kwargs):
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

    Examples
    --------
    >>> import cbadc.specification
    >>> analog_system, digital_control = cbadc.specification.get_leap_frog(ENOB=12, N=5, BW=1e7, excess_delay=0.0)

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
        xi = 7e-2 / np.pi
        if "xi" in kwargs:
            xi = kwargs["xi"]
        gamma = (xi * snr) ** (1.0 / (2.0 * N))
        beta = -(omega_BW / 2.0) * gamma
        alpha = (omega_BW / 2.0) / gamma
        rho = 0
        if "local_feedback" in kwargs and kwargs["local_feedback"] is True:
            rho = -(omega_BW / 2.0) / gamma * 1e-2 * 0
        T = 1.0 / np.abs(2.0 * beta)
        kappa = beta
        t0 = 0.0
        if "excess_delay" in kwargs:
            t0 = kwargs["excess_delay"] * T
            impulse_response = [StepResponse(t0) for _ in range(N)]
            digital_control = DigitalControl(
                Clock(T), N, impulse_response=impulse_response
            )
        else:
            digital_control = DigitalControl(Clock(T), N)

        analog_system = LeapFrog(
            beta * np.ones(N),
            alpha * np.ones(N - 1),
            rho * np.ones(N),
            kappa * np.eye(N),
        )
        return (analog_system, digital_control)

    raise NotImplementedError
