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

    Examples
    --------
    >>> import cbadc.specification
    >>> analog_system, digital_control = cbadc.specification.get_chain_of_integrator(ENOB=12, N=5, BW=1e7, excess_delay=0.0)

    Returns
    -------
    : (:py:class:`cbadc.analog_system.ChainOfIntegrators`, :py:class:`cbadc.digital_control.DigitalControl`)
        returns an analog system and digital control tuple
    """

    if all(param in kwargs for param in ('ENOB', 'N', 'BW')):
        SNR = enob_to_snr(kwargs['ENOB'])
        N = kwargs['N']
        omega_3dB = 2.0 * np.pi * kwargs['BW']
        xi = 5e-2 / (np.pi * (2 * N + 1))
        gain_per_stage = 10 ** ((xi * SNR) / (20 * N))
        beta = gain_per_stage * omega_3dB
        rho = -np.abs(omega_3dB)
        kappa = beta
        T = 1.0 / (2.0 * beta)
        all_ones = np.ones(N)
        analog_system = ChainOfIntegrators(
            beta * all_ones, rho * all_ones, kappa * np.eye(N)
        )
        t0 = kwargs['excess_delay'] * T
        if kwargs.get('digital_control') == 'switch_cap':
            scale = 1e2
            tau = 1.0 / (beta * scale)
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

    Examples
    --------
    >>> import cbadc.specification
    >>> analog_system, digital_control = cbadc.specification.get_leap_frog(ENOB=12, N=5, BW=1e7, excess_delay=0.0)

    Returns
    -------
    : (:py:class:`cbadc.analog_system.LeapFrog`, :py:class:`cbadc.digital_control.DigitalControl`)
        returns an analog system and digital control tuple
    """

    if all(param in kwargs for param in ('ENOB', 'N', 'BW')):
        SNR = enob_to_snr(kwargs['ENOB'])
        N = kwargs['N']
        omega_BW = 2.0 * np.pi * kwargs['BW']
        xi = 1e-1 / np.pi
        beta = 10 ** ((xi * SNR) / (20 * N))
        forward_gain = omega_BW / 2 * beta
        feeback_gain = -omega_BW / 2 / beta
        T = 1.0 / (2.0 * forward_gain)
        kappa = forward_gain
        all_ones = np.ones(N)
        all_but_one_ones = np.ones_like(all_ones)
        all_but_one_ones[0] = 0.0
        t0 = kwargs['excess_delay'] * T
        if kwargs.get('digital_control') == 'switch-cap':
            scale = 1e2
            tau = 1.0 / (forward_gain * scale)
            v_cap = 0.5
            kappa = v_cap * forward_gain * scale
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

        analog_system = LeapFrog(
            forward_gain * all_ones, feeback_gain * all_but_one_ones, kappa * np.eye(N)
        )
        return (analog_system, digital_control)

    raise NotImplementedError
