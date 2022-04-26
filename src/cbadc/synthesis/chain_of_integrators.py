"""Synthesise functions for chain-of-integrators control-bounded ADCs."""
from cbadc.fom import snr_from_dB, enob_to_snr, snr_to_enob
from cbadc.analog_system.chain_of_integrators import ChainOfIntegrators
from cbadc.digital_control.digital_control import DigitalControl
from cbadc.analog_signal.impulse_responses import RCImpulseResponse, StepResponse
from cbadc.analog_signal.clock import Clock
from cbadc.analog_frontend import AnalogFrontend
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
    return 2.0 * N + 1.0


def get_chain_of_integrator(**kwargs) -> AnalogFrontend:
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


    Returns
    -------
    : (:py:class:`cbadc.analog_system.ChainOfIntegrators`, :py:class:`cbadc.digital_control.DigitalControl`)
        returns an analog system and digital control tuple
    """

    if all(param in kwargs for param in ('ENOB', 'N', 'BW')):
        SNR = enob_to_snr(kwargs['ENOB'])
        snr = snr_from_dB(SNR)
        N = kwargs['N']
        omega_3dB = 2.0 * np.pi * kwargs['BW']
        # xi = 1e-1 / (np.pi * (2 * N * 0 + 1))
        xi = kwargs.get('xi', 2.3e-3)
        gamma = (xi / g_i(N) * snr) ** (1.0 / (2.0 * N))
        beta = -gamma * omega_3dB
        if 'local_feedback' in kwargs and kwargs['local_feedback'] is True:
            rho = -omega_3dB / gamma
        else:
            rho = 0
        kappa = beta
        T = 1.0 / np.abs(2.0 * beta)
        all_ones = np.ones(N)
        analog_system = ChainOfIntegrators(
            beta * all_ones, rho * all_ones, kappa * np.eye(N)
        )
        t0 = T * kwargs.get('excess_delay', 0.0)
        if kwargs.get('digital_control') == 'switch_cap':
            scale = 5e1
            tau = 1.0 / (np.abs(beta) * scale)
            v_cap = 0.5
            kappa = v_cap * beta * scale
            impulse_response = RCImpulseResponse(tau, t0)
            digital_control = DigitalControl(
                Clock(T), N, impulse_response=impulse_response
            )
            # print(impulse_response.evaluate(T))
        else:
            impulse_response = StepResponse(t0)
            digital_control = DigitalControl(
                Clock(T), N, impulse_response=impulse_response
            )
            digital_control = DigitalControl(Clock(T), N)

        analog_system = ChainOfIntegrators(
            beta * all_ones, rho * all_ones, kappa * np.eye(N)
        )
        return AnalogFrontend(analog_system, digital_control)
    elif all(param in kwargs for param in ('SNR', 'N', 'BW')):
        return get_chain_of_integrator(ENOB=snr_to_enob(kwargs['SNR']), **kwargs)
    elif all(param in kwargs for param in ('OSR', 'N', 'BW')):
        N = kwargs['N']
        BW = kwargs['BW']
        OSR = kwargs['OSR']
        T = 1.0 / (2 * BW * OSR)
        beta = 1 / (2 * T)
        kappa = beta
        rho = kwargs.get('rho', 0.0)
        analog_system = ChainOfIntegrators(
            beta * np.ones(N),
            rho * np.ones(N),
            kappa * np.eye(N),
        )
        digital_control = DigitalControl(Clock(T), N)
        return AnalogFrontend(analog_system, digital_control)
    elif all(param in kwargs for param in ('OSR', 'N', 'T')):
        N = kwargs['N']
        T = kwargs['T']
        OSR = kwargs['OSR']
        BW = 1.0 / (2 * T * OSR)
        rho = kwargs.get('rho', 0.0)
        return get_chain_of_integrator(N=N, OSR=OSR, BW=BW, rho=rho)
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
