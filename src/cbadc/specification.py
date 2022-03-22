"""Initialize analog systems and digital control from specifications.
"""
from typing import Tuple
from .fom import snr_from_dB, enob_to_snr
from .analog_system.chain_of_integrators import ChainOfIntegrators
from .analog_system.leap_frog import LeapFrog
from .digital_estimator import BatchEstimator
from .digital_control.digital_control import DigitalControl
from .analog_signal.impulse_responses import RCImpulseResponse, StepResponse
from .analog_signal.clock import Clock
import cbadc.fom
import numpy as np
import scipy.integrate


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

    if all(param in kwargs for param in ('ENOB', 'N', 'BW')):
        SNR = enob_to_snr(kwargs['ENOB'])
        snr = snr_from_dB(SNR)
        N = kwargs['N']
        omega_3dB = 2.0 * np.pi * kwargs['BW']
        # xi = 1e-1 / (np.pi * (2 * N * 0 + 1))
        xi = 5e-3 / np.pi
        if 'xi' in kwargs:
            xi = kwargs['xi']
        gi = 2 * N + 1
        if 'gi' in kwargs:
            gi = kwargs['gi']
        gamma = (xi / gi * snr) ** (1.0 / (2.0 * N))
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
        t0 = 0.0
        if 'excess_delay' in kwargs:
            t0 = kwargs['excess_delay'] * T
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

    if all(param in kwargs for param in ('ENOB', 'N', 'BW')):
        SNR = enob_to_snr(kwargs['ENOB'])
        snr = snr_from_dB(SNR)
        N = kwargs['N']
        omega_BW = 2.0 * np.pi * kwargs['BW']
        xi = 7e-2 / np.pi
        if 'xi' in kwargs:
            xi = kwargs['xi']
        gi = 2 ** (2 * N - 1)
        if 'gi' in kwargs:
            gi = kwargs['gi']
        gamma = (xi / gi * snr) ** (1.0 / (2.0 * N))
        beta = -(omega_BW / 2.0) * gamma
        alpha = (omega_BW / 2.0) / gamma
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

        analog_system = LeapFrog(
            beta * np.ones(N),
            alpha * np.ones(N - 1),
            rho * np.ones(N),
            kappa * np.eye(N),
        )
        return (analog_system, digital_control)

    raise NotImplementedError


def get_white_noise(
    digital_estimator: BatchEstimator,
    BW: Tuple[float, float],
    target_SNR: float,
    Pu: float = 1.0,
):
    """Compute white noise components

    Specifically, determine the largest allowed
    white noise disturbance into each state such
    that a given target SNR is mantained for a given
    input signal power.

    Parameters
    ----------
    digital_estimator: :py:class:`cbadc.digital_estimator.BatchEstimator`
        a digital estimator from which to determined the allowed noise.
    BW: [`float`, `float`]
        the bandwidth of interest where BW[0] < BW[1].
    target_SNR: `float`
        the target SNR expressed in dB.
    Pu: `float`, `optional`
        the in-band signal power, defaults to 1.

    Returns
    -------
    : array_like, shape=(L, N)
        upper bound of the largest RMS valued white noise terms entering
        into each corresponding state specified in rms / sqrt(Hz).
    """
    if BW[0] >= BW[1]:
        raise Exception("Bandwith must be specified as interval were BW[1] > BW[0].")

    ntf = digital_estimator.noise_transfer_function
    stf = digital_estimator.analog_system.transfer_function_matrix

    noise_power = np.zeros(
        (digital_estimator.analog_system.L, digital_estimator.analog_system.N)
    )

    signal_power = np.zeros(digital_estimator.analog_system.L)
    noise_local_scale = np.zeros_like(noise_power)

    def derivative(omega, x):
        _omega = np.array([omega])
        return (
            np.abs(
                np.tensordot(
                    ntf(_omega),
                    stf(_omega, general=False),
                    axes=((1, 2), (0, 2)),
                )
            )
            ** 2
        )

    signal_power = (
        scipy.integrate.solve_ivp(
            derivative,
            (BW[0], BW[1]),
            # (digital_control._impulse_response[m].t0, Ts),
            np.zeros(digital_estimator.analog_system.L),
            # atol=atol,
            # rtol=rtol,
            # max_step=max_step,
            # method="Radau",
            # jacobian=tempAf,
            # events=(impulse_start,),
        ).y[:, -1]
        * Pu
        / (BW[1] - BW[0])
    )

    for n in range(digital_estimator.analog_system.N):

        def derivative(omega, x):
            _omega = np.array([omega])
            return (
                np.abs(
                    # np.tensordot(
                    #     ntf(_omega),
                    #     stf(_omega, general=True)[:, n, :],
                    #     axes=((1, 2), (0, 1)),
                    # )
                    np.einsum(
                        'ijk,jk->i', ntf(_omega), stf(_omega, general=True)[:, n, :]
                    )
                )
                ** 2
            )

        noise_power[:, n] = scipy.integrate.solve_ivp(
            derivative,
            (BW[0], BW[1]),
            # (digital_control._impulse_response[m].t0, Ts),
            np.zeros(digital_estimator.analog_system.L),
            # atol=atol,
            # rtol=rtol,
            # max_step=max_step,
            # method="Radau",
            # jacobian=tempAf,
            # events=(impulse_start,),
        ).y[:, -1]

        noise_local_scale[:, n] = (
            1.0 / noise_power[:, n] / digital_estimator.analog_system.N
        )

    global_scale = np.sum(signal_power) / (cbadc.fom.snr_from_dB(target_SNR) ** 2)

    input_referred_noise_power = global_scale * noise_local_scale
    print(
        f"noise_power ={noise_power}, noise_local_scale={noise_local_scale}, global_scale={global_scale}"
    )
    # return np.sqrt(input_referred_noise_power)
    return input_referred_noise_power
