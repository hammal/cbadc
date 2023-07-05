"""Synthesise functions for chain-of-integrators control-bounded ADCs."""
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_system import AnalogSystem
from cbadc.synthesis.leap_frog import g_i, get_leap_frog
from cbadc.digital_control.digital_control import DigitalControl, StepResponse, Clock
from cbadc.digital_control import ModulatorControl
from cbadc.analog_system.topology import stack
from cbadc.fom import enob_to_snr, snr_from_dB, snr_to_enob

import numpy as np
import logging

logger = logging.getLogger(__name__)


def _analog_system_factory(N, beta, rho, kappa, gamma, omega_p):
    A = np.zeros((2 * N, 2 * N))
    B = np.zeros((2 * N, 2))
    CT = np.eye(2 * N)
    Gamma = kappa * np.eye(2 * N)
    Gamma_tildeT = -np.sign(kappa) * np.eye(2 * N)
    for i in range(N):
        A[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.array(
            [[rho, -omega_p], [omega_p, rho]]
        )
        if i < N - 1:
            A[2 * i : 2 * i + 2, 2 * (i + 1) : 2 * (i + 1) + 2] = gamma * np.eye(2)
            A[2 * (i + 1) : 2 * (i + 1) + 2, 2 * i : 2 * i + 2] = beta * np.eye(2)

        B[:2, :] = beta * np.eye(2)

    return AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)


def _rotation_matrix(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def get_bandpass(**kwargs) -> AnalogFrontend:
    if not all(param in kwargs for param in ('analog_frontend', 'fc')):
        raise NotImplementedError
    analog_frontend_baseband = kwargs['analog_frontend']
    analog_system_baseband = analog_frontend_baseband.analog_system
    digital_control_baseband = analog_frontend_baseband.digital_control
    N = analog_system_baseband.N
    beta = analog_system_baseband.Gamma[0, 0]
    T = digital_control_baseband.clock.T
    fc = kwargs['fc']
    omega_c = 2 * np.pi * fc

    analog_system = stack([analog_system_baseband, analog_system_baseband])
    analog_system.A[:N, N:] = -omega_c * np.eye(N)
    analog_system.A[N:, :N] = omega_c * np.eye(N)
    kappa = beta * T * omega_c / (2 * np.sin(omega_c * T / 2))
    phi: float = kwargs.get('phi', omega_c / 2 - np.pi)
    delta_DC: float = kwargs.get("delta_DC", 0.0)

    Gamma = np.zeros((2 * N, 2 * N))
    Gamma_tildeT = np.zeros((2 * N, 2 * N))

    if kwargs.get('modulator', False):
        logger.info("Using modulator")
        analog_system.Gamma = np.eye(2 * N) * kappa
        analog_system.Gamma_tildeT = -np.sign(kappa) * np.eye(2 * N)
        digital_control = digital_control = ModulatorControl(
            Clock(T, tt=1 / fc * 1e-3),
            2 * digital_control_baseband.M,
            fc,
        )
    else:
        logger.info("Using non-modulator")
        kappa = beta * T * omega_c / (2 * np.sin(omega_c * T / 2)) * np.cos(phi)
        bar_kappa = beta * T * omega_c / (2 * np.sin(omega_c * T / 2)) * np.sin(phi)
        tilde_kappa = -1 / (beta * T) * np.cos(omega_c * (T / 2 + delta_DC) - phi)
        bar_tilde_kappa = -1 / (beta * T) * np.sin(omega_c * (T / 2 + delta_DC) - phi)
        # gamma_temp = kappa * _rotation_matrix(phi)
        # gamma_tilde_temp = omega_c / (2 * kappa * np.sin(omega_c * T / 2)) * _rotation_matrix(omega_c * (T / 2 + delta_DC) - phi - np.pi)
        for i in range(N):
            # Gamma[2*i:2*i + 2, 2*i: 2 * i + 2] = kappa * _rotation_matrix(phi)
            Gamma[i, i] = kappa
            Gamma[i + N, i + N] = kappa
            Gamma[i, i + N] = -bar_kappa
            Gamma[i + N, i] = bar_kappa

            Gamma_tildeT[i, i] = tilde_kappa
            Gamma_tildeT[i + N, i + N] = tilde_kappa
            Gamma_tildeT[i, i + N] = -bar_tilde_kappa
            Gamma_tildeT[i + N, i] = bar_tilde_kappa
        #     # Gamma_tildeT[2*i:2*i + 2, 2*i: 2 * i + 2] = omega_p / (2 * kappa * np.sin(omega_p * T / 2)) * _rotation_matrix(omega_p * T / 2 - phi - np.pi)
        #     # Gamma_tildeT[2*i:2*i + 2, 2*i: 2 * i + 2] = _rotation_matrix(omega_p * T)
        analog_system.Gamma = Gamma
        analog_system.Gamma_tildeT = Gamma_tildeT

        digital_control = digital_control = DigitalControl(
            digital_control_baseband.clock,
            2 * digital_control_baseband.M,
            impulse_response=StepResponse(delta_DC, 1.0),
        )

    return AnalogFrontend(
        analog_system=AnalogSystem(
            analog_system.A,
            analog_system.B,
            analog_system.CT,
            analog_system.Gamma,
            analog_system.Gamma_tildeT,
        ),
        digital_control=digital_control,
    )


def get_chain_of_oscillators(**kwargs) -> AnalogFrontend:
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
        target bandwidth in Hz
    fp: `float`,
        oscillator resonant frequency in Hz
    xi: `float`, `optional`
        a proportionality constant, defaults to 0.0016.
    local_feedback: `bool`, `optional`
        include local feedback, defaults to False.
    excess_delay: `float`, `optional`
        delay control actions by an excess delay, defaults to 0.
    finite_gain: `bool`, `optional`
        include finite gain, defaults to False.


    Returns
    -------
    : (:py:class:`cbadc.analog_system.ChainOfIntegrators`, :py:class:`cbadc.digital_control.DigitalControl`)
        returns an analog system and digital control tuple
    """
    finite_gain = kwargs.get("finite_gain", False)

    if all(param in kwargs for param in ('ENOB', 'N', 'BW', 'fp')):
        SNR = enob_to_snr(kwargs['ENOB'])
        snr = snr_from_dB(SNR)
        N = kwargs['N']
        omega_BW = 2.0 * np.pi * kwargs['BW']
        xi = kwargs.get('xi', 4e-3)
        gamma = (xi / g_i(N) * snr) ** (1.0 / (2.0 * N))
        omega_BW = omega_BW / 2.0
        beta = -omega_BW * (2.0 * gamma)
        alpha = omega_BW / (2.0 * gamma)
        rho = 0
        if 'local_feedback' in kwargs and kwargs['local_feedback'] is True:
            rho = -(omega_BW / 2.0) / gamma * 1e-2 * 0
        T = 1.0 / np.abs(2.0 * beta * 10)
        kappa = beta
        omega_p = 2 * np.pi * kwargs['fp']

        analog_system = _analog_system_factory(N, beta, rho, kappa, alpha, omega_p)

        t0 = T * kwargs.get('excess_delay', 0.0)
        if kwargs.get('digital_control') == 'switch_cap':
            raise NotImplementedError
        else:
            impulse_response = StepResponse(t0)
            digital_control = DigitalControl(
                Clock(T), 2 * N, impulse_response=impulse_response
            )
            digital_control = DigitalControl(Clock(T), 2 * N)

        if finite_gain:
            analog_system.A += -omega_BW / (gamma ** (2 * N)) * np.eye(N)

        return AnalogFrontend(analog_system, digital_control)
    elif all(param in kwargs for param in ('SNR', 'N', 'BW', 'fp')):
        return get_chain_of_oscillators(ENOB=snr_to_enob(kwargs['SNR']), **kwargs)
    elif all(param in kwargs for param in ('OSR', 'N', 'BW', 'fp')):
        N = kwargs['N']
        BW = kwargs['BW']
        OSR = kwargs['OSR']
        T = 1.0 / (2 * BW * OSR)
        beta = 1 / (2 * T)
        kappa = beta
        rho = kwargs.get('rho', 0.0)
        omega_BW = 2.0 * np.pi * BW
        alpha = -((omega_BW / 2) ** 2) / beta
        omega_p = 2 * np.pi * kwargs['fp']

        analog_system = _analog_system_factory(N, beta, rho, kappa, alpha, omega_p)

        digital_control = DigitalControl(Clock(T), N)
        return AnalogFrontend(analog_system, digital_control)
    elif all(param in kwargs for param in ('OSR', 'N', 'T', 'fp')):
        N = kwargs['N']
        T = kwargs['T']
        OSR = kwargs['OSR']
        BW = 1.0 / (2 * T * OSR)
        rho = kwargs.get('rho', 0.0)
        fp = kwargs['fp']
        return get_chain_of_oscillators(N=N, OSR=OSR, BW=BW, rho=rho, fp=fp)
    raise NotImplementedError


def get_parallel_leapfrog(**kwargs) -> AnalogSystem:
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
        per system order.
    M: `int`
        system width
    BW: `float`
        target bandwidth in Hz
    fp: `float`,
        oscillator resonant frequency in Hz
    xi: `float`, `optional`
        a proportionality constant, defaults to 0.0016.
    local_feedback: `bool`, `optional`
        include local feedback, defaults to False.
    excess_delay: `float`, `optional`
        delay control actions by an excess delay, defaults to 0.
    finite_gain: `bool`, `optional`
        include finite gain, defaults to False.


    Returns
    -------
    : (:py:class:`cbadc.analog_system.ChainOfIntegrators`, :py:class:`cbadc.digital_control.DigitalControl`)
        returns an analog system and digital control tuple
    """
    # finite_gain = kwargs.get("finite_gain", False)

    if all(param in kwargs for param in ('ENOB', 'N', 'M', 'BW', 'fp')):
        ENOB = kwargs['ENOB']
        N = kwargs['N']
        M = kwargs['M']
        BW = kwargs['BW']
        fp = kwargs['fp']
        omega_p = 2 * np.pi * fp
        TotMN = np.sum([i for i in range(1, M)])
        poles = kwargs.get('fps', np.ones(TotMN) * omega_p)

        # SNR = enob_to_snr(kwargs['ENOB'])
        # snr = snr_from_dB(SNR)
        # omega_BW = 2.0 * np.pi * kwargs['BW']
        # xi = kwargs.get('xi', 4e-3)
        # gamma = (xi / g_i(N) * snr) ** (1.0 / (2.0 * N))
        # omega_BW = omega_BW / 2.0
        # beta = -omega_BW * (2.0 * gamma)
        # alpha = omega_BW / (2.0 * gamma)
        # rho = 0
        # T = 1.0 / np.abs(2.0 * beta * 10)
        # kappa = beta
        # omega_p = 2 * np.pi * kwargs['fp']

        analog_system = stack(
            [get_leap_frog(ENOB=ENOB, N=N, BW=BW).analog_system for _ in range(M)]
        )

        if M == 3:
            analog_system.A[N : 2 * N, 0:N] = poles[0] * np.eye(N)
            analog_system.A[
                0:N,
                N : 2 * N,
            ] = -poles[
                0
            ] * np.eye(N)

            analog_system.A[2 * N : 3 * N, 0:N] = poles[1] * np.eye(N)
            analog_system.A[0:N, 2 * N : 3 * N] = -poles[1] * np.eye(N)

            analog_system.A[2 * N : 3 * N, N : 2 * N] = poles[2] * np.eye(N)
            analog_system.A[N : 2 * N, 2 * N : 3 * N] = -poles[2] * np.eye(N)
        elif M == 4:
            analog_system.A[N : 2 * N, 0:N] = poles[0] * np.eye(N)
            analog_system.A[
                0:N,
                N : 2 * N,
            ] = -poles[
                0
            ] * np.eye(N)

            analog_system.A[2 * N : 3 * N, 0:N] = poles[1] * np.eye(N)
            analog_system.A[0:N, 2 * N : 3 * N] = -poles[1] * np.eye(N)
            analog_system.A[2 * N : 3 * N, N : 2 * N] = poles[2] * np.eye(N)
            analog_system.A[N : 2 * N, 2 * N : 3 * N] = -poles[2] * np.eye(N)

            analog_system.A[3 * N : 4 * N, 0:N] = poles[3] * np.eye(N)
            analog_system.A[0:N, 3 * N : 4 * N] = -poles[3] * np.eye(N)
            analog_system.A[3 * N : 4 * N, N : 2 * N] = poles[4] * np.eye(N)
            analog_system.A[N : 2 * N, 3 * N : 4 * N] = -poles[4] * np.eye(N)
            analog_system.A[3 * N : 4 * N, 2 * N : 3 * N] = poles[5] * np.eye(N)
            analog_system.A[2 * N : 3 * N, 3 * N : 4 * N] = -poles[5] * np.eye(N)
        else:
            for i in range(1, M):
                for j in range(i):
                    print(i, j, (M - 1 - j), (M - j))
                    analog_system.A[i * N : (i + 1) * N, j * N : (j + 1) * N] = poles[
                        i + j * M
                    ] * np.eye(
                        N
                    )  # / (i + 1) * (j)
                    analog_system.A[
                        (M - i - 1) * N : (M - i) * N, (M - 1 - j) * N : (M - j) * N
                    ] = -poles[-(i + j * M)] * np.eye(
                        N
                    )  # / (i + 1) * (j )

        analog_system.B = np.sum(analog_system.B, axis=1).reshape(
            (analog_system.B.shape[0], 1)
        )
        analog_system.D = np.sum(analog_system.D, axis=1).reshape(
            (analog_system.D.shape[0], 1)
        )
        analog_system.B_tilde = np.sum(analog_system.B_tilde, axis=1).reshape(
            (analog_system.B_tilde.shape[0], 1)
        )
        analog_system.L = 1
        return AnalogSystem(
            analog_system.A,
            analog_system.B,
            analog_system.CT,
            analog_system.Gamma,
            analog_system.Gamma_tildeT,
        )

        # t0 = T * kwargs.get('excess_delay', 0.0)
        # if kwargs.get('digital_control') == 'switch_cap':
        #     raise NotImplementedError
        # else:
        #     impulse_response = StepResponse(t0)
        #     digital_control = DigitalControl(
        #         Clock(T), 2 * N, impulse_response=impulse_response
        #     )
        #     digital_control = DigitalControl(Clock(T), 2 * N)

        # if finite_gain:
        #     analog_system.A += -omega_BW / (gamma ** (2 * N)) * np.eye(N)

        # return AnalogFrontend(analog_system, digital_control)

    raise NotImplementedError
