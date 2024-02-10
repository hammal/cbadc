from cbadc.synthesis import get_chain_of_integrator, get_leap_frog
from cbadc.digital_estimator import BatchEstimator
import cbadc
import cbadc.fom
import numpy as np

N = 6
ENOB = 14
BW = 1e5
xi = 1
eta2 = 1.0
K1 = 1 << 9
K2 = 1 << 9


def test_get_white_noise():
    analog_frontend = get_chain_of_integrator(N=N, ENOB=ENOB, BW=BW, xi=xi)
    digital_estimator = BatchEstimator(
        analog_frontend.analog_system, analog_frontend.digital_control, eta2, K1, K2
    )
    white_noise_RMS = digital_estimator.white_noise_balance(
        np.ones(N) / cbadc.fom.enob_to_snr(ENOB),
    )
    print(white_noise_RMS)


def test_get_white_noise_leap_frog():
    analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW, xi=xi)
    digital_estimator = BatchEstimator(
        analog_frontend.analog_system, analog_frontend.digital_control, eta2, K1, K2
    )
    white_noise_RMS = digital_estimator.white_noise_balance(
        np.ones(N) / cbadc.fom.enob_to_snr(ENOB),
    )
    print(white_noise_RMS)


def test_verify_get_white_noise_leap_frog():
    analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW, xi=xi)
    digital_estimator = BatchEstimator(
        analog_frontend.analog_system, analog_frontend.digital_control, eta2, K1, K2
    )
    white_noise_RMS = digital_estimator.white_noise_balance(
        np.ones(N) / cbadc.fom.enob_to_snr(ENOB),
    )
    print(white_noise_RMS)
    input_signal = [cbadc.analog_signal.ConstantSignal(0)]
    noise_covariance_matrix = np.diag(white_noise_RMS.flatten())
    simulator = cbadc.simulator.FullSimulator(
        analog_frontend.analog_system,
        analog_frontend.digital_control,
        input_signal,
        state_noise_covariance_matrix=noise_covariance_matrix,
    )
    digital_estimator(simulator)
    size = 1 << 8
    u_hat = np.zeros(size)
    for index in range(size):
        u_hat[index] = next(digital_estimator)[0]
    print(np.var(u_hat))
