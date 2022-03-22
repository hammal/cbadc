from cbadc.specification import get_chain_of_integrator, get_leap_frog, get_white_noise
from cbadc.digital_estimator import BatchEstimator
import cbadc
import cbadc.fom
import numpy as np
from tests.fixture.chain_of_integrators import chain_of_integrators

N = 6
ENOB = 14
BW = 1e5
xi = 1
eta2 = 1.0
K1 = 1 << 9
K2 = 1 << 9


def test_get_chain_of_integrator():
    analog_system, digital_control = get_chain_of_integrator(
        N=N, ENOB=ENOB, BW=BW, xi=xi
    )


def test_get_leap_frog():
    analog_system, digital_control = get_leap_frog(N=N, ENOB=ENOB, BW=BW, xi=xi)


def test_get_white_noise():
    analog_system, digital_control = get_chain_of_integrator(
        N=N, ENOB=ENOB, BW=BW, xi=xi
    )
    digital_estimator = BatchEstimator(analog_system, digital_control, eta2, K1, K2)
    white_noise_RMS = get_white_noise(
        digital_estimator,
        (BW * 1e-5, BW),
        cbadc.fom.snr_to_dB(cbadc.fom.enob_to_snr(ENOB)),
        1 / 2.0,
    )
    print(white_noise_RMS)


def test_get_white_noise_leap_frog():
    analog_system, digital_control = get_leap_frog(N=N, ENOB=ENOB, BW=BW, xi=xi)
    digital_estimator = BatchEstimator(analog_system, digital_control, eta2, K1, K2)
    white_noise_RMS = get_white_noise(
        digital_estimator,
        (BW * 1e-5, BW),
        cbadc.fom.snr_to_dB(cbadc.fom.enob_to_snr(ENOB)),
        1 / 2.0,
    )
    print(white_noise_RMS)


def test_verify_get_white_noise_leap_frog():
    analog_system, digital_control = get_leap_frog(N=N, ENOB=ENOB, BW=BW, xi=xi)
    digital_estimator = BatchEstimator(analog_system, digital_control, eta2, K1, K2)
    white_noise_RMS = get_white_noise(
        digital_estimator,
        (BW * 1e-5, BW),
        cbadc.fom.snr_to_dB(cbadc.fom.enob_to_snr(ENOB)),
        1 / 2.0,
    )
    print(white_noise_RMS)
    input_signals = [cbadc.analog_signal.ConstantSignal(0)]
    noise_covariance_matrix = np.diag(white_noise_RMS.flatten())
    simulator = cbadc.simulator.FullSimulator(
        analog_system, digital_control, input_signals, cov_x=noise_covariance_matrix
    )
    digital_estimator(simulator)
    size = 1 << 14
    u_hat = np.zeros(size)
    for index in range(size):
        u_hat[index] = next(digital_estimator)
    print(np.var(u_hat))
