import cProfile
from cbadc.simulator.numerical_simulator import PreComputedControlSignalsSimulator
import cbadc
from tests.performance_validation.fixtures import setup_filter
import numpy as np
from scipy.signal import firwin2
from cbadc.fom import snr_to_dB, snr_to_enob
import logging

logger = logging.getLogger(__name__)

N = 4
ENOB = 12
BW = 1e6
analog_system = 'leap_frog'
digital_control = 'default'
calibration_method = {
    "name": "rls",
    "label": "rls_K_9_v2",
    "kwargs": {
        "learning_rate": cbadc.digital_estimator.FixedStepSize(1.0 - 1e-12),
        "delta": 1e-2,
        "projection": False,
        "random_order": False,
        "batch_size": 1,
        "K": lambda x: 1 << 9,
        "epochs": 1 << 16,
    },
}
evaluation_data_size = 1 << 12
training_data_size = 1 << 16
evaluation_interval = 1 << 11
kappa_0_scale = 0.1
simulation_method = PreComputedControlSignalsSimulator


def evaluate(u_hat: np.ndarray, fs: float, BW: float, iteration: int):
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat,
        fs=fs,
        nperseg=u_hat.size,
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=fs
    )
    est_SNR = cbadc.fom.snr_to_dB(fom["snr"])
    est_ENOB = cbadc.fom.snr_to_enob(est_SNR)
    return {
        "u_hat": u_hat,
        "psd": psd,
        "f": f,
        "est_SNR": est_SNR,
        "est_ENOB": est_ENOB,
        "iteration": iteration,
        "t": np.arange(u_hat.size) / fs,
    }


def test_calibration(
    N,
    ENOB,
    BW,
    analog_system,
    digital_control,
    calibration_method,
    evaluation_data_size,
    training_data_size,
    evaluation_interval,
    kappa_0_scale,
    simulation_method,
):
    # if N >= 12 and analog_system == 'leap_frog' and BW >= 1e8:
    #     pytest.skip("Known limitation")

    # if N >= 12 and digital_control == 'switch-cap' and analog_system == 'leap_frog':
    #     pytest.skip("Filter coefficients not good enough.")

    K = [calibration_method['kwargs']['K'](x) for x in range(N + 1)]

    res = setup_filter(N, ENOB, BW, analog_system, digital_control, 0.0)
    analog_system = res['analog_system']
    ref_vector = np.zeros((N, 1))
    ref_vector[0] = analog_system.Gamma[0, 0] * kappa_0_scale
    analog_system = cbadc.analog_system.AnalogSystem(
        res['analog_system'].A,
        res['analog_system'].B,
        res['analog_system'].CT,
        np.hstack(
            (
                ref_vector,
                res['analog_system'].Gamma,
            )
        ),
        res['analog_system'].Gamma_tildeT,
    )
    digital_control = res['digital_control']
    K1 = max(K) // 2
    K2 = K1
    eta2 = (
        np.linalg.norm(
            analog_system.transfer_function_matrix(np.array([2 * np.pi * BW]))
        )
        ** 2
    )

    # Simulator
    wiener_filter_reference = cbadc.digital_estimator.FIRFilter(
        analog_system,
        cbadc.digital_control.DitherControl(
            1,
            cbadc.digital_control.DigitalControl(
                cbadc.analog_signal.Clock(digital_control.clock.T, tt=1e-14, td=0.0),
                N,
            ),
        ),
        eta2,
        K1=K1,
        K2=K2,
    )
    fs = 1 / digital_control.clock.T
    fw = BW * digital_control.clock.T * 2
    h0 = firwin2(K[0], [0, 0.9 * fw, fw, 1], [1, 1, 1e-3, 0]) * kappa_0_scale
    reference_index = [0]
    h = cbadc.digital_estimator.fir_estimator.initial_filter([h0], K, reference_index)
    if calibration_method["name"] == "rls":
        delta = calibration_method["kwargs"]["delta"]
        lambda_ = calibration_method["kwargs"]["learning_rate"]
        projection = calibration_method["kwargs"]["projection"]
        random_order = calibration_method["kwargs"]["random_order"]

        adaptive_filter = cbadc.digital_estimator.AdaptiveFIRFilter(
            h,
            K,
            reference_index,
            downsample=1,
            method="rls",
            delta=delta,
            learning_rate=lambda_,
            projection=projection,
            randomized_order=random_order,
        )
    elif calibration_method["name"] == "lms":
        learning_rate = calibration_method["kwargs"]["learning_rate"]
        momentum = calibration_method["kwargs"]["momentum"]
        projection = calibration_method["kwargs"]["projection"]
        random_order = calibration_method["kwargs"]["random_order"]
        adaptive_filter = cbadc.digital_estimator.AdaptiveFIRFilter(
            h,
            K,
            reference_index,
            downsample=1,
            method="lms",
            learning_rate=learning_rate,
            momentum=momentum,
            projection=projection,
            randomized_order=random_order,
        )
    elif calibration_method["name"] == "adam":
        learning_rate = calibration_method["kwargs"]["learning_rate"]
        beta1 = calibration_method["kwargs"]["beta1"]
        beta2 = calibration_method["kwargs"]["beta2"]
        projection = calibration_method["kwargs"]["projection"]
        random_order = calibration_method["kwargs"]["random_order"]
        adaptive_filter = cbadc.digital_estimator.AdaptiveFIRFilter(
            h,
            K,
            reference_index,
            downsample=1,
            method="adam",
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            projection=projection,
            randomized_order=random_order,
        )
    else:
        raise ValueError("Unknown method")

    amplitude = 0.5
    frequency = 1.0 / res['digital_control'].clock.T
    while frequency > BW / 5:
        frequency /= 2
    phase = 0.0
    offset = 0.0
    testing_input = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)
    training_input = cbadc.analog_signal.ConstantSignal(offset)
    testing_reference = simulation_method(
        analog_system,
        cbadc.digital_control.DitherControl(
            1,
            cbadc.digital_control.DigitalControl(
                cbadc.analog_signal.Clock(digital_control.clock.T, tt=1e-14, td=0.0),
                N,
            ),
        ),
        [testing_input],
    )
    testing_validation = simulation_method(
        analog_system,
        cbadc.digital_control.DitherControl(
            1,
            cbadc.digital_control.DigitalControl(
                cbadc.analog_signal.Clock(digital_control.clock.T, tt=1e-14, td=0.0),
                N,
            ),
        ),
        [testing_input],
    )
    training = simulation_method(
        analog_system,
        cbadc.digital_control.DitherControl(
            1,
            cbadc.digital_control.DigitalControl(
                cbadc.analog_signal.Clock(digital_control.clock.T, tt=1e-14, td=0.0),
                N,
            ),
        ),
        [training_input],
    )

    wiener_filter_reference(testing_reference)
    adaptive_filter(
        training, testing_validation, training_data_size, evaluation_data_size
    )

    training_errors = []
    testing_errors = []
    u_hat = np.zeros(evaluation_data_size)

    for epoch in range(calibration_method['kwargs']['epochs']):
        training_errors.append(
            adaptive_filter.train(calibration_method['kwargs']['batch_size'])
        )
        if epoch % evaluation_interval == 0:
            for i in range(evaluation_data_size):
                u_hat[i] = next(adaptive_filter)
            testing_errors.append(
                evaluate(
                    u_hat[max(K) :],
                    fs,
                    BW,
                    epoch * calibration_method['kwargs']['batch_size'],
                )
            )
            logger.info(
                "Training Error: %.2e, Testing SNR: %.2f dB, Epoch: %d",
                training_errors[-1],
                testing_errors[-1]["est_SNR"],
                epoch,
            )

    u_hat_ref = np.zeros(evaluation_data_size)
    u_hat_val = np.zeros(evaluation_data_size)
    for index in range(evaluation_data_size):
        u_hat_ref[index] = next(wiener_filter_reference)
    for index in range(evaluation_data_size):
        u_hat_val[index] = next(adaptive_filter)
    f, psd_ref = cbadc.utilities.compute_power_spectral_density(
        u_hat_ref[K1 + K2 :], fs=1 / digital_control.clock.T, nperseg=u_hat_ref.size
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd_ref, 15)
    noise_index = np.ones(psd_ref.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd_ref, signal_index, noise_index, fs=1 / digital_control.clock.T
    )
    est_SNR_ref = snr_to_dB(fom['snr'])
    est_ENOB_ref = snr_to_enob(est_SNR_ref)

    f, psd_val = cbadc.utilities.compute_power_spectral_density(
        u_hat_val[K1 + K2 :], fs=1 / digital_control.clock.T, nperseg=u_hat_val.size
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd_val, 15)
    noise_index = np.ones(psd_val.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd_val, signal_index, noise_index, fs=1 / digital_control.clock.T
    )
    est_SNR_val = snr_to_dB(fom['snr'])
    est_ENOB_val = snr_to_enob(est_SNR_val)


def test_profile():
    test_calibration(
        N,
        ENOB,
        BW,
        analog_system,
        digital_control,
        calibration_method,
        evaluation_data_size,
        training_data_size,
        evaluation_interval,
        kappa_0_scale,
        simulation_method,
    )


cProfile.run('test_profile()')
