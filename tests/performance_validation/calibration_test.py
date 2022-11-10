import pytest
import numpy as np
import cbadc
import logging
import matplotlib.pyplot as plt

from cbadc.simulator.numerical_simulator import PreComputedControlSignalsSimulator
from tests.performance_validation.fixtures import setup_filter
from scipy.signal import firwin2
from cbadc.fom import snr_to_dB, snr_to_enob

DEBUG = True

logger = logging.getLogger(__name__)


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


@pytest.mark.parametrize(
    "N",
    [
        # pytest.param(2, id="N=2"),
        pytest.param(4, id="N=4"),
        # pytest.param(6, id="N=6"),
        # pytest.param(8, id="N=8"),
        # pytest.param(10, id="N=10"),
        # pytest.param(12, id="N=12"),
        # pytest.param(14, id="N=14"),
        # pytest.param(16, id="N=16"),
    ],
)
@pytest.mark.parametrize(
    "ENOB",
    [
        # pytest.param(10, id="ENOB=10"),
        pytest.param(12, id="ENOB=12"),
        # pytest.param(14, id="ENOB=14"),
        # pytest.param(16, id="ENOB=16"),
        # pytest.param(20, id="ENOB=20"),
        # pytest.param(23, id="ENOB=23"),
    ],
)
@pytest.mark.parametrize(
    "BW",
    [
        # pytest.param(1e0, id="BW=1Hz"),
        # pytest.param(1e1, id="BW=10Hz"),
        # pytest.param(1e2, id="BW=100Hz"),
        # pytest.param(1e3, id="BW=1kHz"),
        # pytest.param(1e4, id="BW=10kHz"),
        # pytest.param(1e5, id="BW=100kHz"),
        pytest.param(1e6, id="BW=1MHz"),
        # pytest.param(1e7, id="BW=10MHz"),
        # pytest.param(1e8, id="BW=100MHz"),
        # pytest.param(1e9, id="BW=1GHz"),
    ],
)
@pytest.mark.parametrize(
    "analog_system",
    [
        pytest.param('chain-of-integrators', id="chain_of_integrators_as"),
        pytest.param('leap_frog', id="leap_frog_as"),
    ],
)
@pytest.mark.parametrize(
    "digital_control",
    [
        pytest.param('default', id="default_dc"),
        # pytest.param('switch-cap', id="switch_cap_dc"),
    ],
)
@pytest.mark.parametrize(
    "calibration_method",
    [
        # pytest.param(
        # {
        #     "name": "rls",
        #     "label": "rls_K_7_v1",
        #     "kwargs": {
        #         "learning_rate": cbadc.digital_estimator.FixedStepSize(1.0 - 1e-12),
        #         "delta": 1e-2,
        #         "projection": False,
        #         "random_order": False,
        #         "batch_size": 1,
        #         "K": [1 << (7) for _ in range(N + L)],
        #     },
        # }, id='rls_K_7_v1'),
        # pytest.param(
        #     {
        #         "name": "rls",
        #         "label": "rls_K_9_v1",
        #         "kwargs": {
        #             "learning_rate": cbadc.digital_estimator.FixedStepSize(1.0 - 1e-12),
        #             "delta": 1e-2,
        #             "projection": False,
        #             "random_order": False,
        #             "batch_size": 1,
        #             "K": lambda x: 1 << 9,
        #             "epochs": 1 << 14,
        #         },
        #     },
        #     id='rls_K_9_v1',
        # ),
        pytest.param(
            {
                "name": "rls",
                "label": "rls_K_9_v2",
                "kwargs": {
                    "learning_rate": cbadc.digital_estimator.FixedStepSize(1.0 - 1e-12),
                    "delta": 1e-2,
                    "projection": False,
                    "random_order": False,
                    "batch_size": 1,
                    "K": lambda x: 1 << 9,
                    "epochs": 1 << 14,
                },
            },
            id='rls_K_9_v2',
        ),
    ],
)
@pytest.mark.parametrize(
    "evaluation_data_size",
    [
        pytest.param(1 << 12, id="evaluation_data_size=4096"),
    ],
)
@pytest.mark.parametrize(
    "training_data_size",
    [
        pytest.param(1 << 14, id="training_data_size=4096"),
    ],
)
@pytest.mark.parametrize("evaluation_interval", [pytest.param(1 << 11, id="2048")])
@pytest.mark.parametrize(
    "kappa_0_scale",
    [
        pytest.param(0.001, id="0.001"),
        # pytest.param(0.01, id="0.01"),
        pytest.param(0.1, id="0.1"),
        # pytest.param(0.25, id="0.25")
    ],
)
@pytest.mark.parametrize(
    'simulation_method',
    [
        # pytest.param(AnalyticalSimulator, id="an_sim"),
        # pytest.param(FullSimulator, id="full_num_sim"),
        pytest.param(PreComputedControlSignalsSimulator, id="pre_num_sim"),
        # pytest.param(MPSimulator, id="mp_sim"),
    ],
)
@pytest.mark.parametrize(
    'dither_dynamic_type',
    [
        'binary',
        'ternary',
        'uniform',
    ],
)
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
    dither_dynamic_type,
):
    # if N >= 12 and analog_system == 'leap_frog' and BW >= 1e8:
    #     pytest.skip("Known limitation")

    # if N >= 12 and digital_control == 'switch-cap' and analog_system == 'leap_frog':
    #     pytest.skip("Filter coefficients not good enough.")

    K = [calibration_method['kwargs']['K'](x) for x in range(N + 1)]
    analog_system_label = analog_system
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
            dynamic_type=dither_dynamic_type,
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
            dynamic_type=dither_dynamic_type,
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
            dynamic_type=dither_dynamic_type,
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

    if DEBUG:
        label_string = f"dither_dynamic_type={dither_dynamic_type},AS={analog_system_label},N={N},kappa={kappa_0_scale},simulator={simulation_method.__name__}"
        # PSDs
        plt.figure()
        plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
        plt.semilogx(
            f,
            10 * np.log10(np.abs(psd_ref)),
            label=f"Ref,est_ENOB={est_ENOB_ref:.1f} bits, est_SNR={est_SNR_ref:.1f} dB, BW={BW:.0e}",
        )
        plt.semilogx(
            f,
            10 * np.log10(np.abs(psd_val)),
            label=f"Val,est_ENOB={est_ENOB_val:.1f} bits, est_SNR={est_SNR_val:.1f} dB, BW={BW:.0e}",
        )
        plt.xlabel('Hz')
        plt.ylabel('V^2 / Hz dB')
        plt.legend()
        plt.savefig(f"psd_{N}_{ENOB}_{label_string}.png")

        # Training and Testing errors
        f_training_testing_error, ax_training_testing_error = plt.subplots(2, 2)
        training_error = np.array(training_errors)
        training_iterations = (
            np.arange(training_error.size) * calibration_method['kwargs']['batch_size']
        )
        testing_error = np.array([x["est_SNR"] for x in testing_errors])
        testing_iterations = np.array([x["iteration"] for x in testing_errors])

        ax_training_testing_error[0, 0].plot(training_iterations, training_error)
        ax_training_testing_error[0, 1].loglog(
            training_iterations, np.abs(training_error)
        )
        ax_training_testing_error[1, 0].plot(testing_iterations, testing_error)
        ax_training_testing_error[1, 1].semilogx(testing_iterations, testing_error)

        ax_training_testing_error[0, 0].set_title("Training Error")
        ax_training_testing_error[0, 1].set_title("Training Error")
        ax_training_testing_error[1, 0].set_title("Testing SNR")
        ax_training_testing_error[1, 1].set_title("Testing SNR")
        ax_training_testing_error[0, 0].set_ylabel("MSE")
        ax_training_testing_error[0, 1].set_ylabel("MSE")
        ax_training_testing_error[1, 1].set_ylabel("SNR [dB]")
        ax_training_testing_error[1, 0].set_ylabel("SNR [dB]")
        ax_training_testing_error[1, 0].set_xlabel("Iterations")
        ax_training_testing_error[1, 1].set_xlabel("Iterations")
        # ax_training_testing_error[0, 1].legend()
        # ax_training_testing_error[1, 0].legend()
        ax_training_testing_error[0, 0].legend()
        # ax_training_testing_error[1, 1].legend()

        ax_training_testing_error[0, 0].grid(True)
        ax_training_testing_error[0, 1].grid(True)
        ax_training_testing_error[1, 0].grid(True)
        ax_training_testing_error[1, 1].grid(True)

        f_training_testing_error.savefig(f"training_testing_error_{label_string}.png")

        # Filter Coefficients
        f_filter_coefficients, ax_filter_coefficients = plt.subplots(
            N + 1, 2, sharex=True, sharey=True
        )
        cal_fil = adaptive_filter.h
        for n in range(N + 1):
            coeffs = np.array(cal_fil[0][n])
            k_temp = np.arange(K[n]) - K[n] // 2
            ax_filter_coefficients[n, 0].plot(
                k_temp,
                coeffs,
                label=f"adaptive",
            )
            ax_filter_coefficients[n, 1].semilogy(
                k_temp,
                np.abs(coeffs),
                label=f"adaptive",
            )
            # Reference Filters
            ax_filter_coefficients[n, 0].plot(
                np.arange(K1 + K2) - K1,
                wiener_filter_reference.h[0, :, n],
                label="nominal",
            )
            ax_filter_coefficients[n, 1].semilogy(
                np.arange(K1 + K2) - K1,
                np.abs(wiener_filter_reference.h[0, :, n]),
                label="nominal",
            )
            ax_filter_coefficients[n, 0].legend()
            # ax_filter_coefficients[n, 1].legend()
            ax_filter_coefficients[n, 0].set_ylabel(f"$h_{{{n + 1}}}$")
            ax_filter_coefficients[n, 0].grid(True)
            ax_filter_coefficients[n, 1].grid(True)
            ax_filter_coefficients[-1, 0].set_xlabel("taps")
            ax_filter_coefficients[-1, 1].set_xlabel("taps")
            # ax_filter_coefficients[0, 0].set_title(f"Impulse response: dataset {res.args['dataset']['name']}")
            # ax_filter_coefficients[0, 1].set_title(f"Impulse response: dataset {res.args['dataset']['name']}")

        f_filter_coefficients.savefig(f"filter_coefficients_{label_string}.png")
        plt.close(f_filter_coefficients)
    assert est_ENOB_val >= ENOB
