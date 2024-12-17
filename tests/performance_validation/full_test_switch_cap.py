import pytest
import numpy as np
import cbadc
from cbadc.digital_estimator import (
    BatchEstimator,
)
from cbadc.digital_estimator._filter_coefficients import FilterComputationBackend
from cbadc.fom import snr_to_dB, snr_to_enob
from tests.performance_validation.fixtures import setup_filter
from cbadc.simulator import (
    PreComputedControlSignalsSimulator,
)
import matplotlib.pyplot as plt

DEBUG = False


@pytest.mark.parametrize(
    "N",
    [
        # pytest.param(2, id="N=2"),
        # pytest.param(4, id="N=4"),
        pytest.param(6, id="N=6"),
        # pytest.param(8, id="N=8"),
        # pytest.param(10, id="N=10"),
        pytest.param(12, id="N=12"),
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
        pytest.param(20, id="ENOB=20"),
        # pytest.param(23, id="ENOB=23"),
    ],
)
@pytest.mark.parametrize(
    "BW",
    [
        pytest.param(1e0, id="BW=1Hz"),
        # pytest.param(1e1, id="BW=10Hz"),
        # pytest.param(1e2, id="BW=100Hz"),
        # pytest.param(1e3, id="BW=1kHz"),
        # pytest.param(1e4, id="BW=10kHz"),
        # pytest.param(1e5, id="BW=100kHz"),
        # pytest.param(1e6, id="BW=1MHz"),
        # pytest.param(1e7, id="BW=10MHz"),
        pytest.param(1e8, id="BW=100MHz"),
        # pytest.param(1e9, id="BW=1GHz"),
    ],
)
@pytest.mark.parametrize(
    "analog_filter",
    [
        pytest.param("chain-of-integrators", id="chain_of_integrators_as"),
        pytest.param("leap_frog", id="leap_frog_as"),
    ],
)
@pytest.mark.parametrize(
    "digital_control",
    [
        # pytest.param('default', id="default_dc"),
        pytest.param("switch-cap", id="switch_cap_dc"),
    ],
)
@pytest.mark.parametrize(
    "simulation_method",
    [
        # pytest.param(AnalyticalSimulator, id="an_sim"),
        # pytest.param(FullSimulator, id="full_num_sim"),
        pytest.param(PreComputedControlSignalsSimulator, id="pre_num_sim"),
        # pytest.param(MPSimulator, id="mp_sim"),
    ],
)
@pytest.mark.parametrize(
    "reconstruction_method",
    [
        pytest.param(BatchEstimator, id="batch_de"),
        # pytest.param(ParallelEstimator, id="par-batch-de"),
        # pytest.param(IIRFilter, id="IIR_de"),
        # pytest.param(FIRFilter, id="FIR_de"),
    ],
)
@pytest.mark.parametrize(
    "filter_computation_method",
    [
        # pytest.param(FilterComputationBackend.numpy, id="numpy"),
        # pytest.param(FilterComputationBackend.sympy, id="sympy"),
        pytest.param(FilterComputationBackend.mpmath, id="mpmath"),
    ],
)
@pytest.mark.parametrize(
    "eta2",
    [
        # pytest.param(1.0, id="eta2=1"),
        pytest.param("snr", id="eta2=ENOB")
    ],
)
@pytest.mark.parametrize(
    "excess_delay",
    [pytest.param(0.0, id="excess_delay=0"), pytest.param(1e-1, id="excess_delay=0.1")],
)
def test_full_system(
    N,
    ENOB,
    BW,
    analog_filter,
    digital_control,
    simulation_method,
    reconstruction_method,
    filter_computation_method,
    eta2,
    excess_delay,
):
    if N >= 12 and analog_filter == "leap_frog" and BW >= 1e8:
        pytest.skip("Known limitation")

    # if N >= 12 and digital_control == 'switch-cap' and analog_filter == 'leap_frog':
    #     pytest.skip("Filter coefficients not good enough.")

    res = setup_filter(N, ENOB, BW, analog_filter, digital_control, excess_delay)
    K1 = 1 << 10
    K2 = K1
    if eta2 == "snr":
        eta2 = (
            np.linalg.norm(
                res["analog_filter"].transfer_function_matrix(
                    np.array([2 * np.pi * BW])
                )
            )
            ** 2
        )
    # Simulator
    de = reconstruction_method(
        res["analog_filter"],
        res["digital_control"],
        eta2,
        K1=K1,
        K2=K2,
        solver_type=filter_computation_method,
    )
    amplitude = 1e0
    frequency = 1.0 / res["digital_control"].clock.T
    while frequency > BW:
        frequency /= 2
    phase = 0.0
    offset = 0.0
    input = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)
    sim = simulation_method(
        res["analog_filter"],
        res["digital_control"],
        [input],
        initial_state_vector=np.random.randn(N) * 1e-1,
    )
    de(sim)
    # de.warm_up()
    size = 1 << 15
    u_hat = np.zeros(size)
    for index in range(size):
        u_hat[index] = next(de)
    u_hat_cut = u_hat[K1 + K2 :]
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_cut[:], fs=1 / res["digital_control"].clock.T, nperseg=u_hat_cut.size
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / res["digital_control"].clock.T
    )
    est_SNR = snr_to_dB(fom["snr"])
    est_ENOB = snr_to_enob(est_SNR)

    if DEBUG:
        plt.title(
            f"Power spectral density:\nN={N},as={analog_filter},dc={digital_control},ed={excess_delay:0.1e},ENOB={ENOB}"
        )
        plt.semilogx(
            f,
            10 * np.log10(np.abs(psd)),
            label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}",
        )
        plt.xlabel("Hz")
        plt.ylabel("V^2 / Hz dB")
        plt.legend()
        plt.show()
        print(res["analog_filter"].A)
        print(de)
    assert est_ENOB >= ENOB
