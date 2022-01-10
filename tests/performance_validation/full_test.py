import pytest
import numpy as np
import cbadc
from cbadc.digital_estimator import (
    DigitalEstimator,
    ParallelEstimator,
    IIRFilter,
    FIRFilter,
)
from cbadc.digital_estimator._filter_coefficients import FilterComputationBackend
from cbadc.fom import snr_from_dB, enob_to_snr, snr_to_dB, snr_to_enob
from .fixtures import setup_filter
from cbadc.simulator import (
    AnalyticalSimulator,
    FullSimulator,
    PreComputedControlSignalsSimulator,
    MPSimulator,
)


@pytest.mark.parametrize(
    "N",
    [
        # pytest.param(2, id="N=2"),
        pytest.param(6, id="N=6"),
        # pytest.param(12, id="N=12"),
    ],
)
@pytest.mark.parametrize(
    "ENOB",
    [
        pytest.param(12, id="ENOB=12"),
        # pytest.param(23, id="ENOB=23"),
    ],
)
@pytest.mark.parametrize(
    "BW",
    [
        # pytest.param(1e3, id="BW=1kHz"),
        pytest.param(1e7, id="BW=10MHz"),
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
    'simulation_method',
    [
        pytest.param(AnalyticalSimulator, id="an_sim"),
        pytest.param(FullSimulator, id="full_num_sim"),
        pytest.param(PreComputedControlSignalsSimulator, id="pre_num_sim"),
        pytest.param(MPSimulator, id="mp_sim"),
    ],
)
@pytest.mark.parametrize(
    'reconstruction_method',
    [
        pytest.param(DigitalEstimator, id="batch_de"),
        pytest.param(ParallelEstimator, id="par-batch-de"),
        pytest.param(IIRFilter, id="IIR_de"),
        pytest.param(FIRFilter, id="FIR_de"),
    ],
)
@pytest.mark.parametrize(
    'filter_computation_method',
    [
        # pytest.param(FilterComputationBackend.numpy, id="numpy"),
        # pytest.param(FilterComputationBackend.sympy, id="sympy"),
        pytest.param(FilterComputationBackend.mpmath, id="mpmath"),
    ],
)
@pytest.mark.parametrize(
    'eta2',
    [
        # pytest.param(1.0, id="eta2=1"),
        pytest.param('snr', id="eta2=ENOB")
    ],
)
@pytest.mark.parametrize(
    'excess_delay',
    [
        pytest.param(0.0, id="excess_delay=0"),
        # pytest.param(1e-1, id="excess_delay=0.1")
    ],
)
def test_full_system(
    N,
    ENOB,
    BW,
    analog_system,
    digital_control,
    simulation_method,
    reconstruction_method,
    filter_computation_method,
    eta2,
    excess_delay,
):
    if (
        N > 2
        and analog_system == 'leap_frog'
        and simulation_method == AnalyticalSimulator
    ):
        pytest.skip(
            "No analytical solution for AnalyticalSimulator, Leap-frog analog_system, N > 2"
        )

    res = setup_filter(N, ENOB, BW, analog_system, digital_control, excess_delay)
    K1 = 1 << 12
    K2 = 1 << 12
    if eta2 == 'snr':
        eta2 = snr_from_dB(enob_to_snr(ENOB))
    # Simulator
    de = reconstruction_method(
        res['analog_system'],
        res['digital_control'],
        eta2,
        K1,
        K2,
        solver_type=filter_computation_method,
    )
    amplitude = 1e0
    frequency = BW / 3
    phase = 0.0
    offset = 0.0
    input = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)
    sim = simulation_method(res['analog_system'], res['digital_control'], [input])
    de(sim)
    # de.warm_up()
    size = 1 << 15
    u_hat = np.zeros(size)
    for index in range(size):
        u_hat[index] = next(de)
    u_hat_cut = u_hat[K1 + K2 :]
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_cut[:], fs=1 / res['digital_control'].clock.T, nperseg=u_hat_cut.size
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 75)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-3)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / res['digital_control'].clock.T
    )
    est_ENOB = snr_to_enob(snr_to_dB(fom['snr']))
    assert est_ENOB >= ENOB
