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
    [pytest.param(2, id="N=2"), pytest.param(5, id="N=5"), pytest.param(8, id="N=8")],
)
@pytest.mark.parametrize(
    "ENOB",
    [pytest.param(12, id="ENOB=12"), pytest.param(23, id="ENOB=23")],
)
@pytest.mark.parametrize(
    "BW",
    [pytest.param(1e3, id="BW=1kHz"), pytest.param(1e7, id="BW=10MHz")],
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
        pytest.param('switch_cap', id="switch_cap_dc"),
    ],
)
@pytest.mark.parametrize(
    'simulation_method',
    [
        pytest.param(FullSimulator, id="full_num_sim"),
        pytest.param(PreComputedControlSignalsSimulator, id="pre_num_sim"),
        pytest.param(MPSimulator, id="mp_sim"),
    ],
)
@pytest.mark.parametrize(
    'number_of_coherent_cycles',
    [
        # pytest.param(1 << 8, id="256_cycles"),
        # pytest.param(1 << 10, id="1024_cycles"),
        pytest.param(1 << 12, id="4096_cycles"),
    ],
)
def test_simulator(
    N,
    ENOB,
    BW,
    analog_system,
    digital_control,
    simulation_method,
    number_of_coherent_cycles,
):
    res = setup_filter(N, ENOB, BW, analog_system, digital_control)
    res2 = setup_filter(N, ENOB, BW, analog_system, digital_control)

    # input = cbadc.analog_signal.Sinusoidal(1e-7, BW / 100)
    input = cbadc.analog_signal.ConstantSignal(0.1)
    initial_state = np.random.rand(N) * 2.0 - 1.0

    if N > 2 and analog_system == 'leap_frog':
        pytest.skip("No analytical solution for Leap-frog N > 2")

    analytical_sim = AnalyticalSimulator(
        res['analog_system'],
        res['digital_control'],
        [input],
        initial_state_vector=initial_state[:],
    )
    sim = simulation_method(
        res2['analog_system'],
        res2['digital_control'],
        [input],
        initial_state_vector=initial_state[:],
    )
    count = 0
    for ref, est in zip(analytical_sim, sim):
        if count > (number_of_coherent_cycles):
            break
        elif np.all(ref == est):
            count += 1
        else:
            break

    assert count > number_of_coherent_cycles
