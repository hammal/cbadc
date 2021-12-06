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
        pytest.param(2, id="N=2"),
        # pytest.param(5, id="N=5"),
        # pytest.param(8, id="N=8"),
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
        pytest.param(1e3, id="BW=1kHz"),
        # pytest.param(1e7, id="BW=10MHz"),
    ],
)
@pytest.mark.parametrize(
    "analog_system",
    [
        pytest.param('chain-of-integrators', id="chain_of_integrators_as"),
        # pytest.param('leap_frog', id="leap_frog_as"),
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
        pytest.param(FullSimulator, id="full_num_sim"),
        pytest.param(PreComputedControlSignalsSimulator, id="pre_num_sim"),
        pytest.param(MPSimulator, id="mp_sim"),
    ],
)
def test_simulator(
    N,
    ENOB,
    BW,
    analog_system,
    digital_control,
    simulation_method,
):
    res = setup_filter(N, ENOB, BW, analog_system, digital_control)

    input = cbadc.analog_signal.ConstantSignal(0.1)
    initial_state = np.random.rand(N) * 2.0 - 1.0

    analytical_sim = AnalyticalSimulator(
        res['analog_system'],
        res['digital_control'],
        [input],
        initial_state_vector=initial_state,
    )
    sim = simulation_method(
        res['analog_system'],
        res['digital_control'],
        [input],
        initial_state_vector=initial_state,
    )
    count = 0

    for ref, est in zip(analytical_sim, sim):
        if np.all(ref == est):
            count += 1
        elif count > (1 << 14):
            break
        else:
            break

    assert count > 100
