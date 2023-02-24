import pytest
import numpy as np
import cbadc
from .fixtures import setup_filter
from cbadc.simulator import (
    AnalyticalSimulator,
    FullSimulator,
    PreComputedControlSignalsSimulator,
    MPSimulator,
)


@pytest.mark.skip(reason="to long simulation time")
@pytest.mark.parametrize(
    "N",
    [
        pytest.param(2, id="N=2"),
        # pytest.param(4, id="N=4"),
        pytest.param(8, id="N=8"),
        # pytest.param(10, id="N=10"),
        # pytest.param(12, id="N=12")
    ],
)
@pytest.mark.parametrize(
    "ENOB",
    [pytest.param(12, id="ENOB=12"), pytest.param(20, id="ENOB=20")],
)
@pytest.mark.parametrize(
    "BW",
    [pytest.param(1e3, id="BW=1kHz"), pytest.param(1e8, id="BW=100MHz")],
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
        # pytest.param(AnalyticalSimulator, id="ana_sim"),
    ],
)
@pytest.mark.parametrize(
    'reference_method',
    [
        # pytest.param(FullSimulator, id="full_num_sim"),
        # pytest.param(PreComputedControlSignalsSimulator, id="pre_num_sim"),
        # pytest.param(MPSimulator, id="mp_sim"),
        pytest.param(AnalyticalSimulator, id="ana_sim"),
    ],
)
@pytest.mark.parametrize(
    'excess_delay',
    [
        pytest.param(0.0, id="excess_delay=0"),
        # pytest.param(1e-1, id="excess_delay=0.1")
    ],
)
def test_simulator(
    N,
    ENOB,
    BW,
    analog_system,
    digital_control,
    simulation_method,
    reference_method,
    excess_delay,
):
    if simulation_method == reference_method:
        return
    number_of_coherent_cycles = 1 << 12
    res = setup_filter(
        N, ENOB, BW, analog_system, digital_control, excess_delay, local_feedback=True
    )
    res2 = setup_filter(
        N, ENOB, BW, analog_system, digital_control, excess_delay, local_feedback=True
    )

    # input = cbadc.analog_signal.Sinusoidal(1e-7, BW / 100)
    input = cbadc.analog_signal.ConstantSignal(0.0)
    # initial_state = np.random.rand(N) * 2.0 - 1.0
    initial_state = np.ones(N) * 0.1
    if N > 5 and analog_system == 'leap_frog':
        pytest.skip("No analytical solution for Leap-frog N > 2")

    analog_system = res['analog_system']

    ref_sim = reference_method(
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
    for ref, est in zip(ref_sim, sim):
        # print(ref_sim._state_vector, sim._state_vector)
        if count > (number_of_coherent_cycles):
            break
        elif np.all(ref == est):
            count += 1
        else:
            break

    assert count > number_of_coherent_cycles
