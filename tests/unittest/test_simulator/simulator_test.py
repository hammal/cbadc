import cbadc
import numpy as np
from tests.fixture.chain_of_integrators import chain_of_integrators
import pytest
import cbadc.simulator.numerical_simulator as numerical_simulator

beta = 6250.0
rho = -62.5
N = 5
M = 5
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0] = beta
CT = np.zeros((1, N)).transpose()
CT[-1] = 1.0
Gamma_tildeT = np.eye(M)
Gamma = Gamma_tildeT * (-beta)
Ts = 1 / (2 * beta)


def test_initialization(chain_of_integrators):
    analogSignals = [cbadc.analog_signal.ConstantSignal(0.1)]
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    cbadc.simulator.get_simulator(
        chain_of_integrators["system"], digitalControl, analogSignals, clock
    )


def test_iterator(chain_of_integrators):
    analogSignals = [cbadc.analog_signal.ConstantSignal(0.1)]
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    statespacesimulator = cbadc.simulator.get_simulator(
        chain_of_integrators["system"], digitalControl, analogSignals, t_stop=Ts * 100
    )
    for control_signal in statespacesimulator:
        pass


def test_pre_and_non_pre_computations():
    N = 2
    M = N

    C_x = 1e-9
    R_beta = 16e4

    beta = 1 / (R_beta * C_x)
    T = 1 / (2 * beta)

    A = beta * np.eye(N, k=-1)
    B = np.zeros(N)
    B[0] = beta
    CT = np.eye(N)
    clock = cbadc.analog_signal.Clock(T)

    digital_control_sc = cbadc.digital_control.DigitalControl(clock, M)

    Gamma = -beta * np.eye(M)
    Gamma_tildeT = np.eye(M)

    analog_system_sc = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)

    print(digital_control_sc)
    print(analog_system_sc)

    amplitude = 0.1
    analog_signal = cbadc.analog_signal.Sinusoidal(amplitude, 1 / T / 32)

    size = 1 << 5

    rtol = 3e-14
    atol = 1e-100

    simulator_sc = cbadc.simulator.extended_simulation_result(
        cbadc.simulator.PreComputedControlSignalsSimulator(
            analog_system_sc,
            digital_control_sc,
            [analog_signal],
            atol=atol,
            rtol=rtol,
            initial_state_vector=0.5 * np.ones(N),
        )
    )

    digital_control_ref = cbadc.digital_control.DigitalControl(clock, M)
    simulator_ref = cbadc.simulator.extended_simulation_result(
        cbadc.simulator.FullSimulator(
            analog_system_sc,
            digital_control_ref,
            [analog_signal],
            clock,
            atol=atol,
            rtol=rtol,
            initial_state_vector=0.5 * np.ones(N),
        )
    )

    # Simulations
    for time_step in cbadc.utilities.show_status(range(size)):
        sim_state = next(simulator_sc)
        sim_state_ref = next(simulator_ref)
        print(
            time_step,
            sim_state["analog_state"],
            sim_state_ref["analog_state"],
            sim_state["analog_state"] - sim_state_ref["analog_state"],
            sim_state["control_signal"],
            sim_state_ref["control_signal"],
        )
        # np.testing.assert_allclose(
        #     sim_state["analog_state"], sim_state_ref["analog_state"], rtol=1e-1
        # )
        np.testing.assert_allclose(
            sim_state["control_signal"], sim_state_ref["control_signal"]
        )


def test_Ts_multiple_of_T(chain_of_integrators):
    analogSignals = [cbadc.analog_signal.ConstantSignal(0.1)]
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    sim_clock = cbadc.analog_signal.Clock(clock.T * 1e-2)
    cbadc.simulator.FullSimulator(
        chain_of_integrators["system"], digitalControl, analogSignals, sim_clock
    )


def test_noise_simulation_FullSimulator(chain_of_integrators):
    analogSignals = [cbadc.analog_signal.ConstantSignal(0.1)]
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    thermal_noise_covariance_matrix = 1e-5 * np.eye(N)
    noise_simulator = cbadc.simulator.FullSimulator(
        chain_of_integrators["system"],
        digitalControl,
        analogSignals,
        cov_x=thermal_noise_covariance_matrix,
    )

    size = 1 << 8
    for time_step in cbadc.utilities.show_status(range(size)):
        next(noise_simulator)


def test_noise_simulation_PreComputedSimulator(chain_of_integrators):
    analogSignals = [cbadc.analog_signal.ConstantSignal(0.1)]
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    thermal_noise_covariance_matrix = 1e-5 * np.eye(N)
    noise_simulator = cbadc.simulator.PreComputedControlSignalsSimulator(
        chain_of_integrators["system"],
        digitalControl,
        analogSignals,
        cov_x=thermal_noise_covariance_matrix,
    )

    size = 1 << 8
    for time_step in cbadc.utilities.show_status(range(size)):
        next(noise_simulator)
