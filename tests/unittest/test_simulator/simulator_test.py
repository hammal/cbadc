import cbadc
import numpy as np
from tests.fixture.chain_of_integrators import chain_of_integrators
import pytest
import cbadc.simulator.numerical_simulator

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


# def test_variable_sampling_rate():
#     N = 4
#     M = N

#     C_x = 1e-9
#     C_Gamma = C_x / 2
#     R_s = 100.0
#     R_beta = 16e4

#     beta = 1 / (R_beta * C_x)
#     T = 1 / (2 * beta)

#     A = beta * np.eye(N, k=-1)
#     B = np.zeros(N)
#     B[0] = beta
#     CT = np.eye(N)
#     impulse_response = cbadc.analog_signal.RCImpulseResponse(R_s * C_Gamma)
#     clock = cbadc.analog_signal.Clock(T)
#     digital_control_sc = cbadc.digital_control.DigitalControl(
#         clock, M, impulse_response=impulse_response
#     )

#     # Gamma = -beta * np.eye(M)
#     Gamma = -1 / (R_s * C_x) * np.eye(M)
#     Gamma_tildeT = np.eye(M)

#     analog_system_sc = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)

#     print(digital_control_sc)
#     print(analog_system_sc)

#     amplitude = 0.1
#     analog_signal = cbadc.analog_signal.Sinusoidal(amplitude, 1 / T / 32)

#     Downsampling = 1 << 3
#     Ts = T / Downsampling
#     size = 1 << 5
#     sim_clock = cbadc.analog_signal.Clock(Ts)

#     simulator_sc = cbadc.simulator.extended_simulation_result(
#         cbadc.simulator.FullSimulator(
#             analog_system_sc, digital_control_sc, [analog_signal], sim_clock
#         )
#     )

#     digital_control_ref = cbadc.digital_control.DigitalControl(
#         clock, M, impulse_response=impulse_response
#     )
#     simulator_ref = cbadc.simulator.extended_simulation_result(
#         cbadc.simulator.FullSimulator(
#             analog_system_sc, digital_control_ref, [analog_signal], clock,
#         )
#     )

#     # Simulations
#     sim_state = np.zeros(N)
#     for time_step in cbadc.utilities.show_status(range(size)):
#         for _ in range(Downsampling):
#             sim_state = next(simulator_sc)
#             print(
#                 sim_state["t"], sim_state["analog_state"],
#             )
#         sim_state_ref = next(simulator_ref)
#         print(
#             "\n",
#             time_step,
#             sim_state["t"],
#             sim_state_ref["t"],
#             sim_state["analog_state"],
#             sim_state_ref["analog_state"],
#             sim_state["analog_state"] - sim_state_ref["analog_state"],
#             sim_state["control_signal"],
#             sim_state_ref["control_signal"],
#             "\n\n",
#         )
#         # np.testing.assert_allclose(
#         #     sim_state["analog_state"], sim_state_ref["analog_state"], rtol=1e-0
#         # )
#         # np.testing.assert_allclose(
#         #     sim_state["control_signal"], sim_state_ref["control_signal"]
#         # )


# def test_simulator_verify_with_estimator():
#     N = 4
#     M = N

#     C_x = 1e-9
#     C_Gamma = C_x / 2
#     R_s = 100.0
#     R_beta = 16e4

#     beta = 1 / (R_beta * C_x)
#     T = 1 / (2 * beta)

#     A = beta * np.eye(N, k=-1)
#     B = np.zeros(N)
#     B[0] = beta
#     CT = np.eye(N)
#     impulse_response = cbadc.analog_signal.RCImpulseResponse(R_s * C_Gamma)
#     clock = cbadc.analog_signal.Clock(T)
#     digital_control_sc = cbadc.digital_control.DigitalControl(
#         clock, M, impulse_response=impulse_response
#     )

#     Gamma = -1 / (R_s * C_x) * np.eye(M)
#     Gamma_tildeT = np.eye(M)

#     analog_system_sc = cbadc.analog_system.AnalogSystem(
#         A, B, CT, Gamma, Gamma_tildeT)

#     print(digital_control_sc)
#     print(analog_system_sc)

#     amplitude = 1.0
#     analog_signal = cbadc.analog_signal.Sinusoidal(amplitude, 1 / T / 64)

#     size = 1 << 14

#     simulator_sc = cbadc.simulator.PreComputedControlSignalsSimulator(
#         analog_system_sc,
#         digital_control_sc,
#         [analog_signal],
#         clock,
#         initial_state_vector=0.5 * np.ones(N),
#         atol=1e-12,
#         rtol=1e-8
#     )

#     digital_control_ref = cbadc.digital_control.DigitalControl(
#         clock, M, impulse_response=impulse_response
#     )
#     simulator_ref = cbadc.simulator.FullSimulator(
#         analog_system_sc,
#         digital_control_ref,
#         [analog_signal],
#         clock,
#         atol=1e-30,
#         rtol=1e-11,
#         initial_state_vector=0.5 * np.ones(N),
#     )

#     # Digital Estimators
#     eta2 = 1e2
#     K1 = 1 << 8
#     K2 = K1
#     # prepending an anti-aliasing filter
#     omega_3dB = 2 * np.pi / T / 16
#     wp = omega_3dB / 2.0
#     ws = omega_3dB
#     gpass = 1.0
#     gstop = 60
#     filter = cbadc.analog_system.IIRDesign(wp, ws, gpass, gstop, ftype="ellip")
#     digital_estimator_sc = cbadc.digital_estimator.FIRFilter(
#         cbadc.analog_system.topology.chain([filter, analog_system_sc]),
#         digital_control_sc,
#         eta2,
#         K1,
#         K2,
#     )
#     digital_estimator_sc(simulator_sc)
#     digital_estimator_ref = cbadc.digital_estimator.FIRFilter(
#         cbadc.analog_system.topology.chain([filter, analog_system_sc]),
#         digital_control_ref,
#         eta2,
#         K1,
#         K2,
#     )
#     digital_estimator_ref(simulator_ref)
#     # Simulations
#     for _ in cbadc.utilities.show_status(range(size)):
#         u_hat = next(digital_estimator_sc)
#         u_hat_ref = next(digital_estimator_ref)
#         print(np.round(simulator_sc.t, 4), u_hat, u_hat_ref, u_hat - u_hat_ref)
#         # np.testing.assert_allclose(u_hat, u_hat_ref, rtol=1e-0)


# def test_Ts_not_multiple_of_T(chain_of_integrators):
#     analogSignals = [cbadc.analog_signal.ConstantSignal(0.1)]
#     digitalControl = cbadc.digital_control.DigitalControl(Ts, M)
#     with pytest.raises(Exception):
#         cbadc.simulator.StateSpaceSimulator(
#             chain_of_integrators["system"], digitalControl, analogSignals, Ts=Ts / np.pi
#         )


def test_Ts_multiple_of_T(chain_of_integrators):
    analogSignals = [cbadc.analog_signal.ConstantSignal(0.1)]
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    sim_clock = cbadc.analog_signal.Clock(clock.T * 1e-2)
    cbadc.simulator.FullSimulator(
        chain_of_integrators["system"], digitalControl, analogSignals, sim_clock
    )