import cbadc
import numpy as np
import pytest
from cbadc.analog_signal import impulse_responses
import cbadc.simulator.mp_simulator

beta = 6250.0
rho = -62.5
N = 4
M = 4
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0] = beta
CT = np.zeros((1, N))
CT[0, -1] = 1.0
Gamma_tildeT = np.eye(M)
Gamma = Gamma_tildeT * (-beta)
Ts = 1 / (2 * beta)
amplitude = 0.5


def test_initialization():
    analogSignals = [cbadc.analog_signal.ConstantSignal(0.1)]
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    sim = cbadc.simulator.mp_simulator.MPSimulator(
        analog_system, digitalControl, analogSignals, clock
    )
    print(sim)
    assert True


def test_iterator():
    analogSignals = [cbadc.analog_signal.ConstantSignal(0.1)]
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    sim = cbadc.simulator.mp_simulator.MPSimulator(
        analog_system, digitalControl, analogSignals, clock, t_stop=Ts * 100
    )
    for control_signal in sim:
        print(control_signal)


def test_LeapFrog_type_system():
    analogSignals = [cbadc.analog_signal.Sinusoidal(amplitude, 1 / Ts / 32)]
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    A = np.eye(N, k=1) * rho + np.eye(N, k=-1) * beta
    analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    sim = cbadc.simulator.mp_simulator.MPSimulator(
        analog_system, digitalControl, analogSignals, clock, t_stop=Ts * 100
    )
    for control_signal in sim:
        print(control_signal)


def test_switch_cap_control_system():
    analogSignals = [cbadc.analog_signal.Sinusoidal(amplitude, 1 / Ts / 32)]
    clock = cbadc.analog_signal.Clock(Ts)
    tau = 1 / beta * 1e-2
    t0 = 0.0
    impulse_response = cbadc.analog_signal.RCImpulseResponse(tau, t0)
    digitalControl = cbadc.digital_control.DigitalControl(
        clock, M, impulse_response=impulse_response
    )
    A = np.eye(N, k=-1) * beta
    analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    sim = cbadc.simulator.mp_simulator.MPSimulator(
        analog_system, digitalControl, analogSignals, clock, t_stop=Ts * 100
    )
    for control_signal in sim:
        print(control_signal)


def test_switch_cap_control_system_with_delay():
    analogSignals = [cbadc.analog_signal.Sinusoidal(amplitude, 1 / Ts / 32)]
    clock = cbadc.analog_signal.Clock(Ts)
    tau = 1 / beta * 1e-2
    t0 = Ts / 4 * 3
    impulse_response = cbadc.analog_signal.RCImpulseResponse(tau, t0)
    digitalControl = cbadc.digital_control.DigitalControl(
        clock, M, impulse_response=impulse_response
    )
    A = np.eye(N, k=-1) * beta
    analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    sim = cbadc.simulator.mp_simulator.MPSimulator(
        analog_system, digitalControl, analogSignals, clock, t_stop=Ts * 100
    )
    for control_signal in sim:
        print(control_signal)


# def test_analytical_vs_numerical_pre_computations():
#     N = 5
#     M = N

#     beta = 2 * np.pi * 1e6
#     T = 1 / (2 * beta)

#     A = beta * np.eye(N, k=-1)
#     B = np.zeros(N)
#     B[0] = beta
#     CT = np.eye(N)
#     clock = cbadc.analog_signal.Clock(T)

#     digital_control = cbadc.digital_control.DigitalControl(clock, M)

#     Gamma = -beta * np.eye(M)
#     Gamma_tildeT = np.eye(M)

#     analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)

#     amplitude = 0.1
#     analog_signal = cbadc.analog_signal.Sinusoidal(amplitude, 1 / T / 32)

#     Ts = T
#     size = 1 << 5

#     rtol = 3e-14
#     atol = 1e-100

#     num_sim = cbadc.simulator.PreComputedControlSignalsSimulator(
#         analog_system,
#         digital_control,
#         [analog_signal],
#         atol=atol,
#         rtol=rtol,
#         initial_state_vector=0.5 * np.ones(N),
#     )

#     num_sim_ext = cbadc.simulator.extended_simulation_result(num_sim)

#     digital_control_analytical = cbadc.digital_control.DigitalControl(clock, M)
#     an_sim = cbadc.simulator.mp_simulator.MPSimulator(
#         analog_system,
#         digital_control_analytical,
#         [analog_signal],
#         clock,
#         initial_state_vector=0.5 * np.ones(N),
#     )
#     an_sim_ext = cbadc.simulator.extended_simulation_result(an_sim)

#     # Simulations
#     for time_step in cbadc.utilities.show_status(range(size)):
#         sim_state = next(num_sim_ext)
#         sim_state_ref = next(an_sim_ext)
#         print(
#             time_step,
#             sim_state["analog_state"],
#             sim_state_ref["analog_state"],
#             sim_state["analog_state"] - sim_state_ref["analog_state"],
#             sim_state["control_signal"],
#             sim_state_ref["control_signal"],
#         )
