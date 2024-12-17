# from cbadc.circuit.state_space import StateSpaceFrontend
# from cbadc.circuit.components.sources import (
#     SinusoidalVoltageSource,
# )
# from cbadc.circuit import Terminal, SpiceDialect
# from cbadc.circuit.testbench import OpAmpTestBench, OTATestBench
# from cbadc.synthesis.leap_frog import get_leap_frog
# from cbadc.circuit.simulator import NGSpiceSimulator, SpectreSimulator
# from cbadc.digital_estimator import BatchEstimator
# from cbadc.utilities import (
#     compute_power_spectral_density,
#     find_sinusoidal,
#     snr_spectrum_computation_extended,
# )
# from cbadc.fom import snr_to_dB, snr_to_enob
# from cbadc.analog_signal import Sinusoidal
# from cbadc.digital_control import DitherControl, MultiPhaseDigitalControl
# from cbadc.analog_filter import AnalogSystem
# from cbadc.analog_frontend import AnalogFrontend, get_global_control
# from cbadc.simulator import get_simulator
# from cbadc.analog_signal import Clock
# from . import plot_state_dist
# import os
# import pytest
# import matplotlib.pyplot as plt
# import numpy as np
# import subprocess
# import scipy.linalg
# import scipy.integrate

# two_terminals = [Terminal() for _ in range(2)]
# three_terminals = two_terminals + [Terminal()]
# N = 4
# ENOB = 15
# BW = 1e6

# analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW, xi=1e-2)


# max_input_frequency = BW / 2.0
# estimate_simulation_length = 1 << 15
# K1 = 1 << 11
# K2 = K1
# show_state_plots = False


# # vdd = 3.7
# vdd = 1.0

# global_amplitude = vdd
# # amplitude = 0.0
# frequency = BW / 32
# frequency = BW
# delay = 1e-60
# offset = 0.0
# damping_factor = 0.0
# phase = 0.0

# figure_path = os.path.join(os.path.dirname(__file__), "global_control_figures")


# def test_ngspice_ref_testbench():
#     os.makedirs(figure_path, exist_ok=True)
#     number_of_simulated_samples = 1 << 12
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     # vdd = 1.2
#     # amplitude = vdd
#     DC_gain = 1e8
#     GBWP = DC_gain * BW

#     testbench = OpAmpTestBench(
#         analog_frontend,
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         clock=analog_frontend.digital_control.clock,
#         vdd_voltage=vdd,
#         GBWP=GBWP,
#         DC_gain=DC_gain,
#     )

#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T,
#         number_of_simulated_samples * analog_frontend.digital_control.clock.T,
#         netlist_filename='opamp_ref_phase_test.cir',
#         raw_output_filename='opamp_ref_phase_test.raw',
#     )
#     ngspice_simulator.make_netlist()
#     ngspice_simulator.run()
#     ngspice_simulator.parse()

#     controls = np.zeros((number_of_simulated_samples, N))

#     for i, s in enumerate(ngspice_simulator):
#         controls[i, :] = s

#     figure_name = os.path.join(figure_path, 'ngspice_state_ref.png')

#     headers, data = ngspice_simulator.get_input_signals()

#     for i in range(1, analog_frontend.analog_filter.L + 1):
#         plt.plot(data[:, 0], data[:, i], label=headers[i])

#     headers, data = ngspice_simulator.get_state_trajectories()
#     for i in range(1, analog_frontend.analog_filter.N + 1):
#         plt.plot(
#             data[:, 0],
#             data[:, i] - data[:, i + analog_frontend.analog_filter.N],
#             label=f"{headers[i]}-{headers[i + analog_frontend.analog_filter.N]}",
#         )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i],
#         #     label=f"{headers[i]}",
#         # )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i + analog_frontend.analog_filter.N],
#         #     label=f"{headers[i + analog_frontend.analog_filter.N]}",
#         # )

#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name)

#     plt.figure()
#     figure_name_2 = os.path.join(figure_path, 'ngspice_control_ref.png')

#     for i in range(N):
#         plt.plot(
#             np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
#             controls[:, i],
#             label="$s_{" + str(i) + "}$",
#         )
#     plt.plot(data[:, 0], data[:, -1], label="$u$")
#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name_2)
#     diff_states = (
#         data[:, 1 : 1 + analog_frontend.analog_filter.N]
#         - data[
#             :,
#             1
#             + analog_frontend.analog_filter.N : 1
#             + 2 * analog_frontend.analog_filter.N,
#         ]
#     )
#     plot_state_dist(
#         diff_states.transpose(), os.path.join(figure_path, 'ngspice_state_dist_ref.png')
#     )
#     # ngspice_simulator.cleanup()
#     assert True


# # def test_ngspice_simulator_multi_phase_digital_control_with_opamp_testbench():
# #     os.makedirs(figure_path, exist_ok=True)
# #     number_of_simulated_samples = 1 << 10
# #     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
# #     # vdd = 1.2
# #     # amplitude = vdd
# #     DC_gain = 1e8
# #     GBWP = DC_gain * BW

# #     # Setup multi-phase
# #     phi_delay = np.linspace(
# #         0, analog_frontend.digital_control.clock.T, N, endpoint=False
# #     )

# #     # phi_delay = 0 * phi_delay

# #     digital_control = MultiPhaseDigitalControl(
# #         analog_frontend.digital_control.clock, phi_delay
# #     )
# #     # Create a global control object
# #     global_control_frontend = get_global_control(
# #         analog_frontend.analog_filter, digital_control, phi_delay
# #     )

# #     testbench = OpAmpTestBench(
# #         global_control_frontend,
# #         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
# #         clock=analog_frontend.digital_control.clock,
# #         vdd_voltage=vdd,
# #         GBWP=GBWP,
# #         DC_gain=DC_gain,
# #     )

# #     ngspice_simulator = NGSpiceSimulator(
# #         testbench,
# #         analog_frontend.digital_control.clock.T / N,
# #         number_of_simulated_samples * analog_frontend.digital_control.clock.T,
# #         netlist_filename='opamp_m_phase_test.cir',
# #         raw_output_filename='opamp_m_phase_test.raw',
# #     )
# #     ngspice_simulator.make_netlist()
# #     ngspice_simulator.run()
# #     ngspice_simulator.parse()

# #     controls = np.zeros((number_of_simulated_samples, N))

# #     for i, s in enumerate(ngspice_simulator):
# #         controls[i, :] = s

# #     figure_name = os.path.join(figure_path, 'ngspice_state_m_phase_trajectories_1.png')

# #     headers, data = ngspice_simulator.get_input_signals()
# #     for i in range(1, analog_frontend.analog_filter.L + 1):
# #         plt.plot(data[:, 0], data[:, i], label=headers[i])

# #     headers, data = ngspice_simulator.get_state_trajectories()
# #     for i in range(1, analog_frontend.analog_filter.N + 1):
# #         plt.plot(
# #             data[:, 0],
# #             data[:, i] - data[:, i + analog_frontend.analog_filter.N],
# #             label=f"{headers[i]}-{headers[i + analog_frontend.analog_filter.N]}",
# #         )
# #         # plt.plot(
# #         #     data[:, 0],
# #         #     data[:, i],
# #         #     label=f"{headers[i]}",
# #         # )
# #         # plt.plot(
# #         #     data[:, 0],
# #         #     data[:, i + analog_frontend.analog_filter.N],
# #         #     label=f"{headers[i + analog_frontend.analog_filter.N]}",
# #         # )

# #     plt.xlabel('time [s]')
# #     plt.ylabel('amplitude')
# #     plt.legend()
# #     plt.savefig(figure_name)

# #     plt.figure()
# #     figure_name_2 = os.path.join(figure_path, 'ngspice_control_m_phase_signals_1.png')

# #     for i in range(N):
# #         plt.plot(
# #             np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
# #             controls[:, i],
# #             label="$s_{" + str(i) + "}$",
# #         )
# #     plt.plot(data[:, 0], data[:, -1], label="$u$")
# #     plt.xlabel('time [s]')
# #     plt.ylabel('amplitude')
# #     plt.legend()
# #     plt.savefig(figure_name_2)

# #     ngspice_simulator.cleanup()
# #     assert True


# def test_ngspice_simulator_no_p_multi_phase_digital_control_with_opamp_testbench():
#     os.makedirs(figure_path, exist_ok=True)
#     number_of_simulated_samples = 1 << 12
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     # vdd = 1.0
#     # global_amplitude = vdd * 1e-3
#     # amplitude = vdd
#     DC_gain = 1e8
#     GBWP = DC_gain * BW

#     T = analog_frontend.digital_control.clock.T
#     phi_delay = np.zeros(N)

#     analog_filter = analog_frontend.analog_filter
#     Gamma = analog_frontend.analog_filter.Gamma

#     digital_control = MultiPhaseDigitalControl(Clock(T), phi_delay)

#     # Create a global control object
#     global_control_frontend = get_global_control(
#         AnalogSystem(
#             analog_filter.A,
#             analog_filter.B,
#             analog_filter.CT,
#             Gamma,
#             -Gamma.transpose(),
#         ),
#         digital_control,
#         phi_delay,
#     )

#     print(global_control_frontend.analog_filter)
#     print(global_control_frontend.digital_control)

#     testbench = OpAmpTestBench(
#         global_control_frontend,
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         clock=analog_frontend.digital_control.clock,
#         vdd_voltage=vdd,
#         GBWP=GBWP,
#         DC_gain=DC_gain,
#     )

#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T,
#         number_of_simulated_samples * analog_frontend.digital_control.clock.T,
#         netlist_filename=f'no_p_opamp_m={N}_phase_test.cir',
#         raw_output_filename=f'no_p_opamp_m={N}_phase_test.raw',
#     )
#     ngspice_simulator.make_netlist()
#     ngspice_simulator.run()
#     ngspice_simulator.parse()

#     controls = np.zeros((number_of_simulated_samples, N))

#     for i, s in enumerate(ngspice_simulator):
#         controls[i, :] = s

#     figure_name = os.path.join(
#         figure_path, f'ngspice_no_p_state_m={N}_phase_trajectories_1.png'
#     )

#     headers, data = ngspice_simulator.get_input_signals()
#     for i in range(1, analog_frontend.analog_filter.L + 1):
#         plt.plot(data[:, 0] / T, data[:, i], label=headers[i])

#     headers, data = ngspice_simulator.get_state_trajectories()

#     diff_states = (
#         data[:, 1 : 1 + analog_filter.N]
#         - data[:, 1 + analog_filter.N : 1 + 2 * analog_filter.N]
#     )
#     print(diff_states.shape)
#     for i in range(analog_frontend.analog_filter.N):
#         plt.plot(
#             data[:, 0] / T,
#             diff_states[:, i],
#             label=f"{headers[i + 1]}-{headers[i + 1 + analog_frontend.analog_filter.N]}",
#         )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i],
#         #     label=f"{headers[i]}",
#         # )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i + analog_frontend.analog_filter.N],
#         #     label=f"{headers[i + analog_frontend.analog_filter.N]}",
#         # )
#     plt.ylim((-1, 1))
#     plt.xlabel('t/T')
#     plt.ylabel('amplitude')
#     plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
#     plt.legend()
#     plt.savefig(figure_name)

#     plt.figure()
#     figure_name_2 = os.path.join(
#         figure_path, f'ngspice_no_p_control_m={N}_phase_signals_1.png'
#     )

#     for i in range(N):
#         plt.plot(
#             np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
#             controls[:, i],
#             label="$s_{" + str(i) + "}$",
#         )
#     plt.plot(data[:, 0], data[:, -1], label="$u$")
#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name_2)

#     plot_state_dist(
#         diff_states.transpose(),
#         os.path.join(figure_path, f'ngspice_no_p_state_dist.png'),
#     )
#     ngspice_simulator.cleanup()
#     assert True


# @pytest.mark.parametrize(
#     "multiplier",
#     [
#         pytest.param(1),
#         pytest.param(2),
#         pytest.param(3),
#         pytest.param(4),
#         pytest.param(6),
#         pytest.param(8),
#     ],
# )
# def test_ngspice_simulator_higher_control_with_opamp_testbench(
#     multiplier,
# ):
#     os.makedirs(figure_path, exist_ok=True)

#     number_of_simulated_samples = 1 << 12
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     # vdd = 1.0
#     # global_amplitude = vdd * 1e-3
#     # amplitude = vdd
#     DC_gain = 1e8
#     GBWP = DC_gain * BW

#     T = analog_frontend.digital_control.clock.T
#     phi_delay = np.linspace(0, T, multiplier * N, endpoint=False)

#     analog_filter = analog_frontend.analog_filter
#     Gamma = np.hstack(
#         [analog_frontend.analog_filter.Gamma for _ in range(multiplier)]
#     ) / float(multiplier)

#     digital_control = MultiPhaseDigitalControl(Clock(T), phi_delay)

#     A = analog_filter.A
#     for n in range(N - 1):
#         A[n + 1, n] *= multiplier
#         A[n, n + 1] /= multiplier
#     # Create a global control object
#     global_control_frontend = get_global_control(
#         AnalogSystem(
#             A,
#             multiplier * analog_filter.B,
#             analog_filter.CT,
#             Gamma,
#             -Gamma.transpose(),
#         ),
#         digital_control,
#         phi_delay,
#     )

#     print(global_control_frontend.analog_filter)
#     print(global_control_frontend.digital_control)

#     testbench = OpAmpTestBench(
#         global_control_frontend,
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         clock=analog_frontend.digital_control.clock,
#         vdd_voltage=vdd,
#         GBWP=GBWP,
#         DC_gain=DC_gain,
#     )

#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T / (N * multiplier),
#         number_of_simulated_samples * analog_frontend.digital_control.clock.T,
#         netlist_filename=f'opamp_High_m={N * multiplier}_phase_test.cir',
#         raw_output_filename=f'opamp_High_m={N * multiplier}_phase_test.raw',
#     )
#     ngspice_simulator.make_netlist()
#     ngspice_simulator.run()
#     ngspice_simulator.parse()

#     controls = np.zeros((number_of_simulated_samples, N * multiplier))

#     for i, s in enumerate(ngspice_simulator):
#         controls[i, :] = s

#     figure_name = os.path.join(
#         figure_path, f'ngspice_state_High_m={N * multiplier}_phase_trajectories_1.png'
#     )

#     headers, data = ngspice_simulator.get_input_signals()
#     for i in range(1, analog_frontend.analog_filter.L + 1):
#         plt.plot(data[:, 0] / T, data[:, i], label=headers[i])

#     headers, data = ngspice_simulator.get_state_trajectories()

#     diff_states = (
#         data[:, 1 : 1 + analog_filter.N]
#         - data[:, 1 + analog_filter.N : 1 + 2 * analog_filter.N]
#     )
#     print(diff_states.shape)
#     for i in range(analog_frontend.analog_filter.N):
#         plt.plot(
#             data[:, 0] / T,
#             diff_states[:, i],
#             label=f"{headers[i + 1]}-{headers[i + 1 + analog_frontend.analog_filter.N]}",
#         )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i],
#         #     label=f"{headers[i]}",
#         # )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i + analog_frontend.analog_filter.N],
#         #     label=f"{headers[i + analog_frontend.analog_filter.N]}",
#         # )
#     plt.ylim((-1, 1))
#     plt.xlabel('t/T')
#     plt.ylabel('amplitude')
#     plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
#     plt.legend()
#     plt.savefig(figure_name)

#     plt.figure()
#     figure_name_2 = os.path.join(
#         figure_path, f'ngspice_control_High_m={N * multiplier}_phase_signals_1.png'
#     )

#     for i in range(N):
#         plt.plot(
#             np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
#             controls[:, i],
#             label="$s_{" + str(i) + "}$",
#         )
#     plt.plot(data[:, 0], data[:, -1], label="$u$")
#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name_2)

#     plot_state_dist(
#         diff_states.transpose(),
#         os.path.join(figure_path, f'ngspice_High_mul_{multiplier}_state_dist.png'),
#     )
#     ngspice_simulator.cleanup()
#     assert True


# @pytest.mark.parametrize(
#     "multiplier",
#     [
#         pytest.param(1),
#         pytest.param(2),
#         pytest.param(3),
#         pytest.param(4),
#         pytest.param(6),
#         pytest.param(8),
#     ],
# )
# def test_ngspice_GAIN(
#     multiplier,
# ):
#     os.makedirs(figure_path, exist_ok=True)

#     number_of_simulated_samples = 1 << 12
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     # vdd = 1.0
#     # global_amplitude = vdd * 1e-3
#     # amplitude = vdd
#     DC_gain = 1e8
#     GBWP = DC_gain * BW

#     T = analog_frontend.digital_control.clock.T * multiplier
#     phi_delay = np.linspace(0, T, multiplier * N, endpoint=False)

#     analog_filter = analog_frontend.analog_filter

#     Gamma = np.hstack(
#         [analog_frontend.analog_filter.Gamma for _ in range(multiplier)]
#     ) / float(multiplier)

#     digital_control = MultiPhaseDigitalControl(Clock(T), phi_delay)

#     global_control_frontend = get_global_control(
#         AnalogSystem(
#             analog_filter.A,
#             analog_filter.B,
#             analog_filter.CT,
#             Gamma,
#             -Gamma.transpose(),
#         ),
#         digital_control,
#         phi_delay,
#     )

#     print(global_control_frontend.analog_filter)
#     print(global_control_frontend.digital_control)

#     testbench = OpAmpTestBench(
#         global_control_frontend,
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         clock=analog_frontend.digital_control.clock,
#         vdd_voltage=vdd,
#         GBWP=GBWP,
#         DC_gain=DC_gain,
#     )

#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T / (N * multiplier),
#         number_of_simulated_samples * analog_frontend.digital_control.clock.T,
#         netlist_filename=f'opamp_OSR={multiplier}_phase_test.cir',
#         raw_output_filename=f'opamp_OSR={multiplier}_phase_test.raw',
#     )
#     ngspice_simulator.make_netlist()
#     ngspice_simulator.run()
#     ngspice_simulator.parse()

#     controls = np.zeros((number_of_simulated_samples, N * multiplier))

#     for i, s in enumerate(ngspice_simulator):
#         controls[i, :] = s

#     figure_name = os.path.join(
#         figure_path, f'ngspice_state_OSR={multiplier}_phase_trajectories_1.png'
#     )

#     headers, data = ngspice_simulator.get_input_signals()
#     for i in range(1, analog_frontend.analog_filter.L + 1):
#         plt.plot(data[:, 0] / T, data[:, i], label=headers[i])

#     headers, data = ngspice_simulator.get_state_trajectories()

#     diff_states = (
#         data[:, 1 : 1 + analog_frontend.analog_filter.N]
#         - data[
#             :,
#             1
#             + analog_frontend.analog_filter.N : 1
#             + 2 * analog_frontend.analog_filter.N,
#         ]
#     )
#     print(diff_states.shape)
#     for i in range(analog_frontend.analog_filter.N):
#         plt.plot(
#             data[:, 0] / T,
#             diff_states[:, i],
#             label=f"{headers[i + 1]}-{headers[i + 1 + analog_frontend.analog_filter.N]}",
#         )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i],
#         #     label=f"{headers[i]}",
#         # )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i + analog_frontend.analog_filter.N],
#         #     label=f"{headers[i + analog_frontend.analog_filter.N]}",
#         # )
#     # plt.ylim((-1, 1))
#     plt.xlabel('t/T')
#     plt.ylabel('amplitude')
#     plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
#     plt.legend()
#     plt.savefig(figure_name)

#     plt.figure()
#     figure_name_2 = os.path.join(
#         figure_path, f'ngspice_control_OSR={multiplier}_phase_signals_1.png'
#     )

#     for i in range(N):
#         plt.plot(
#             np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
#             controls[:, i],
#             label="$s_{" + str(i) + "}$",
#         )
#     plt.plot(data[:, 0], data[:, -1], label="$u$")
#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name_2)

#     plot_state_dist(
#         diff_states.transpose(),
#         os.path.join(figure_path, f'ngspice_OSR_{multiplier}_state_dist.png'),
#     )
#     ngspice_simulator.cleanup()
#     assert True


# @pytest.mark.parametrize(
#     "multiplier",
#     [
#         pytest.param(1),
#         pytest.param(2),
#         pytest.param(3),
#         pytest.param(4),
#         pytest.param(6),
#         pytest.param(8),
#     ],
# )
# def test_ngspice_OSR(
#     multiplier,
# ):
#     os.makedirs(figure_path, exist_ok=True)

#     number_of_simulated_samples = 1 << 12
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     # vdd = 1.0
#     # global_amplitude = vdd * 1e-3
#     # amplitude = vdd
#     DC_gain = 1e8
#     GBWP = DC_gain * BW

#     T = analog_frontend.digital_control.clock.T * multiplier
#     phi_delay = np.linspace(0, T, multiplier * N, endpoint=False)

#     analog_filter = analog_frontend.analog_filter

#     Gamma = np.hstack(
#         [analog_frontend.analog_filter.Gamma for _ in range(multiplier)]
#     ) / float(multiplier)

#     digital_control = MultiPhaseDigitalControl(Clock(T), phi_delay)

#     global_control_frontend = get_global_control(
#         AnalogSystem(
#             analog_filter.A,
#             analog_filter.B,
#             analog_filter.CT,
#             Gamma,
#             -Gamma.transpose(),
#         ),
#         digital_control,
#         phi_delay,
#     )

#     print(global_control_frontend.analog_filter)
#     print(global_control_frontend.digital_control)

#     testbench = OpAmpTestBench(
#         global_control_frontend,
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         clock=analog_frontend.digital_control.clock,
#         vdd_voltage=vdd,
#         GBWP=GBWP,
#         DC_gain=DC_gain,
#     )

#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T / (N * multiplier),
#         number_of_simulated_samples * analog_frontend.digital_control.clock.T,
#         netlist_filename=f'opamp_OSR={multiplier}_phase_test.cir',
#         raw_output_filename=f'opamp_OSR={multiplier}_phase_test.raw',
#     )
#     ngspice_simulator.make_netlist()
#     ngspice_simulator.run()
#     ngspice_simulator.parse()

#     controls = np.zeros((number_of_simulated_samples, N * multiplier))

#     for i, s in enumerate(ngspice_simulator):
#         controls[i, :] = s

#     figure_name = os.path.join(
#         figure_path, f'ngspice_state_OSR={multiplier}_phase_trajectories_1.png'
#     )

#     headers, data = ngspice_simulator.get_input_signals()
#     for i in range(1, analog_frontend.analog_filter.L + 1):
#         plt.plot(data[:, 0] / T, data[:, i], label=headers[i])

#     headers, data = ngspice_simulator.get_state_trajectories()

#     diff_states = (
#         data[:, 1 : 1 + analog_frontend.analog_filter.N]
#         - data[
#             :,
#             1
#             + analog_frontend.analog_filter.N : 1
#             + 2 * analog_frontend.analog_filter.N,
#         ]
#     )
#     print(diff_states.shape)
#     for i in range(analog_frontend.analog_filter.N):
#         plt.plot(
#             data[:, 0] / T,
#             diff_states[:, i],
#             label=f"{headers[i + 1]}-{headers[i + 1 + analog_frontend.analog_filter.N]}",
#         )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i],
#         #     label=f"{headers[i]}",
#         # )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i + analog_frontend.analog_filter.N],
#         #     label=f"{headers[i + analog_frontend.analog_filter.N]}",
#         # )
#     # plt.ylim((-1, 1))
#     plt.xlabel('t/T')
#     plt.ylabel('amplitude')
#     plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
#     plt.legend()
#     plt.savefig(figure_name)

#     plt.figure()
#     figure_name_2 = os.path.join(
#         figure_path, f'ngspice_control_OSR={multiplier}_phase_signals_1.png'
#     )

#     for i in range(N):
#         plt.plot(
#             np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
#             controls[:, i],
#             label="$s_{" + str(i) + "}$",
#         )
#     plt.plot(data[:, 0], data[:, -1], label="$u$")
#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name_2)

#     plot_state_dist(
#         diff_states.transpose(),
#         os.path.join(figure_path, f'ngspice_OSR_{multiplier}_state_dist.png'),
#     )
#     ngspice_simulator.cleanup()
#     assert True


# def test_ngspice_try_to_replicate_python():
#     os.makedirs(figure_path, exist_ok=True)

#     number_of_simulated_samples = 1 << 14
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     # vdd = 1.0
#     # global_amplitude = vdd * 1e-3
#     # amplitude = vdd
#     DC_gain = 1e8
#     GBWP = DC_gain * BW

#     T = analog_frontend.digital_control.clock.T
#     phi_delay = np.linspace(0, T, N, endpoint=False)

#     analog_filter = analog_frontend.analog_filter
#     Gamma = analog_frontend.analog_filter.Gamma

#     for n in range(N):
#         Gamma[n, n] *= 0.7**n

#     digital_control = MultiPhaseDigitalControl(Clock(T), phi_delay)

#     # Create a global control object
#     global_control_frontend = get_global_control(
#         AnalogSystem(
#             analog_filter.A,
#             analog_filter.B,
#             analog_filter.CT,
#             Gamma,
#             -Gamma.transpose(),
#         ),
#         digital_control,
#         phi_delay,
#     )

#     print(global_control_frontend.analog_filter)
#     print(global_control_frontend.digital_control)

#     testbench = OpAmpTestBench(
#         global_control_frontend,
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         clock=analog_frontend.digital_control.clock,
#         vdd_voltage=vdd,
#         GBWP=GBWP,
#         DC_gain=DC_gain,
#     )

#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T / (N),
#         number_of_simulated_samples * analog_frontend.digital_control.clock.T,
#         netlist_filename='opamp_damp_phase_test.cir',
#         raw_output_filename='opamp_damp_phase_test.raw',
#     )
#     ngspice_simulator.make_netlist()
#     ngspice_simulator.run()
#     ngspice_simulator.parse()

#     controls = np.zeros((number_of_simulated_samples, N))

#     for i, s in enumerate(ngspice_simulator):
#         controls[i, :] = s

#     figure_name = os.path.join(
#         figure_path, 'ngspice_state_damped_gamma_phase_trajectories.png'
#     )

#     headers, data = ngspice_simulator.get_input_signals()
#     for i in range(1, analog_frontend.analog_filter.L + 1):
#         plt.plot(data[:, 0] / T, data[:, i], label=headers[i])

#     headers, data = ngspice_simulator.get_state_trajectories()

#     diff_states = (
#         data[:, 1 : 1 + analog_filter.N]
#         - data[:, 1 + analog_filter.N : 1 + 2 * analog_filter.N]
#     )
#     print(diff_states.shape)
#     for i in range(analog_frontend.analog_filter.N):
#         plt.plot(
#             data[:, 0] / T,
#             diff_states[:, i],
#             label=f"{headers[i + 1]}-{headers[i + 1 + analog_frontend.analog_filter.N]}",
#         )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i],
#         #     label=f"{headers[i]}",
#         # )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i + analog_frontend.analog_filter.N],
#         #     label=f"{headers[i + analog_frontend.analog_filter.N]}",
#         # )
#     plt.ylim((-1, 1))
#     plt.xlabel('t/T')
#     plt.ylabel('amplitude')
#     plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
#     plt.legend()
#     plt.savefig(figure_name)

#     plt.figure()
#     figure_name_2 = os.path.join(
#         figure_path, f'ngspice_control_damped_gamma_phase_signals.png'
#     )

#     for i in range(N):
#         plt.plot(
#             np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
#             controls[:, i],
#             label="$s_{" + str(i) + "}$",
#         )
#     plt.plot(data[:, 0], data[:, -1], label="$u$")
#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name_2)

#     plot_state_dist(
#         diff_states.transpose(),
#         os.path.join(figure_path, f'ngspice_mul_damped_gamma_state_dist.png'),
#     )
#     ngspice_simulator.cleanup()
#     assert True


# def test_ngspice_try_to_replicate_python_increased_gain():
#     os.makedirs(figure_path, exist_ok=True)

#     number_of_simulated_samples = 1 << 14
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     # vdd = 1.0
#     # global_amplitude = vdd * 1e-3
#     # amplitude = vdd
#     mul = 1 / 0.7
#     DC_gain = 1e8
#     GBWP = DC_gain * BW

#     T = analog_frontend.digital_control.clock.T
#     phi_delay = np.linspace(0, T, N, endpoint=False)

#     analog_filter = analog_frontend.analog_filter
#     Gamma = analog_frontend.analog_filter.Gamma

#     B = analog_frontend.analog_filter.B
#     A = analog_frontend.analog_filter.A
#     for n in range(N - 1):
#         A[n + 1, n] *= mul
#         A[n, n + 1] /= mul

#     digital_control = MultiPhaseDigitalControl(Clock(T), phi_delay)

#     # Create a global control object
#     global_control_frontend = get_global_control(
#         AnalogSystem(
#             A,
#             B,
#             analog_filter.CT,
#             Gamma,
#             -Gamma.transpose(),
#         ),
#         digital_control,
#         phi_delay,
#     )

#     print(global_control_frontend.analog_filter)
#     print(global_control_frontend.digital_control)

#     testbench = OpAmpTestBench(
#         global_control_frontend,
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         clock=analog_frontend.digital_control.clock,
#         vdd_voltage=vdd,
#         GBWP=GBWP,
#         DC_gain=DC_gain,
#     )

#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T / (N),
#         number_of_simulated_samples * analog_frontend.digital_control.clock.T,
#         netlist_filename='opamp_gain_phase_test.cir',
#         raw_output_filename='opamp_gain_phase_test.raw',
#     )
#     ngspice_simulator.make_netlist()
#     ngspice_simulator.run()
#     ngspice_simulator.parse()

#     controls = np.zeros((number_of_simulated_samples, N))

#     for i, s in enumerate(ngspice_simulator):
#         controls[i, :] = s

#     figure_name = os.path.join(
#         figure_path, 'ngspice_state_gain_gamma_phase_trajectories.png'
#     )

#     headers, data = ngspice_simulator.get_input_signals()
#     for i in range(1, analog_frontend.analog_filter.L + 1):
#         plt.plot(data[:, 0] / T, data[:, i], label=headers[i])

#     headers, data = ngspice_simulator.get_state_trajectories()

#     diff_states = (
#         data[:, 1 : 1 + analog_filter.N]
#         - data[:, 1 + analog_filter.N : 1 + 2 * analog_filter.N]
#     )
#     print(diff_states.shape)
#     for i in range(analog_frontend.analog_filter.N):
#         plt.plot(
#             data[:, 0] / T,
#             diff_states[:, i],
#             label=f"{headers[i + 1]}-{headers[i + 1 + analog_frontend.analog_filter.N]}",
#         )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i],
#         #     label=f"{headers[i]}",
#         # )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i + analog_frontend.analog_filter.N],
#         #     label=f"{headers[i + analog_frontend.analog_filter.N]}",
#         # )
#     plt.ylim((-1, 1))
#     plt.xlabel('t/T')
#     plt.ylabel('amplitude')
#     plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
#     plt.legend()
#     plt.savefig(figure_name)

#     plt.figure()
#     figure_name_2 = os.path.join(
#         figure_path, f'ngspice_control_gain_gamma_phase_signals.png'
#     )

#     for i in range(N):
#         plt.plot(
#             np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
#             controls[:, i],
#             label="$s_{" + str(i) + "}$",
#         )
#     plt.plot(data[:, 0], data[:, -1], label="$u$")
#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name_2)

#     plot_state_dist(
#         diff_states.transpose(),
#         os.path.join(figure_path, f'ngspice_mul_gain_gamma_state_dist.png'),
#     )
#     ngspice_simulator.cleanup()
#     assert True


# def test_ngspice_try_Hadamard():
#     os.makedirs(figure_path, exist_ok=True)

#     number_of_simulated_samples = 1 << 12
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     # vdd = 1.0
#     global_amplitude = vdd / np.sqrt(N)
#     # amplitude = vdd
#     DC_gain = 1e8
#     GBWP = DC_gain * BW

#     T = analog_frontend.digital_control.clock.T
#     phi_delay = np.linspace(0, T, N, endpoint=False)

#     analog_filter = analog_frontend.analog_filter
#     Gamma = np.dot(scipy.linalg.hadamard(N) / np.sqrt(N), analog_filter.Gamma)

#     digital_control = MultiPhaseDigitalControl(Clock(T), phi_delay)

#     # Create a global control object
#     global_control_frontend = get_global_control(
#         AnalogSystem(
#             analog_filter.A,
#             analog_filter.B,
#             analog_filter.CT,
#             Gamma,
#             -Gamma.transpose(),
#         ),
#         digital_control,
#         phi_delay,
#     )

#     print(global_control_frontend.analog_filter)
#     print(global_control_frontend.digital_control)

#     testbench = OpAmpTestBench(
#         global_control_frontend,
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         clock=analog_frontend.digital_control.clock,
#         vdd_voltage=vdd,
#         GBWP=GBWP,
#         DC_gain=DC_gain,
#     )

#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T / (N),
#         number_of_simulated_samples * analog_frontend.digital_control.clock.T,
#         netlist_filename='opamp_Hadamard_phase_test.cir',
#         raw_output_filename='opamp_Hadamard_phase_test.raw',
#     )
#     ngspice_simulator.make_netlist()
#     ngspice_simulator.run()
#     ngspice_simulator.parse()

#     controls = np.zeros((number_of_simulated_samples, N))

#     for i, s in enumerate(ngspice_simulator):
#         controls[i, :] = s

#     figure_name = os.path.join(
#         figure_path, 'ngspice_state_Hadamard_gamma_phase_trajectories.png'
#     )

#     headers, data = ngspice_simulator.get_input_signals()
#     for i in range(1, analog_frontend.analog_filter.L + 1):
#         plt.plot(data[:, 0] / T, data[:, i], label=headers[i])

#     headers, data = ngspice_simulator.get_state_trajectories()

#     diff_states = (
#         data[:, 1 : 1 + analog_filter.N]
#         - data[:, 1 + analog_filter.N : 1 + 2 * analog_filter.N]
#     )
#     print(diff_states.shape)
#     for i in range(analog_frontend.analog_filter.N):
#         plt.plot(
#             data[:, 0] / T,
#             diff_states[:, i],
#             label=f"{headers[i + 1]}-{headers[i + 1 + analog_frontend.analog_filter.N]}",
#         )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i],
#         #     label=f"{headers[i]}",
#         # )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i + analog_frontend.analog_filter.N],
#         #     label=f"{headers[i + analog_frontend.analog_filter.N]}",
#         # )
#     plt.ylim((-1, 1))
#     plt.xlabel('t/T')
#     plt.ylabel('amplitude')
#     plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
#     plt.legend()
#     plt.savefig(figure_name)

#     plt.figure()
#     figure_name_2 = os.path.join(
#         figure_path, f'ngspice_control_Hadamard_gamma_phase_signals.png'
#     )

#     for i in range(N):
#         plt.plot(
#             np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
#             controls[:, i],
#             label="$s_{" + str(i) + "}$",
#         )
#     plt.plot(data[:, 0], data[:, -1], label="$u$")
#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name_2)

#     plot_state_dist(
#         diff_states.transpose(),
#         os.path.join(figure_path, f'ngspice_mul_Hadamard_gamma_state_dist.png'),
#     )
#     ngspice_simulator.cleanup()
#     assert True


# def test_ngspice_try_Hadamard_scaled():
#     os.makedirs(figure_path, exist_ok=True)

#     number_of_simulated_samples = 1 << 12
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     # vdd = 1.0
#     global_amplitude = vdd / np.sqrt(N)
#     # amplitude = vdd
#     DC_gain = 1e8
#     GBWP = DC_gain * BW

#     T = analog_frontend.digital_control.clock.T
#     phi_delay = np.linspace(0, T, N, endpoint=False)

#     analog_filter = analog_frontend.analog_filter
#     Gamma = np.dot(scipy.linalg.hadamard(N) / np.sqrt(N), analog_filter.Gamma)
#     for n in range(N):
#         Gamma[:, n] *= 0.7**n

#     digital_control = MultiPhaseDigitalControl(Clock(T), phi_delay)

#     # Create a global control object
#     global_control_frontend = get_global_control(
#         AnalogSystem(
#             analog_filter.A,
#             analog_filter.B,
#             analog_filter.CT,
#             Gamma,
#             -Gamma.transpose(),
#         ),
#         digital_control,
#         phi_delay,
#     )

#     print(global_control_frontend.analog_filter)
#     print(global_control_frontend.digital_control)

#     testbench = OpAmpTestBench(
#         global_control_frontend,
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         clock=analog_frontend.digital_control.clock,
#         vdd_voltage=vdd,
#         GBWP=GBWP,
#         DC_gain=DC_gain,
#     )

#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T / (N),
#         number_of_simulated_samples * analog_frontend.digital_control.clock.T,
#         netlist_filename='opamp_Hadamard_scaled_phase_test.cir',
#         raw_output_filename='opamp_Hadamard_scaled_phase_test.raw',
#     )
#     ngspice_simulator.make_netlist()
#     ngspice_simulator.run()
#     ngspice_simulator.parse()

#     controls = np.zeros((number_of_simulated_samples, N))

#     for i, s in enumerate(ngspice_simulator):
#         controls[i, :] = s

#     figure_name = os.path.join(
#         figure_path, 'ngspice_state_Hadamard_scaled_gamma_phase_trajectories.png'
#     )

#     headers, data = ngspice_simulator.get_input_signals()
#     for i in range(1, analog_frontend.analog_filter.L + 1):
#         plt.plot(data[:, 0] / T, data[:, i], label=headers[i])

#     headers, data = ngspice_simulator.get_state_trajectories()

#     diff_states = (
#         data[:, 1 : 1 + analog_filter.N]
#         - data[:, 1 + analog_filter.N : 1 + 2 * analog_filter.N]
#     )
#     print(diff_states.shape)
#     for i in range(analog_frontend.analog_filter.N):
#         plt.plot(
#             data[:, 0] / T,
#             diff_states[:, i],
#             label=f"{headers[i + 1]}-{headers[i + 1 + analog_frontend.analog_filter.N]}",
#         )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i],
#         #     label=f"{headers[i]}",
#         # )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i + analog_frontend.analog_filter.N],
#         #     label=f"{headers[i + analog_frontend.analog_filter.N]}",
#         # )
#     plt.ylim((-1, 1))
#     plt.xlabel('t/T')
#     plt.ylabel('amplitude')
#     plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
#     plt.legend()
#     plt.savefig(figure_name)

#     plt.figure()
#     figure_name_2 = os.path.join(
#         figure_path, f'ngspice_control_Hadamard_scaled_gamma_phase_signals.png'
#     )

#     for i in range(N):
#         plt.plot(
#             np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
#             controls[:, i],
#             label="$s_{" + str(i) + "}$",
#         )
#     plt.plot(data[:, 0], data[:, -1], label="$u$")
#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name_2)

#     plot_state_dist(
#         diff_states.transpose(),
#         os.path.join(figure_path, f'ngspice_mul_Hadamard_scaled_gamma_state_dist.png'),
#     )
#     ngspice_simulator.cleanup()
#     assert True
