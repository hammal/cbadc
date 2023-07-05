# from cbadc.circuit.state_space import StateSpaceFrontend
# from cbadc.circuit.components.sources import (
#     SinusoidalVoltageSource,
# )
# from cbadc.circuit import Terminal, SpiceDialect
# from cbadc.circuit.testbench import OpAmpTestBench, OTATestBench, LCTestBench
# from cbadc.synthesis.leap_frog import get_leap_frog
# from cbadc.circuit.simulator import NGSpiceSimulator, SpectreSimulator
# from cbadc.digital_estimator import BatchEstimator
# from cbadc.fom import snr_to_dB, snr_to_enob
# from cbadc.analog_signal import Sinusoidal
# from cbadc.analog_signal import Clock
# from cbadc.analog_frontend import AnalogFrontend
# from . import plot_state_dist
# import os
# import pytest
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.linalg
# import scipy.integrate

# BW = 1e8
# OSR = 8
# omega_p = 2 * np.pi * BW
# T = 1 / (2 * BW * OSR)
# M = 4

# Rin = 1e3
# gm = 1e-3
# C = 10e-12

# vdd = 1.2
# global_amplitude = vdd * 1e-0
# # frequency = BW / 32
# frequency = BW / 3
# frequency = BW / 100
# # frequency = 0.0
# delay = 1e-6
# offset = 0.0
# damping_factor = 0.0
# phase = np.pi / 2

# figure_path = 'lc_tank'
# os.makedirs(figure_path, exist_ok=True)


# def non_homogenious_weights(
#     analog_frontend: AnalogFrontend, input_function
# ) -> np.ndarray:
#     analog_system = analog_frontend.analog_system
#     digital_control = analog_frontend.digital_control

#     def input_derivative(t: float, x: np.ndarray):
#         return np.dot(analog_system.A, x) + analog_system.B.flatten() * input_function(
#             t
#         )

#     y = scipy.integrate.solve_ivp(
#         input_derivative,
#         (0, digital_control.clock.T),
#         np.zeros(analog_system.N),
#     ).y[:, -1]

#     A = np.zeros((analog_system.N, analog_system.M))

#     for m in range(analog_system.M):
#         identity_vector = np.zeros(analog_system.N)
#         identity_vector[m] = 1.0

#         def control_derivative(t: float, x: np.ndarray):
#             return np.dot(analog_system.A, x) + identity_vector

#         A[:, m] = scipy.integrate.solve_ivp(
#             control_derivative,
#             (0, digital_control.clock.T),
#             np.zeros(analog_system.N),
#         ).y[:, -1]

#     return np.linalg.lstsq(A, y)[0]


# def test_non_hom_weights():
#     testbench = LCTestBench(
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         M=M,
#         omega_p=omega_p,
#         clock=Clock(T),
#         Rin=Rin,
#         gm=gm,
#         C=C,
#         vdd_voltage=vdd,
#     )

#     def step(t: float):
#         return 1.0 if t > 0 else 0.0

#     def sinc(t: float):
#         return np.sinc(2 * BW * (t - T / 2))

#     def sin(t: float):
#         return np.sin(2 * np.pi * BW * (t - T / 2))

#     def cos(t: float):
#         return np.cos(2 * np.pi * BW * (t - T / 2))

#     res1 = non_homogenious_weights(testbench.Xaf.extended_analog_frontend, step)
#     print(res1)
#     res2 = non_homogenious_weights(testbench.Xaf.extended_analog_frontend, sinc)
#     print(res2)
#     res3 = non_homogenious_weights(testbench.Xaf.extended_analog_frontend, sin)
#     print(res3)
#     res4 = non_homogenious_weights(testbench.Xaf.extended_analog_frontend, cos)
#     print(res4)
#     assert True


# def test_transfer_function():
#     testbench = LCTestBench(
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         M=M,
#         omega_p=omega_p,
#         clock=Clock(T),
#         Rin=Rin,
#         gm=gm,
#         C=C,
#         vdd_voltage=vdd,
#     )
#     analog_system = testbench.Xaf.extended_analog_frontend.analog_system

#     # Compute the transfer function
#     omega = 2 * np.pi * np.logspace(6, 10, 1000)
#     tf = analog_system.transfer_function_matrix(omega)

#     print(tf.shape)
#     for n in range(tf.shape[0]):
#         for l in range(tf.shape[1]):
#             plt.semilogx(
#                 omega / (2 * np.pi), 20 * np.log10(np.abs(tf[n, l, :])), label=f'n={n}'
#             )
#     plt.xlabel('frequency [Hz]')
#     plt.ylabel('|G(w)| dB')
#     plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
#     plt.legend()
#     figure_name = os.path.join(figure_path, f'LC_tf.png')
#     plt.savefig(figure_name)


# def test_setup():
#     number_of_simulated_samples = 1 << 10

#     testbench = LCTestBench(
#         input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
#         M=M,
#         omega_p=omega_p,
#         clock=Clock(T),
#         Rin=Rin,
#         gm=gm,
#         C=C,
#         vdd_voltage=vdd,
#     )

#     print(testbench.Xaf.analog_frontend.analog_system)
#     print(testbench.Xaf.analog_frontend.digital_control)

#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         T / M,
#         number_of_simulated_samples * T,
#         netlist_filename=f'LC_tank.cir',
#         raw_output_filename=f'LC_tank.raw',
#     )
#     ngspice_simulator.make_netlist()
#     ngspice_simulator.run()
#     ngspice_simulator.parse()

#     controls = np.zeros((number_of_simulated_samples, M))

#     for i, s in enumerate(ngspice_simulator):
#         controls[i, :] = s

#     figure_name = os.path.join(figure_path, f'LC_tank_state_trajectories.png')

#     headers, data = ngspice_simulator.get_input_signals()
#     plt.plot(data[:, 0] / T, data[:, 1] - vdd / 2, label=headers[1])

#     headers, data = ngspice_simulator.get_state_trajectories()

#     diff_states = (
#         data[:, 1 : 1 + M]
#         - data[
#             :,
#             1 + M : 1 + 2 * M,
#         ]
#     )
#     print(diff_states.shape)
#     for i in range(M):
#         plt.plot(
#             data[:, 0] / T,
#             diff_states[:, i],
#             label=f"{headers[i + 1]}-{headers[i + 1 + M]}",
#         )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i],
#         #     label=f"{headers[i]}",
#         # )
#         # plt.plot(
#         #     data[:, 0],
#         #     data[:, i + analog_frontend.analog_system.N],
#         #     label=f"{headers[i + analog_frontend.analog_system.N]}",
#         # )
#     # plt.ylim((-1, 1))
#     plt.xlabel('t/T')
#     plt.ylabel('amplitude')
#     plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
#     plt.legend()
#     plt.savefig(figure_name)

#     plt.figure()
#     figure_name_2 = os.path.join(figure_path, f'LC_tank_control_signals.png')

#     for i in range(M):
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
#         os.path.join(figure_path, f'LC_tank_state_dist.png'),
#         # xlim=(-1e-1, 1e-1),
#     )
#     # ngspice_simulator.cleanup()
#     assert True
