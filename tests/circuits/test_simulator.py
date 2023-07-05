from cbadc.circuit.state_space import StateSpaceFrontend
from cbadc.circuit.components.sources import (
    SinusoidalVoltageSource,
)
from cbadc.circuit import Terminal, SpiceDialect
from cbadc.circuit.testbench import OpAmpTestBench, OTATestBench
from cbadc.synthesis.leap_frog import get_leap_frog
from cbadc.circuit.simulator import NGSpiceSimulator, SpectreSimulator
from cbadc.digital_estimator import BatchEstimator
from cbadc.utilities import (
    compute_power_spectral_density,
    find_sinusoidal,
    snr_spectrum_computation_extended,
)
from cbadc.fom import snr_to_dB, snr_to_enob
from cbadc.analog_signal import Sinusoidal
from cbadc.digital_control import DitherControl, MultiPhaseDigitalControl
from cbadc.analog_system import AnalogSystem
from cbadc.analog_frontend import AnalogFrontend, get_global_control
from cbadc.simulator import get_simulator
from cbadc.analog_signal import Clock
from . import plot_state_dist
import os
import pytest
import matplotlib.pyplot as plt
import numpy as np
import subprocess


two_terminals = [Terminal() for _ in range(2)]
three_terminals = two_terminals + [Terminal()]
N = 6
ENOB = 15
BW = 1e6

xi = 5e-2
analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW, xi=xi)


max_input_frequency = BW / 2.0
estimate_simulation_length = 1 << 15
K1 = 1 << 11
K2 = K1
show_state_plots = True


# vdd = 3.7
vdd = 1.0

global_amplitude = vdd * 1e-0
# amplitude = 0.0
frequency = 1e6 / 32
delay = 1e-60
offset = 0.0
damping_factor = 0.0
phase = 0.0


# def test_ngspice_simulator():
#     number_of_simulated_samples = 1 << 8
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     testbench = StateSpaceTestBench(
#         analog_frontend=StateSpaceAnalogFrontend(analog_frontend, vdd),
#         input_signals=[
#             SinusoidalVoltageSource(
#                 offset=offset,
#                 amplitude=amplitude,
#                 frequency=frequency,
#                 delay_time=delay,
#                 phase=phase,
#                 damping_factor=damping_factor,
#                 instance_name='2',
#             )
#         ],
#         clock=analog_frontend.digital_control.clock,
#     )

#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T,
#         (number_of_simulated_samples + K1 + K2) * analog_frontend.digital_control.clock.T,
#     )
#     ngspice_simulator.make_netlist()
#     ngspice_simulator.cleanup()
#     assert True


# def test_spectre_simulator():
#     number_of_simulated_samples = 1 << 8
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     testbench = StateSpaceTestBench(
#         analog_frontend=StateSpaceAnalogFrontend(analog_frontend, vdd),
#         input_signals=[
#             SinusoidalVoltageSource(
#                 offset=offset,
#                 amplitude=amplitude,
#                 frequency=frequency,
#                 delay_time=delay,
#                 phase=phase,
#                 damping_factor=damping_factor,
#                 instance_name='2',
#             )
#         ],
#         clock=analog_frontend.digital_control.clock,
#     )

#     with pytest.raises(FileNotFoundError):
#         spectre_simulator = SpectreSimulator(
#             testbench,
#             (number_of_simulated_samples + K1 + K2) * analog_frontend.digital_control.clock.T,
#             1 / analog_frontend.digital_control.clock.T,
#             0.0,
#         )
#         spectre_simulator.cleanup()
#     assert True


# def test_ngspice_simulator_iterator():
#     number_of_simulated_samples = 1 << 9
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     testbench = StateSpaceTestBench(
#         analog_frontend=StateSpaceAnalogFrontend(analog_frontend, vdd),
#         input_signals=[
#             SinusoidalVoltageSource(
#                 offset=offset,
#                 amplitude=amplitude,
#                 frequency=frequency,
#                 delay_time=delay,
#                 phase=phase,
#                 damping_factor=damping_factor,
#                 instance_name='2',
#             )
#         ],
#         clock=analog_frontend.digital_control.clock,
#     )
#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T,
#         (number_of_simulated_samples + K1 + K2) * analog_frontend.digital_control.clock.T,
#     )

#     for s in ngspice_simulator:
#         print(s)
#     ngspice_simulator.cleanup()
#     assert True


# def test_ngspice_simulator_state_trajectories():
#     number_of_simulated_samples = 1 << 12
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     testbench = StateSpaceTestBench(
#         analog_frontend=StateSpaceAnalogFrontend(
#             analog_frontend, vdd, in_high=0.0, in_low=-0.0
#         ),
#         input_signals=[
#             SinusoidalVoltageSource(
#                 offset=offset,
#                 amplitude=amplitude,
#                 frequency=frequency,
#                 delay_time=delay,
#                 phase=phase,
#                 damping_factor=damping_factor,
#                 instance_name='2',
#             )
#         ],
#         clock=analog_frontend.digital_control.clock,
#     )
#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T,
#         (number_of_simulated_samples + K1 + K2) * analog_frontend.digital_control.clock.T,
#         netlist_filename='state_test.cir',
#         raw_output_filename='state_test.raw',
#     )

#     ngspice_simulator.make_netlist()
#     ngspice_simulator.run()
#     ngspice_simulator.parse()

#     data = ngspice_simulator.get_state_trajectories()

#     controls = np.zeros((number_of_simulated_samples, N))

#     for i, s in enumerate(ngspice_simulator):
#         controls[i, :] = s

#     print(data)
#     figure_name = 'ngspice_state_trajectories.png'

#     for i in range(1, N + 1):
#         plt.plot(data[:, 0], data[:, i], label="$x_{" + str(i) + "}$")
#     plt.plot(data[:, 0], data[:, -1], label="$u$")
#     # for i in range(1,N):
#     #     plt.plot(np.linspace(data[0, 0],data[-1, 0], controls.shape[0]), controls[:, i], label="$s_{" + str(i) + "}$")
#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name)

#     plt.figure()
#     figure_name_2 = 'ngspice_control_signals.png'

#     # for i in range(1,N + 1):
#     #     plt.plot(data[:, 0], data[:, i], label="$x_{" + str(i) + "}$")
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

#     plt.figure()
#     figure_name_3 = 'ngspice_control_observations.png'

#     control_observations = ngspice_simulator.get_state_observations()

#     for i in range(1, N + 1):
#         plt.plot(
#             control_observations[:, 0],
#             control_observations[:, i],
#             label="$\\tilde{s}_{" + str(i) + "}$",
#         )
#     plt.plot(control_observations[:, 0], control_observations[:, -1], label="$u$")
#     plt.xlabel('time [s]')
#     plt.ylabel('amplitude')
#     plt.legend()
#     plt.savefig(figure_name_3)

#     os.remove(figure_name)
#     os.remove(figure_name_2)
#     os.remove(figure_name_3)
#     ngspice_simulator.cleanup()
#     assert True


# def test_ngspice_simulator_estimate():
#     number_of_simulated_samples = 1 << 14
#     # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
#     testbench = StateSpaceTestBench(
#         analog_frontend=StateSpaceAnalogFrontend(analog_frontend, vdd),
#         input_signals=[
#             SinusoidalVoltageSource(
#                 offset=offset,
#                 amplitude=amplitude,
#                 frequency=frequency,
#                 delay_time=delay,
#                 phase=phase,
#                 damping_factor=damping_factor,
#                 instance_name='2',
#             )
#         ],
#         clock=analog_frontend.digital_control.clock,
#     )
#     ngspice_simulator = NGSpiceSimulator(
#         testbench,
#         analog_frontend.digital_control.clock.T,
#         (number_of_simulated_samples + K1 + K2) * analog_frontend.digital_control.clock.T,
#         netlist_filename='estimate_test.cir',
#         raw_output_filename='estimate_test.raw',
#     )

#     eta2 = (
#         np.linalg.norm(
#             analog_frontend.analog_system.transfer_function_matrix(
#                 np.array([2 * np.pi * BW])
#             )
#         )
#         ** 2
#     )
#     K1 = 1 << 10
#     K2 = K1
#     digital_estimator = BatchEstimator(
#         analog_frontend.analog_system, analog_frontend.digital_control, eta2, K1, K2
#     )

#     digital_estimator(ngspice_simulator)

#     u_hat = np.zeros(number_of_simulated_samples)
#     for i, e in enumerate(digital_estimator):
#         u_hat[i] = e

#     u_hat_cut = u_hat[K1 + K2 :]
#     f, psd = compute_power_spectral_density(
#         u_hat_cut[:],
#         fs=1 / analog_frontend.digital_control.clock.T,
#         nperseg=u_hat_cut.size,
#     )
#     signal_index = find_sinusoidal(psd, 15)
#     noise_index = np.ones(psd.size, dtype=bool)
#     noise_index[signal_index] = False
#     noise_index[f < (BW * 1e-2)] = False
#     noise_index[f > BW] = False
#     fom = snr_spectrum_computation_extended(
#         psd, signal_index, noise_index, fs=1 / analog_frontend.digital_control.clock.T
#     )
#     est_SNR = snr_to_dB(fom['snr'])
#     est_ENOB = snr_to_enob(est_SNR)

#     plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
#     plt.semilogx(
#         f,
#         10 * np.log10(np.abs(psd)),
#         label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}",
#     )
#     plt.xlabel('Hz')
#     plt.ylabel('V^2 / Hz dB')
#     plt.legend()
#     figure_name = 'ngspice_estimate.png'
#     plt.savefig(figure_name)
#     ngspice_simulator.cleanup()
#     os.remove(figure_name)
#     assert est_ENOB >= ENOB

TIME_STEPS = 1 << 8


def save_time_plot(u_hat: np.ndarray, filename: str):
    plt.figure()
    plt.title(f"Time Plot")
    plt.plot(u_hat[:TIME_STEPS])
    plt.xlabel('time [s]')
    plt.ylabel('u_hat')
    plt.legend()
    plt.savefig(filename)

    filename_path, filename = os.path.split(filename)

    plt.figure()
    plt.title(f"Time Plot")
    plt.plot(u_hat[:])
    plt.xlabel('time [s]')
    plt.ylabel('u_hat')
    plt.legend()
    plt.savefig(os.path.join(filename_path, 'full' + filename))


def test_nominal():
    number_of_simulated_samples = estimate_simulation_length
    # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
    # vdd = 1.2
    amplitude = 0.9
    # amplitude = 0.0
    frequency = 1 / analog_frontend.digital_control.clock.T
    while frequency > max_input_frequency:
        frequency /= 2

    # Logspace frequencies
    frequencies = np.logspace(3, 7, 256)
    omega = 2 * np.pi * frequencies

    # Compute transfer functions for each frequency in frequencies
    transfer_function = analog_frontend.analog_system.transfer_function_matrix(omega)
    transfer_function_dB = 20 * np.log10(np.abs(transfer_function))

    # For each output 1,...,N compute the corresponding tranfer function seen
    # from the input.
    plt.figure()
    plt.title(f"Transfer function")
    for n in range(N):
        plt.semilogx(
            frequencies,
            transfer_function_dB[n, 0, :],
            label=f"$G_{n+1}(\omega)$",
        )
    plt.xlabel('Hz')
    plt.ylabel('|G(w)| dB')
    plt.legend()
    plt.grid()
    figure_name = 'nominal_transfer_function.png'
    plt.savefig(figure_name)

    simulator = get_simulator(
        analog_frontend.analog_system,
        analog_frontend.digital_control,
        [Sinusoidal(amplitude, frequency, phase, offset)],
    )

    eta2 = (
        np.linalg.norm(
            analog_frontend.analog_system.transfer_function_matrix(
                np.array([2 * np.pi * BW])
            )
        )
        ** 2
    )
    # K1 = 1 << 12
    # K2 = K1
    digital_estimator = BatchEstimator(
        analog_frontend.analog_system, analog_frontend.digital_control, eta2, K1, K2
    )

    digital_estimator(simulator)

    u_hat = np.zeros(number_of_simulated_samples)
    for i in range(number_of_simulated_samples):
        u_hat[i] = next(digital_estimator)

    u_hat_cut = u_hat[K1 + K2 :]
    f, psd = compute_power_spectral_density(
        u_hat_cut[:],
        fs=1 / analog_frontend.digital_control.clock.T,
        nperseg=u_hat_cut.size,
    )
    signal_index = find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / analog_frontend.digital_control.clock.T
    )
    est_SNR = snr_to_dB(fom['snr'])
    est_ENOB = snr_to_enob(est_SNR)

    plt.figure()
    plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}",
    )
    plt.xlabel('Hz')
    plt.ylabel('V^2 / Hz dB')
    plt.legend()
    figure_name = 'nominal_estimate.png'
    plt.savefig(figure_name)

    save_time_plot(u_hat, 'nominal_time_plot.png')

    assert est_ENOB >= ENOB


def test_ngspice_simulator_with_opamp_testbench():
    number_of_simulated_samples = 1 << 12
    # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
    # vdd = 1.2
    # amplitude = vdd
    DC_gain = 1e8
    GBWP = DC_gain * BW
    testbench = OpAmpTestBench(
        analog_frontend,
        input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
        clock=analog_frontend.digital_control.clock,
        vdd_voltage=vdd,
        GBWP=GBWP,
        DC_gain=DC_gain,
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend.digital_control.clock.T,
        number_of_simulated_samples * analog_frontend.digital_control.clock.T,
        netlist_filename='opamp_test.cir',
        raw_output_filename='opamp_test.raw',
    )
    ngspice_simulator.make_netlist()
    ngspice_simulator.run()
    ngspice_simulator.parse()

    controls = np.zeros((number_of_simulated_samples, N))

    for i, s in enumerate(ngspice_simulator):
        controls[i, :] = s

    figure_name = 'ngspice_state_trajectories_1.png'

    headers, data = ngspice_simulator.get_input_signals()
    for i in range(1, analog_frontend.analog_system.L + 1):
        plt.plot(data[:, 0], data[:, i], label=headers[i])

    headers, data = ngspice_simulator.get_state_trajectories()
    for i in range(1, analog_frontend.analog_system.N + 1):
        plt.plot(
            data[:, 0],
            data[:, i] - data[:, i + analog_frontend.analog_system.N],
            label=f"{headers[i]}-{headers[i + analog_frontend.analog_system.N]}",
        )
        # plt.plot(
        #     data[:, 0],
        #     data[:, i],
        #     label=f"{headers[i]}",
        # )
        # plt.plot(
        #     data[:, 0],
        #     data[:, i + analog_frontend.analog_system.N],
        #     label=f"{headers[i + analog_frontend.analog_system.N]}",
        # )

    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.legend()
    plt.savefig(figure_name)

    plt.figure()
    figure_name_2 = 'ngspice_control_signals_1.png'

    for i in range(N):
        plt.plot(
            np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
            controls[:, i],
            label="$s_{" + str(i) + "}$",
        )
    plt.plot(data[:, 0], data[:, -1], label="$u$")
    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.legend()
    plt.savefig(figure_name_2)

    # plt.figure()
    # figure_name_3 = 'ngspice_control_observations.png'

    # control_observations = ngspice_simulator.get_state_observations()

    # for i in range(1, N + 1):
    #     plt.plot(
    #         control_observations[:, 0],
    #         control_observations[:, i],
    #         label="$\\tilde{s}_{" + str(i) + "}$",
    #     )
    # plt.plot(control_observations[:, 0], control_observations[:, -1], label="$u$")
    # plt.xlabel('time [s]')
    # plt.ylabel('amplitude')
    # plt.legend()
    # plt.savefig(figure_name_3)
    # os.remove(figure_name)
    # os.remove(figure_name_2)
    # os.remove(figure_name_3)
    diff_states = (
        data[:, 1 : 1 + analog_frontend.analog_system.N]
        - data[
            :,
            1
            + analog_frontend.analog_system.N : 1
            + 2 * analog_frontend.analog_system.N,
        ]
    )
    plot_state_dist(diff_states.transpose(), 'ngspice_opamp_ref_state_dist.png')
    ngspice_simulator.cleanup()
    assert True


def test_ngspice_opamp_simulator_estimate():
    number_of_simulated_samples = estimate_simulation_length
    # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
    # vdd = 1.2
    # amplitude = vdd
    # amplitude = 0.0
    DC_gain = 1e5
    GBWP = DC_gain * BW * 1e3
    frequency = 1 / analog_frontend.digital_control.clock.T
    while frequency > max_input_frequency:
        frequency /= 2
    testbench = OpAmpTestBench(
        analog_frontend,
        input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
        clock=analog_frontend.digital_control.clock,
        vdd_voltage=vdd,
        GBWP=GBWP,
        DC_gain=DC_gain,
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend.digital_control.clock.T,
        (number_of_simulated_samples + K1 + K2)
        * analog_frontend.digital_control.clock.T,
        netlist_filename='opamp_estimate_test.cir',
        raw_output_filename='opamp_estimate_test.raw',
    )
    ngspice_simulator.make_netlist()
    ngspice_simulator.run()
    ngspice_simulator.parse()

    eta2 = (
        np.linalg.norm(
            analog_frontend.analog_system.transfer_function_matrix(
                np.array([2 * np.pi * BW])
            )
        )
        ** 2
    )
    # K1 = 1 << 12
    # K2 = K1
    digital_estimator = BatchEstimator(
        analog_frontend.analog_system, analog_frontend.digital_control, eta2, K1, K2
    )

    digital_estimator(ngspice_simulator)

    u_hat = np.zeros(number_of_simulated_samples)
    for i in range(number_of_simulated_samples):
        u_hat[i] = next(digital_estimator)

    u_hat_cut = u_hat[K1 + K2 :]
    f, psd = compute_power_spectral_density(
        u_hat_cut[:],
        fs=1 / analog_frontend.digital_control.clock.T,
        nperseg=u_hat_cut.size,
    )
    signal_index = find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / analog_frontend.digital_control.clock.T
    )
    est_SNR = snr_to_dB(fom['snr'])
    est_ENOB = snr_to_enob(est_SNR)

    plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}, s_time={ngspice_simulator.simulation_time}",
    )
    plt.xlabel('Hz')
    plt.ylabel('V^2 / Hz dB')
    plt.legend()
    figure_name = 'ngspice_opamp_estimate.png'
    plt.savefig(figure_name)

    if show_state_plots:
        figure_name_2 = 'ngspice_opamp_estimate_state_trajectories.png'
        plt.figure()
        headers, data = ngspice_simulator.get_input_signals()
        for i in range(1, data.shape[1]):
            plt.plot(data[:, 0], data[:, i], label=headers[i])
        headers, data = ngspice_simulator.get_state_trajectories()
        for i in range(1, data.shape[1]):
            plt.plot(data[:, 0], data[:, i], label=headers[i])
        plt.xlabel('time [s]')
        plt.ylabel('amplitude')
        plt.legend()
        plt.savefig(figure_name_2)

    # ngspice_simulator.cleanup()
    # os.remove(figure_name)
    # if show_state_plots:
    # os.remove(figure_name_2)
    save_time_plot(u_hat, 'opamp_time_plot.png')
    save_time_plot(
        ngspice_simulator.control_vector[:, 0], 'opamp_control_time_plot.png'
    )

    assert est_ENOB >= ENOB


def test_ngspice_simulator_with_ota_testbench():
    number_of_simulated_samples = 1 << 8
    # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
    # vdd = 2.4
    # amplitude = vdd
    DC_GAIN = 1e4
    testbench = OTATestBench(
        analog_frontend,
        input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
        clock=analog_frontend.digital_control.clock,
        vdd_voltage=vdd,
        DC_gain=DC_GAIN,
        # C_int=1e-16
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend.digital_control.clock.T,
        (number_of_simulated_samples) * analog_frontend.digital_control.clock.T,
        netlist_filename='ota_test.cir',
        raw_output_filename='ota_test.raw',
    )
    ngspice_simulator.make_netlist()
    ngspice_simulator.run()
    ngspice_simulator.parse()

    controls = np.zeros((number_of_simulated_samples, N))

    for i, s in enumerate(ngspice_simulator):
        controls[i, :] = s

    figure_name = 'ngspice_state_trajectories_2.png'

    headers, data = ngspice_simulator.get_state_trajectories()
    for i in range(1, data.shape[1]):
        plt.plot(data[:, 0], data[:, i], label=headers[i])
    headers, data = ngspice_simulator.get_input_signals()
    for i in range(1, data.shape[1]):
        plt.plot(data[:, 0], data[:, i], label=headers[i])
    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.legend()
    plt.savefig(figure_name)

    plt.figure()
    figure_name_2 = 'ngspice_control_signals.png'

    # for i in range(1,N + 1):
    #     plt.plot(data[:, 0], data[:, i], label="$x_{" + str(i) + "}$")
    for i in range(N):
        plt.plot(
            np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
            controls[:, i],
            label="$s_{" + str(i) + "}$",
        )
    plt.plot(data[:, 0], data[:, -1], label="$u$")
    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.legend()
    plt.savefig(figure_name_2)

    plt.figure()
    figure_name_3 = 'ngspice_control_observations_2.png'

    # control_observations = ngspice_simulator.get_state_observations()

    # for i in range(1, N + 1):
    #     plt.plot(
    #         control_observations[:, 0],
    #         control_observations[:, i],
    #         label="$\\tilde{s}_{" + str(i) + "}$",
    #     )
    # plt.plot(control_observations[:, 0], control_observations[:, -1], label="$u$")
    # plt.xlabel('time [s]')
    # plt.ylabel('amplitude')
    # plt.legend()
    # plt.savefig(figure_name_3)

    os.remove(figure_name)
    os.remove(figure_name_2)
    # os.remove(figure_name_3)
    ngspice_simulator.cleanup()
    assert True


def test_ngspice_ota_simulator_estimate():
    analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW, xi=xi)

    number_of_simulated_samples = estimate_simulation_length
    # vdd = 2.4
    # amplitude = vdd
    # amplitude = 0.0
    DC_gain = np.inf
    # K1 = 1 << 12
    # K2 = K1
    frequency = 1.0 / analog_frontend.digital_control.clock.T
    while frequency > max_input_frequency:
        frequency /= 2
    while (number_of_simulated_samples - K1 - K2) % int(
        1 / (frequency * analog_frontend.digital_control.clock.T)
    ) != 0:
        number_of_simulated_samples += 1

    testbench = OTATestBench(
        analog_frontend,
        input_signals=[Sinusoidal(global_amplitude, frequency, phase, offset)],
        clock=analog_frontend.digital_control.clock,
        vdd_voltage=vdd,
        DC_gain=DC_gain,
        # C_int=1e-16
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend.digital_control.clock.T,
        (number_of_simulated_samples + K1 + K2)
        * analog_frontend.digital_control.clock.T,
        netlist_filename='ota_estimate_test.cir',
        raw_output_filename='ota_estimate_test.raw',
    )
    ngspice_simulator.make_netlist()
    ngspice_simulator.run()
    ngspice_simulator.parse()

    eta2 = (
        np.linalg.norm(
            analog_frontend.analog_system.transfer_function_matrix(
                np.array([2 * np.pi * BW])
            )
        )
        ** 2
    )

    digital_estimator = BatchEstimator(
        analog_frontend.analog_system, analog_frontend.digital_control, eta2, K1, K2
    )

    digital_estimator(ngspice_simulator)

    u_hat = np.zeros(number_of_simulated_samples)
    for i in range(number_of_simulated_samples):
        u_hat[i] = next(digital_estimator)

    u_hat_cut = u_hat[K1 + K2 :]
    f, psd = compute_power_spectral_density(
        u_hat_cut[:],
        fs=1 / analog_frontend.digital_control.clock.T,
        nperseg=u_hat_cut.size,
    )
    signal_index = find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / analog_frontend.digital_control.clock.T
    )
    est_SNR = snr_to_dB(fom['snr'])
    est_ENOB = snr_to_enob(est_SNR)
    OSR = 1 / (2 * analog_frontend.digital_control.clock.T * BW)
    gamma = OSR / (2 * np.pi)
    plt.title(
        f"Power spectral density:\nN={N}, ENOB={ENOB}, BW={BW:.0e}, OSR={OSR:.2f}, $\gamma$={gamma:.2f}"
    )
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"est_ENOB={est_ENOB:.1f} bits,\nest_SNR={est_SNR:.1f} dB,\nsim_time={ngspice_simulator.simulation_time}",
    )
    plt.xlabel('Hz')
    plt.ylabel('V^2 / Hz dB')
    plt.legend()
    plt.grid(True)
    figure_name = 'ngspice_ota_estimate.png'
    plt.savefig(figure_name)

    figure_name_2 = 'ngspice_ota_estimate_state_trajectories.png'
    if show_state_plots:
        plt.figure()
        headers, data = ngspice_simulator.get_input_signals()
        for i in range(1, data.shape[1]):
            plt.plot(data[:, 0], data[:, i], label=headers[i])
        headers, data = ngspice_simulator.get_state_trajectories()
        for i in range(1, data.shape[1]):
            plt.plot(data[:, 0], data[:, i], label=headers[i])
        plt.xlabel('time [s]')
        plt.ylabel('amplitude')
        plt.legend()
        plt.savefig(figure_name_2)
    save_time_plot(u_hat, 'ota_time_plot.png')

    # ngspice_simulator.cleanup()
    # os.remove(figure_name)
    # if show_state_plots:
    # os.remove(figure_name_2)
    assert est_ENOB >= ENOB


def test_ngspice_simulator_with_opamp_testbench_and_dither_control():
    number_of_simulated_samples = 1 << 4
    # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
    Gamma = np.hstack((np.zeros((N, 1)), analog_frontend.analog_system.Gamma))
    Gamma[0, 0] = analog_frontend.analog_system.Gamma[0, 0] / 10.0

    analog_frontend_modified = AnalogFrontend(
        AnalogSystem(
            analog_frontend.analog_system.A,
            analog_frontend.analog_system.B,
            analog_frontend.analog_system.CT,
            Gamma,
            analog_frontend.analog_system.Gamma_tildeT,
        ),
        DitherControl(1, analog_frontend.digital_control),
    )

    # vdd = 1.2
    amplitude = global_amplitude * 0.9
    DC_gain = 1e8
    GBWP = DC_gain * BW
    testbench = OpAmpTestBench(
        analog_frontend_modified,
        input_signals=[Sinusoidal(amplitude, frequency, phase, offset)],
        clock=analog_frontend.digital_control.clock,
        vdd_voltage=vdd,
        GBWP=GBWP,
        DC_gain=DC_gain,
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend.digital_control.clock.T,
        (number_of_simulated_samples) * analog_frontend.digital_control.clock.T,
        netlist_filename='opamp_test_with_dither.cir',
        raw_output_filename='opamp_test_with_dither.raw',
    )
    ngspice_simulator.make_netlist()
    ngspice_simulator.run()
    ngspice_simulator.parse()

    controls = np.zeros((number_of_simulated_samples, N + 1))

    for i, s in enumerate(ngspice_simulator):
        controls[i, :] = s

    figure_name = 'ngspice_state_trajectories_opamp_test_with_dither.png'

    headers, data = ngspice_simulator.get_input_signals()
    for i in range(1, data.shape[1]):
        plt.plot(data[:, 0], data[:, i], label=headers[i])

    headers, data = ngspice_simulator.get_state_trajectories()
    for i in range(1, data.shape[1]):
        plt.plot(data[:, 0], data[:, i], label=headers[i])

    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.legend()
    plt.savefig(figure_name)

    plt.figure()
    figure_name_2 = 'ngspice_control_signals_opamp_test_with_dither.png'

    for i in range(N + 1):
        plt.plot(
            np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
            controls[:, i],
            label="$s_{" + str(i) + "}$",
        )
    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.legend()
    plt.savefig(figure_name_2)

    os.remove(figure_name)
    os.remove(figure_name_2)
    # os.remove(figure_name_3)
    ngspice_simulator.cleanup()
    assert True


def test_ngspice_opamp_simulator_estimate_and_dither_control():
    kappa_0 = 0.05
    number_of_simulated_samples = estimate_simulation_length
    # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
    Gamma = np.hstack((np.zeros((N, 1)), analog_frontend.analog_system.Gamma))
    Gamma[0, 0] = analog_frontend.analog_system.Gamma[0, 0] * kappa_0

    analog_frontend_modified = AnalogFrontend(
        AnalogSystem(
            analog_frontend.analog_system.A,
            analog_frontend.analog_system.B,
            analog_frontend.analog_system.CT,
            Gamma,
            analog_frontend.analog_system.Gamma_tildeT,
        ),
        DitherControl(1, analog_frontend.digital_control),
    )

    # vdd = 1.2
    amplitude = global_amplitude * (1.0 - kappa_0)
    # amplitude = 0.0
    DC_gain = 1e5
    GBWP = DC_gain * BW * 1e3
    frequency = 1 / analog_frontend.digital_control.clock.T
    while frequency > max_input_frequency:
        frequency /= 2
    testbench = OpAmpTestBench(
        analog_frontend_modified,
        input_signals=[Sinusoidal(amplitude, frequency, phase, offset)],
        clock=analog_frontend.digital_control.clock,
        vdd_voltage=vdd,
        GBWP=GBWP,
        DC_gain=DC_gain,
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend.digital_control.clock.T,
        (number_of_simulated_samples + K1 + K2)
        * analog_frontend.digital_control.clock.T,
        netlist_filename='opamp_estimate_test_and_dither_control.cir',
        raw_output_filename='opamp_estimate_test_and_dither_control.raw',
    )
    ngspice_simulator.make_netlist()
    ngspice_simulator.run()
    ngspice_simulator.parse()

    eta2 = (
        np.linalg.norm(
            analog_frontend.analog_system.transfer_function_matrix(
                np.array([2 * np.pi * BW])
            )
        )
        ** 2
    )
    # K1 = 1 << 12
    # K2 = K1
    digital_estimator = BatchEstimator(
        analog_frontend_modified.analog_system,
        analog_frontend_modified.digital_control,
        eta2,
        K1,
        K2,
    )

    digital_estimator(ngspice_simulator)

    u_hat = np.zeros(number_of_simulated_samples)
    for i in range(number_of_simulated_samples):
        u_hat[i] = next(digital_estimator)

    u_hat_cut = u_hat[K1 + K2 :]
    f, psd = compute_power_spectral_density(
        u_hat_cut[:],
        fs=1 / analog_frontend.digital_control.clock.T,
        nperseg=u_hat_cut.size,
    )
    signal_index = find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / analog_frontend.digital_control.clock.T
    )
    est_SNR = snr_to_dB(fom['snr'])
    est_ENOB = snr_to_enob(est_SNR)

    plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}, s_time={ngspice_simulator.simulation_time}",
    )
    plt.xlabel('Hz')
    plt.ylabel('V^2 / Hz dB')
    plt.legend()
    figure_name = 'ngspice_opamp_estimate_and_dither_control.png'
    plt.savefig(figure_name)

    figure_name_2 = 'ngspice_opamp_estimate_state_trajectories_and_dither_control.png'
    if show_state_plots:
        plt.figure()
        headers, data = ngspice_simulator.get_input_signals()
        for i in range(1, data.shape[1]):
            plt.plot(data[:, 0], data[:, i], label=headers[i])
        headers, data = ngspice_simulator.get_state_trajectories()
        for i in range(1, data.shape[1]):
            plt.plot(data[:, 0], data[:, i], label=headers[i])
        plt.xlabel('time [s]')
        plt.ylabel('amplitude')
        plt.legend()
        plt.savefig(figure_name_2)

    save_time_plot(u_hat, 'opamp_dither_time_plot.png')

    # ngspice_simulator.cleanup()
    # os.remove(figure_name)
    # if show_state_plots:
    #     os.remove(figure_name_2)
    assert est_ENOB >= ENOB


def test_ngspice_simulator_with_ota_and_dither_control_testbench():
    number_of_simulated_samples = 1 << 8
    # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
    Gamma = np.hstack((np.zeros((N, 1)), analog_frontend.analog_system.Gamma))
    Gamma[0, 0] = analog_frontend.analog_system.Gamma[0, 0] / 10.0

    analog_frontend_modified = AnalogFrontend(
        AnalogSystem(
            analog_frontend.analog_system.A,
            analog_frontend.analog_system.B,
            analog_frontend.analog_system.CT,
            Gamma,
            analog_frontend.analog_system.Gamma_tildeT,
        ),
        DitherControl(1, analog_frontend.digital_control),
    )
    # vdd = 2.4
    amplitude = global_amplitude * 0.9
    DC_GAIN = 1e4
    testbench = OTATestBench(
        analog_frontend_modified,
        input_signals=[Sinusoidal(amplitude, frequency, phase, offset)],
        clock=analog_frontend.digital_control.clock,
        vdd_voltage=vdd,
        DC_gain=DC_GAIN,
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend.digital_control.clock.T,
        (number_of_simulated_samples) * analog_frontend.digital_control.clock.T,
        netlist_filename='ota_test_and_dither_control.cir',
        raw_output_filename='ota_test_and_dither_control.raw',
    )
    ngspice_simulator.make_netlist()
    ngspice_simulator.run()
    ngspice_simulator.parse()

    controls = np.zeros((number_of_simulated_samples, N + 1))

    for i, s in enumerate(ngspice_simulator):
        controls[i, :] = s

    figure_name = 'ngspice_state_trajectories_and_dither_control.png'

    headers, data = ngspice_simulator.get_state_trajectories()
    for i in range(1, data.shape[1]):
        plt.plot(data[:, 0], data[:, i], label=headers[i])
    headers, data = ngspice_simulator.get_input_signals()
    for i in range(1, data.shape[1]):
        plt.plot(data[:, 0], data[:, i], label=headers[i])
    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.legend()
    plt.savefig(figure_name)

    plt.figure()
    figure_name_2 = 'ngspice_control_signals_and_dither_control.png'

    # for i in range(1,N + 1):
    #     plt.plot(data[:, 0], data[:, i], label="$x_{" + str(i) + "}$")
    for i in range(N + 1):
        plt.plot(
            np.linspace(data[0, 0], data[-1, 0], controls.shape[0]),
            controls[:, i],
            label="$s_{" + str(i) + "}$",
        )
    plt.plot(data[:, 0], data[:, -1], label="$u$")
    plt.xlabel('time [s]')
    plt.ylabel('amplitude')
    plt.legend()
    plt.savefig(figure_name_2)

    plt.figure()

    os.remove(figure_name)
    os.remove(figure_name_2)
    # os.remove(figure_name_3)
    ngspice_simulator.cleanup()
    assert True


def test_ngspice_ota_simulator_estimate_and_dither_control():
    kappa_0 = 0.05
    number_of_simulated_samples = estimate_simulation_length
    # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
    Gamma = np.hstack((np.zeros((N, 1)), analog_frontend.analog_system.Gamma))
    Gamma[0, 0] = analog_frontend.analog_system.Gamma[0, 0] * kappa_0

    analog_frontend_modified = AnalogFrontend(
        AnalogSystem(
            analog_frontend.analog_system.A,
            analog_frontend.analog_system.B,
            analog_frontend.analog_system.CT,
            Gamma,
            analog_frontend.analog_system.Gamma_tildeT,
        ),
        DitherControl(1, analog_frontend.digital_control),
    )
    # vdd = 2.4
    amplitude = global_amplitude * (1.0 - kappa_0)
    # amplitude = 0.0
    DC_gain = np.inf
    frequency = 1.0 / analog_frontend.digital_control.clock.T
    while frequency > max_input_frequency:
        frequency /= 2

    testbench = OTATestBench(
        analog_frontend_modified,
        input_signals=[Sinusoidal(amplitude, frequency, phase, offset)],
        clock=analog_frontend.digital_control.clock,
        vdd_voltage=vdd,
        DC_gain=DC_gain,
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend.digital_control.clock.T,
        (number_of_simulated_samples + K1 + K2)
        * analog_frontend.digital_control.clock.T,
        netlist_filename='ota_estimate_test_and_dither_control.cir',
        raw_output_filename='ota_estimate_test_and_dither_control.raw',
    )
    ngspice_simulator.make_netlist()
    ngspice_simulator.run()
    ngspice_simulator.parse()

    eta2 = (
        np.linalg.norm(
            analog_frontend.analog_system.transfer_function_matrix(
                np.array([2 * np.pi * BW])
            )
        )
        ** 2
    )
    # K1 = 1 << 12
    # K2 = K1
    digital_estimator = BatchEstimator(
        analog_frontend_modified.analog_system,
        analog_frontend_modified.digital_control,
        eta2,
        K1,
        K2,
    )

    digital_estimator(ngspice_simulator)

    u_hat = np.zeros(number_of_simulated_samples)
    for i in range(number_of_simulated_samples):
        u_hat[i] = next(digital_estimator)

    u_hat_cut = u_hat[K1 + K2 :]
    f, psd = compute_power_spectral_density(
        u_hat_cut[:],
        fs=1 / analog_frontend.digital_control.clock.T,
        nperseg=u_hat_cut.size,
    )
    signal_index = find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / analog_frontend.digital_control.clock.T
    )
    est_SNR = snr_to_dB(fom['snr'])
    est_ENOB = snr_to_enob(est_SNR)

    plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}, s_time={ngspice_simulator.simulation_time}",
    )
    plt.xlabel('Hz')
    plt.ylabel('V^2 / Hz dB')
    plt.legend()
    figure_name = 'ngspice_ota_estimate_and_dither_control.png'
    plt.savefig(figure_name)

    figure_name_2 = 'ngspice_ota_estimate_and_dither_control_state_trajectories.png'
    if show_state_plots:
        plt.figure()
        headers, data = ngspice_simulator.get_input_signals()
        for i in range(1, data.shape[1]):
            plt.plot(data[:, 0], data[:, i], label=headers[i])
        headers, data = ngspice_simulator.get_state_trajectories()
        for i in range(1, data.shape[1]):
            plt.plot(data[:, 0], data[:, i], label=headers[i])
        plt.xlabel('time [s]')
        plt.ylabel('amplitude')
        plt.legend()
        plt.savefig(figure_name_2)

    # ngspice_simulator.cleanup()
    # os.remove(figure_name)
    # if show_state_plots:
    #     os.remove(figure_name_2)
    save_time_plot(u_hat, 'ota_dither_time_plot.png')

    assert est_ENOB >= ENOB


CALIBRATE_SIM = True
VALIDATE_SIM = True
CALIB_CREATE_FILTER = True
CALIBRATE_CALIB = True

K1 = 1 << 11

calib_iterations = 1 << 27
calib_batch_size = 1 << 7
calib_step_size = 1e-5

calibration_sim_size = 1 << 17
validation_sim_size = 1 << 15


@pytest.mark.skip(reason="to long simulation time")
def test_ngspice_ota_simulator_calibration():
    results_folder = 'ota_calib_results'
    os.makedirs(results_folder, exist_ok=True)

    training_data_name = os.path.join(results_folder, 'train.npy')
    testing_data_name = os.path.join(results_folder, 'test.npy')

    # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
    Gamma = np.hstack((np.zeros((N, 1)), analog_frontend.analog_system.Gamma))
    Gamma[0, 0] = analog_frontend.analog_system.Gamma[0, 0] / 10.0

    analog_frontend_modified = AnalogFrontend(
        AnalogSystem(
            analog_frontend.analog_system.A,
            analog_frontend.analog_system.B,
            analog_frontend.analog_system.CT,
            Gamma,
            analog_frontend.analog_system.Gamma_tildeT,
        ),
        DitherControl(1, analog_frontend.digital_control),
    )

    # vdd = 2.4
    DC_gain = np.inf

    frequency = 1.0 / analog_frontend_modified.digital_control.clock.T
    while frequency > max_input_frequency:
        frequency /= 2

    # Training sequence
    amplitude = 0.0

    testbench = OTATestBench(
        analog_frontend_modified,
        input_signals=[Sinusoidal(amplitude, frequency, phase, offset)],
        clock=analog_frontend_modified.digital_control.clock,
        vdd_voltage=vdd,
        DC_gain=DC_gain,
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend_modified.digital_control.clock.T,
        calibration_sim_size * analog_frontend_modified.digital_control.clock.T,
        netlist_filename='ota_calibrate_train.cir',
        raw_output_filename='ota_calibrate_train.raw',
    )

    if CALIBRATE_SIM:
        ngspice_simulator.save_control_vector(training_data_name)

    cleanup_filenames = []

    filter_name = os.path.join(results_folder, 'filter1.npy')
    temp_name = os.path.join(results_folder, 'temp.npy')
    cleanup_filenames.append(filter_name)
    fs = 1 / analog_frontend_modified.digital_control.clock.T
    BW_rel = 2 * BW / fs

    calib_filter = '/home/hammal/.local/bin/calib_filter'
    calib_visualize = '/home/hammal/.local/bin/calib_visualize'
    calib = '/home/hammal/.cargo/bin/calib'

    if CALIB_CREATE_FILTER:
        create_and_check_filter_commands = [
            # create filter
            [
                calib_filter,
                'create',
                '-bw',
                str(BW_rel),
                '-m',
                str(analog_frontend_modified.analog_system.M),
                '-k',
                str(K1),
                filter_name,
            ],
            # plot initial filter
            [calib_filter, 'plot', filter_name],
        ]
        print('Creating filter...')
        print(' '.join(create_and_check_filter_commands[0]))
        print(' '.join(create_and_check_filter_commands[1]))
        for cmd in create_and_check_filter_commands:
            subprocess.run(cmd)

        os.rename(
            'bode_plot.png', os.path.join(results_folder, 'initial_bode_plot.png')
        )
        os.rename(
            'impulse_response.png',
            os.path.join(results_folder, 'initial_impulse_response.png'),
        )

    if CALIBRATE_CALIB:
        calibrate_filter_commands = [
            [
                calib,
                'calibrate',
                '-i',
                training_data_name,
                '-f',
                filter_name,
                '-o',
                temp_name,
                '--iterations',
                str(calib_iterations),
                '--batch-size',
                str(calib_batch_size),
                '--step-size',
                str(calib_step_size),
            ],
            # plot filter
            [calib_filter, 'plot', filter_name],
            # visualize training error
            [calib_visualize, '-bw', str(BW_rel), temp_name],
        ]

        print('Calibrating filter...')
        print(' '.join(calibrate_filter_commands[0]))
        print(' '.join(calibrate_filter_commands[1]))
        print(' '.join(calibrate_filter_commands[2]))
        for cmd in calibrate_filter_commands:
            subprocess.run(cmd)

        os.rename(
            'bode_plot.png', os.path.join(results_folder, 'calibrated_bode_plot.png')
        )
        os.rename(
            'impulse_response.png',
            os.path.join(results_folder, 'calibrated_impulse_response.png'),
        )
        os.rename('time.png', os.path.join(results_folder, 'calibration_error.png'))

    # generate testing sequence

    amplitude = vdd * 0.9

    testbench = OTATestBench(
        analog_frontend_modified,
        input_signals=[Sinusoidal(amplitude, frequency, phase, offset)],
        clock=analog_frontend_modified.digital_control.clock,
        vdd_voltage=vdd,
        DC_gain=DC_gain,
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend_modified.digital_control.clock.T,
        validation_sim_size * analog_frontend_modified.digital_control.clock.T,
        netlist_filename='ota_calibrate_test.cir',
        raw_output_filename='ota_calibrate_test.raw',
    )

    if VALIDATE_SIM:
        ngspice_simulator.save_control_vector(testing_data_name)

    validate_filter_commands = [
        [
            calib,
            'validate',
            '-f',
            filter_name,
            '-i',
            testing_data_name,
            '-o',
            temp_name,
        ],
        # visualize testing performance
        [calib_visualize, '-bw', str(BW_rel), temp_name],
    ]

    print('Validating filter...')
    for cmd in validate_filter_commands:
        subprocess.run(cmd)

    os.rename('time.png', os.path.join(results_folder, 'time_est.png'))
    os.rename('psd.png', os.path.join(results_folder, 'psd_est.png'))

    u_hat = np.load(temp_name)

    u_hat_cut = u_hat[K1:]
    f, psd = compute_power_spectral_density(
        u_hat_cut[:],
        fs=1 / analog_frontend_modified.digital_control.clock.T,
        nperseg=u_hat_cut.size,
    )
    signal_index = find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = snr_spectrum_computation_extended(
        psd,
        signal_index,
        noise_index,
        fs=1 / analog_frontend_modified.digital_control.clock.T,
    )
    est_SNR = snr_to_dB(fom['snr'])
    est_ENOB = snr_to_enob(est_SNR)

    plt.figure()
    plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}",
    )
    plt.xlabel('Hz')
    plt.ylabel('V^2 / Hz dB')
    plt.legend()
    figure_name = os.path.join(results_folder, 'ota_calibrated_estimate.png')
    plt.savefig(figure_name)

    save_time_plot(u_hat, os.path.join(results_folder, 'ota_time_plot.png'))

    # os.remove(temp_name)

    # ngspice_simulator.cleanup()
    # subprocess.

    # calib_filter create -bw 0.1 -fn 1.0 -m 7 -k 512 filter1.npy
    # calib_filter check filter1.npy
    # calib_filter plot filter1.npy
    # calib calibrate -i train.npy -f filter1.npy -o train_est.npy --iterations 10000000 --batch-size 200 --step-size 1e-5
    #

    # calib_visualize -bw 0.1 train_est.npy

    # calib validate -f filter1.npy -i test.npy -o test_est.npy
    # calib_visualize -bw 0.1 test_est.npy

    assert est_ENOB >= ENOB


@pytest.mark.skip(reason="to long simulation time")
def test_ngspice_opamp_simulator_calibration():
    results_folder = 'opamp_calib_results'
    os.makedirs(results_folder, exist_ok=True)
    # calibration_size = 1 << 17
    # validation_size = 1 << 15

    training_data_name = os.path.join(results_folder, 'train.npy')
    testing_data_name = os.path.join(results_folder, 'test.npy')

    # analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
    Gamma = np.hstack((np.zeros((N, 1)), analog_frontend.analog_system.Gamma))
    Gamma[0, 0] = analog_frontend.analog_system.Gamma[0, 0] / 10.0

    analog_frontend_modified = AnalogFrontend(
        AnalogSystem(
            analog_frontend.analog_system.A,
            analog_frontend.analog_system.B,
            analog_frontend.analog_system.CT,
            Gamma,
            analog_frontend.analog_system.Gamma_tildeT,
        ),
        DitherControl(1, analog_frontend.digital_control),
    )

    # vdd = 2.4
    DC_gain = np.inf

    frequency = 1.0 / analog_frontend_modified.digital_control.clock.T
    while frequency > max_input_frequency:
        frequency /= 2

    # Training sequence
    amplitude = 0.0

    testbench = OTATestBench(
        analog_frontend_modified,
        input_signals=[Sinusoidal(amplitude, frequency, phase, offset)],
        clock=analog_frontend_modified.digital_control.clock,
        vdd_voltage=vdd,
        DC_gain=DC_gain,
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend_modified.digital_control.clock.T,
        calibration_sim_size * analog_frontend_modified.digital_control.clock.T,
        netlist_filename='opamp_calibrate_train.cir',
        raw_output_filename='opamp_calibrate_train.raw',
    )

    if CALIBRATE_SIM:
        ngspice_simulator.save_control_vector(training_data_name)

    cleanup_filenames = []

    filter_name = os.path.join(results_folder, 'filter1.npy')
    temp_name = os.path.join(results_folder, 'temp.npy')
    cleanup_filenames.append(filter_name)
    fs = 1 / analog_frontend_modified.digital_control.clock.T
    BW_rel = 2 * BW / fs
    # K1 = 1 << 11

    # iterations = 1 << 27
    # batch_size = 1 << 7
    # step_size = 1e-5

    calib_filter = '/home/hammal/.local/bin/calib_filter'
    calib_visualize = '/home/hammal/.local/bin/calib_visualize'
    calib = '/home/hammal/.cargo/bin/calib'

    if CALIB_CREATE_FILTER:
        create_and_check_filter_commands = [
            # create filter
            [
                calib_filter,
                'create',
                '-bw',
                str(BW_rel),
                '-m',
                str(analog_frontend_modified.analog_system.M),
                '-k',
                str(K1),
                filter_name,
            ],
            # plot initial filter
            [calib_filter, 'plot', filter_name],
        ]

        print('Creating filter...')
        print(' '.join(create_and_check_filter_commands[0]))
        print(' '.join(create_and_check_filter_commands[1]))
        for cmd in create_and_check_filter_commands:
            subprocess.run(cmd)

        os.rename(
            'bode_plot.png', os.path.join(results_folder, 'initial_bode_plot.png')
        )
        os.rename(
            'impulse_response.png',
            os.path.join(results_folder, 'initial_impulse_response.png'),
        )

    if CALIBRATE_CALIB:
        calibrate_filter_commands = [
            [
                calib,
                'calibrate',
                '-i',
                training_data_name,
                '-f',
                filter_name,
                '-o',
                temp_name,
                '--iterations',
                str(calib_iterations),
                '--batch-size',
                str(calib_batch_size),
                '--step-size',
                str(calib_step_size),
            ],
            # plot filter
            [calib_filter, 'plot', filter_name],
            # visualize training error
            [calib_visualize, '-bw', str(BW_rel), temp_name],
        ]

        print('Calibrating filter...')
        print(' '.join(calibrate_filter_commands[0]))
        print(' '.join(calibrate_filter_commands[1]))
        print(' '.join(calibrate_filter_commands[2]))
        for cmd in calibrate_filter_commands:
            subprocess.run(cmd)

        os.rename(
            'bode_plot.png', os.path.join(results_folder, 'calibrated_bode_plot.png')
        )
        os.rename(
            'impulse_response.png',
            os.path.join(results_folder, 'calibrated_impulse_response.png'),
        )
        os.rename('time.png', os.path.join(results_folder, 'calibration_error.png'))

    # generate testing sequence

    amplitude = vdd * 0.9

    testbench = OTATestBench(
        analog_frontend_modified,
        input_signals=[Sinusoidal(amplitude, frequency, phase, offset)],
        clock=analog_frontend_modified.digital_control.clock,
        vdd_voltage=vdd,
        DC_gain=DC_gain,
    )

    ngspice_simulator = NGSpiceSimulator(
        testbench,
        analog_frontend_modified.digital_control.clock.T,
        validation_sim_size * analog_frontend_modified.digital_control.clock.T,
        netlist_filename='ota_calibrate_test.cir',
        raw_output_filename='ota_calibrate_test.raw',
    )

    if VALIDATE_SIM:
        ngspice_simulator.save_control_vector(testing_data_name)

    validate_filter_commands = [
        [
            calib,
            'validate',
            '-f',
            filter_name,
            '-i',
            testing_data_name,
            '-o',
            temp_name,
        ],
        # visualize testing performance
        [calib_visualize, '-bw', str(BW_rel), temp_name],
    ]

    print('Validating filter...')
    for cmd in validate_filter_commands:
        subprocess.run(cmd)

    os.rename('time.png', os.path.join(results_folder, 'time_est.png'))
    os.rename('psd.png', os.path.join(results_folder, 'psd_est.png'))

    u_hat = np.load(temp_name)

    u_hat_cut = u_hat[K1:]
    f, psd = compute_power_spectral_density(
        u_hat_cut[:],
        fs=1 / analog_frontend_modified.digital_control.clock.T,
        nperseg=u_hat_cut.size,
    )
    signal_index = find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = snr_spectrum_computation_extended(
        psd,
        signal_index,
        noise_index,
        fs=1 / analog_frontend_modified.digital_control.clock.T,
    )
    est_SNR = snr_to_dB(fom['snr'])
    est_ENOB = snr_to_enob(est_SNR)

    plt.figure()
    plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}",
    )
    plt.xlabel('Hz')
    plt.ylabel('V^2 / Hz dB')
    plt.legend()
    figure_name = os.path.join(results_folder, 'opamp_calibrated_estimate.png')
    plt.savefig(figure_name)

    save_time_plot(u_hat, os.path.join(results_folder, 'opamp_time_plot.png'))

    # os.remove(temp_name)

    # ngspice_simulator.cleanup()
    # subprocess.

    # calib_filter create -bw 0.1 -fn 1.0 -m 7 -k 512 filter1.npy
    # calib_filter check filter1.npy
    # calib_filter plot filter1.npy
    # calib calibrate -i train.npy -f filter1.npy -o train_est.npy --iterations 10000000 --batch-size 200 --step-size 1e-5
    #

    # calib_visualize -bw 0.1 train_est.npy

    # calib validate -f filter1.npy -i test.npy -o test_est.npy
    # calib_visualize -bw 0.1 test_est.npy

    assert est_ENOB >= ENOB
