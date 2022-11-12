import cbadc
import shlib
from pade.utils import init_logger
import numpy as np
import matplotlib.pyplot as plt

logger = init_logger()

work_dir = shlib.to_path(__file__).parent
netlist_dir = shlib.to_path(work_dir, 'netlist')
observation_filename = shlib.to_path(work_dir, 'observations.csv')
spectre_raw_data_dir = shlib.to_path(work_dir, 'spectre_raw_data')
spectre_log_file = shlib.to_path(work_dir, 'spectre_sim.log')
shlib.mkdir(spectre_raw_data_dir)
shlib.mkdir(netlist_dir)

RERUN_SIM = 1

ENOB = 12
N = 5
BW = 1e6

SIM_NAME = f'LF_N{N}_B{ENOB}'
netlist_filename = f"{SIM_NAME}.scs"

analog_frontend_target = cbadc.synthesis.get_leap_frog(ENOB=ENOB, N=N, BW=BW)
amplitude = 1.0
frequency = 1.0 / analog_frontend_target.digital_control.clock.T
while frequency > BW:
    frequency /= 2
input_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency)


def test_testbench_to_file():
    testbench = cbadc.circuit.get_testbench(
        analog_frontend_target,
        [input_signal],
        save_all_variables=True,
        save_to_filename=observation_filename,
    )
    testbench.to_file(netlist_filename, path=netlist_dir)


def test_testbench_sim_1():

    testbench = cbadc.circuit.get_testbench(
        analog_frontend_target,
        [input_signal],
        save_all_variables=True,
        save_to_filename=observation_filename,
    )
    testbench.run_spectre_simulation(
        netlist_filename,
        path=netlist_dir,
        raw_data_dir=spectre_raw_data_dir,
        log_file=spectre_log_file,
    )


def test_testbench_sim_2():
    testbench = cbadc.circuit.get_opamp_testbench(
        analog_frontend_target,
        [input_signal],
        C=1e-12,
        save_all_variables=True,
        save_to_filename=observation_filename,
    )
    testbench.run_spectre_simulation(
        netlist_filename,
        path=netlist_dir,
        raw_data_dir=spectre_raw_data_dir,
        log_file=spectre_log_file,
    )


def test_testbench_sim_3():
    testbench = cbadc.circuit.get_opamp_testbench(
        analog_frontend_target,
        [input_signal],
        C=1e-12,
        GBWP=1e3 * BW,
        A_DC=1e3,
        save_all_variables=True,
        save_to_filename=observation_filename,
    )
    testbench.run_spectre_simulation(
        netlist_filename,
        path=netlist_dir,
        raw_data_dir=spectre_raw_data_dir,
        log_file=spectre_log_file,
    )


def test_get_spectre_simulator_default():
    testbench = cbadc.circuit.get_opamp_testbench(
        analog_frontend_target,
        [input_signal],
        C=1e-12,
        GBWP=1e3 * BW,
        A_DC=1e3,
        save_to_filename=observation_filename,
    )
    simulator = testbench.get_spectre_simulator(observation_filename)
    for i in range(10):
        print(next(simulator))


def test_get_spectre_simulator_all_variables():
    testbench = cbadc.circuit.get_opamp_testbench(
        analog_frontend_target,
        [input_signal],
        C=1e-12,
        GBWP=1e3 * BW,
        A_DC=1e3,
        save_all_variables=True,
        save_to_filename=observation_filename,
    )
    simulator = testbench.get_spectre_simulator(observation_filename)
    for i in range(10):
        print(next(simulator))


def test_with_estimator():
    testbench = cbadc.circuit.get_testbench(
        analog_frontend_target,
        [input_signal],
        save_all_variables=False,
        save_to_filename=observation_filename,
    )
    simulator = testbench.get_spectre_simulator(observation_filename)
    eta2 = testbench.analog_frontend.analog_system.analog_system.eta2(BW)
    K1 = 1 << 8
    K2 = K1
    estimator = testbench.analog_frontend.get_estimator(
        cbadc.digital_estimator.FIRFilter, eta2, K1, K2
    )
    estimator(simulator)

    size = 1 << 12
    u_hat = np.zeros(size)
    for i in range(size):
        u_hat[i] = next(estimator)
    u_hat_cut = u_hat[K1 + K2 :]
    fs = 1 / testbench.analog_frontend.digital_control.digital_control.clock.T
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_cut[:], fs=fs, nperseg=u_hat_cut.size
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=fs
    )
    est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
    est_ENOB = cbadc.fom.snr_to_enob(est_SNR)

    plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}",
    )
    plt.xlabel('Hz')
    plt.ylabel('V^2 / Hz dB')
    plt.legend()
    plt.savefig("test_with_estimator_psd.png")
    assert est_ENOB >= ENOB


def test_full():
    testbench = cbadc.circuit.get_testbench(
        analog_frontend_target,
        [input_signal],
        C=1e-12,
        GBWP=1e3 * BW,
        A_DC=1e3,
        save_all_variables=False,
        save_to_filename=observation_filename,
    )
    testbench.run_spectre_simulation(
        netlist_filename,
        path=netlist_dir,
        raw_data_dir=spectre_raw_data_dir,
        log_file=spectre_log_file,
    )
    simulator = testbench.get_spectre_simulator(observation_filename)
    eta2 = 1.0
    K1 = 1 << 8
    K2 = K1
    estimator = testbench.analog_frontend.get_estimator(
        cbadc.digital_estimator.FIRFilter, eta2, K1, K2
    )
    estimator(simulator)

    size = 1 << 12
    u_hat = np.zeros(size)
    for i in range(size):
        u_hat[i] = next(estimator)
    u_hat_cut = u_hat[K1 + K2 :]
    fs = 1 / testbench.analog_frontend.digital_control.digital_control.clock.T
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_cut[:], fs=fs, nperseg=u_hat_cut.size
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=fs
    )
    est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
    est_ENOB = cbadc.fom.snr_to_enob(est_SNR)

    plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}",
    )
    plt.xlabel('Hz')
    plt.ylabel('V^2 / Hz dB')
    plt.legend()
    plt.savefig("test_full_psd.png")
    assert est_ENOB >= ENOB


if __name__ == '__main__':

    test_full()
