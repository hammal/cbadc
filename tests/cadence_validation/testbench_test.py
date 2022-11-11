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


def test_full():
    testbench = cbadc.circuit.get_opamp_testbench(
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
    K1 = 1 << 9
    K2 = K1
    estimator = testbench.analog_frontend.get_estimator(
        cbadc.digital_estimator.FIRFilter, eta2, K1, K2
    )
    estimator(simulator)

    size = 1 << 12
    u_hat = np.zeros(size)
    for i in range(size):
        u_hat[i] = next(estimator)
    plt.plot(u_hat)
    plt.savefig('testbench_uhat.png')


if __name__ == '__main__':

    test_full()
