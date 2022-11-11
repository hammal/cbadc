import cbadc
import shlib
from pade.spectre import simulate_netlist
from inform import warn
from pade.utils import init_logger

logger = init_logger()

work_dir = shlib.to_path(__file__).parent
netlist_dir = shlib.to_path(work_dir, 'netlist')
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
        analog_frontend_target, [input_signal], save_all_variables=True, save_to_filename=shlib.to_path(work_dir, 'observations.csv')
    )
    testbench.to_file(netlist_filename, path=netlist_dir)

def test_testbench_sim_1():

    testbench = cbadc.circuit.get_testbench(
        analog_frontend_target, [input_signal], save_all_variables=True, save_to_filename=shlib.to_path(work_dir, 'observations.csv')
    )
    spectre_raw_data_dir = shlib.to_path(work_dir, 'spectre_raw_data')
    shlib.mkdir(spectre_raw_data_dir)
    spectre_log_file = shlib.to_path(work_dir, 'spectre_sim.log')
    testbench._run_spectre_simulation(netlist_filename, path=netlist_dir, raw_data_dir=spectre_raw_data_dir, log_file=spectre_log_file)


def test_testbench_sim_2():
    testbench = cbadc.circuit.get_opamp_testbench(
        analog_frontend_target, [input_signal], C=1e-12, save_all_variables=True, save_to_filename=shlib.to_path(work_dir, 'observations.csv')
    )
    spectre_raw_data_dir = shlib.to_path(work_dir, 'spectre_raw_data')
    shlib.mkdir(spectre_raw_data_dir)
    spectre_log_file = shlib.to_path(work_dir, 'spectre_sim.log')
    testbench._run_spectre_simulation(netlist_filename, path=netlist_dir, raw_data_dir=spectre_raw_data_dir, log_file=spectre_log_file)


def test_testbench_sim_3():
    testbench = cbadc.circuit.get_opamp_testbench(
        analog_frontend_target, [input_signal], C=1e-12, GBWP=1e3*BW, A_DC=1e3, save_all_variables=True, save_to_filename=shlib.to_path(work_dir, 'observations.csv')
    )
    spectre_raw_data_dir = shlib.to_path(work_dir, 'spectre_raw_data')
    shlib.mkdir(spectre_raw_data_dir)
    spectre_log_file = shlib.to_path(work_dir, 'spectre_sim.log')
    testbench._run_spectre_simulation(netlist_filename, path=netlist_dir, raw_data_dir=spectre_raw_data_dir, log_file=spectre_log_file)

