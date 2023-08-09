from typing import Iterator, Union
from .testbench import TestBench
from jinja2 import Environment, PackageLoader, select_autoescape
from ltspice import Ltspice
from pandas import read_csv

import numpy as np
import logging
import subprocess
import os
import time

logger = logging.getLogger(__name__)

_template_env = Environment(
    loader=PackageLoader("cbadc", package_path="circuit/templates"),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


class NGSpiceSimulator(Iterator[np.ndarray]):
    """Simulate a testbench using NGSpice

    Parameters
    ----------

    """

    step: float
    t_end: float
    control_vector: np.ndarray
    testbench: TestBench
    netlist_filename: str
    raw_output_filename: str
    raw_file: Ltspice
    stdout_filename: str
    stderr_filename: str
    simulation_time: float

    def __init__(
        self,
        testbench: TestBench,
        step: float,
        t_end: float,
        netlist_filename: str = 'ngspice_sim.cir',
        raw_output_filename: str = 'ngspice_sim.raw',
        stdout_filename: str = 'ngspice_sim.stdout',
        stderr_filename: str = 'ngspice_sim.stderr',
        ac_freq_range=(1e3, 1e7),
    ):
        self.testbench = testbench
        self.transient_analysis = {
            'start_time': 0.0,
            'stop_time': t_end,
            'step': step,
            'max_step': step / 10.0,
        }

        self._save_variable_names = []
        for terminal_list in testbench.highlighted_terminals.values():
            self._save_variable_names.extend(
                [terminal.name for terminal in terminal_list]
            )
        self._save_variable_names.extend(
            [
                f'Xaf.{term.name}'
                for term in self.testbench.Xaf.xp + self.testbench.Xaf.xn
            ]
        )

        self.ac_analysis = {
            'number_of_points': 1 << 8,
            'start_frequency': ac_freq_range[0],
            'stop_frequency': ac_freq_range[1],
        }

        self.options = {
            # 'method': 'gear',
            # 'itl3': 100,
            # 'itl4': 100,
            # 'itl5': 10000,
            # 'maxord': 6,
            # 'reltol': 1e-6,
            # 'abstol': 1e-18,
            'opts': None,
            'warn': 1,
        }

        self.control_vector = np.zeros(
            (
                int(
                    np.ceil(
                        t_end / testbench.Xaf.analog_frontend.digital_control.clock.T
                    )
                ),
                testbench.Xaf.analog_frontend.analog_system.M,
            )
        )

        self.testbench = testbench
        self.netlist_filename = netlist_filename
        self.raw_output_filename = raw_output_filename
        self.stderr_filename = stderr_filename
        self.stdout_filename = stdout_filename
        self.simulation_time = 0.0

        self._made_netlist = False
        self._ran = False
        self._parsed = False

    def run(self):
        """Run the simulation"""
        cmd = ['ngspice', '-b', '-r', self.raw_output_filename, self.netlist_filename]
        start_time = time.time()
        completed_process = subprocess.run(cmd, check=True, capture_output=True)
        end_time = time.time()
        self.simulation_time = time.strftime(
            "%H:%M:%S", time.gmtime(end_time - start_time)
        )
        logger.info(completed_process.args)
        logger.debug(completed_process.stdout)
        if completed_process.stderr:
            logger.error(completed_process.stderr)
        with open(self.stdout_filename, 'w') as f:
            f.write(completed_process.stdout.decode())
        with open(self.stderr_filename, 'w') as f:
            f.write(completed_process.stderr.decode())
        self._ran = True

    def get_input_signals(self):
        """Get the input signals from the simulation"""
        if 'input' not in self.testbench.highlighted_terminals:
            raise ValueError('No input signals to plot')
        time = self.raw_file.get_time()
        headers = ['t']
        input_signal_terminal_names = [
            term.name for term in self.testbench.highlighted_terminals['input']
        ]
        headers.extend(input_signal_terminal_names)
        u_hat = [
            np.array(self.raw_file.get_data(f'V({term_name})'))
            for term_name in input_signal_terminal_names
        ]
        return headers, np.vstack((time, *u_hat)).transpose()

    def get_state_trajectories(self):
        """Get the state trajectories from the simulation"""
        time = self.raw_file.get_time()

        headers = ['t']

        state_terminal_names = [
            f'XAF.{term.name}' for term in self.testbench.Xaf.xp + self.testbench.Xaf.xn
        ]

        headers.extend(state_terminal_names)

        data = [
            np.array(self.raw_file.get_data(f'V({term_name})'))
            for term_name in state_terminal_names
        ]

        return headers, np.vstack((time, *data)).transpose()

    # def get_state_observations(self):
    #     time = self.raw_file.get_time()
    #     data = [
    #         np.array(self.raw_file.get_data(f'V(XAF.STILDE{i})'))
    #         for i in range(self.control_vector.shape[1])
    #     ]
    #     u_hat = np.array(self.raw_file.get_data(f'V(IN0)'))
    #     return np.vstack((time, *data, u_hat)).transpose()

    def parse(self):
        """Parse the simulation results"""
        self.raw_file = Ltspice(self.raw_output_filename)
        self.raw_file.parse()
        # time = raw_file.get_time()
        clk = np.array(self.raw_file.get_data('V(CLK)'))

        if 'control' not in self.testbench.highlighted_terminals:
            raise ValueError('No control signals highlighted in testbench')

        control_signal_terminals = [
            term.name
            for term in self.testbench.highlighted_terminals['control'][
                : self.testbench.Xaf.analog_frontend.analog_system.M
            ]
        ]

        vcm = float(self.testbench.Vdd._parameters_dict['dc'])

        s_raw = [
            np.array(self.raw_file.get_data(f'V({control_signal_terminals[i]})'))
            / (vcm * 2.0)
            for i in range(self.control_vector.shape[1])
        ]
        trigger_enable = False
        index: int = 0
        float_to_ternary = np.vectorize(_float_to_ternary)
        for s in zip(clk, *s_raw):
            if not trigger_enable and s[0] <= vcm:
                trigger_enable = True
                if index < self.control_vector.shape[0]:
                    self.control_vector[index, :] = float_to_ternary(s[1:])
                    index += 1
            elif trigger_enable and s[0] > vcm:
                trigger_enable = False
        self._control_iterator = iter(self.control_vector)
        self._parsed = True

    def save_control_vector(self, filename):
        """Save the control vector to a numpy file"""
        self._lazy_initialize()
        np.save(filename, self.control_vector)

    def make_netlist(self):
        """Make the netlist for the simulation"""
        netlist = _template_env.get_template('ngspice/simulator.cir.j2').render(
            {
                'testbench': self.testbench.get_ngspice(),
                'transient_analysis': self.transient_analysis,
                'ac_analysis': self.ac_analysis,
                'state_variables': [
                    (f'Xaf.{x[0]}', f'Xaf.{x[1]}')
                    for x in zip(self.testbench.Xaf.xp, self.testbench.Xaf.xn)
                ],
                'save_variables': self._save_variable_names,
                'options': self.options,
            }
        )
        with open(self.netlist_filename, 'w') as f:
            f.write(netlist)

        self._made_netlist = True

    def __iter__(self):
        return self

    def _lazy_initialize(self):
        if not self._made_netlist or not self._ran or not self._parsed:
            # Make netlist
            self.make_netlist()
            # simulate with ngspice
            self.run()
            # parse simulation result
            self.parse()

    def __next__(self):
        self._lazy_initialize()
        return next(self._control_iterator)

    def cleanup(self):
        """Cleanup the simulation files"""
        os.remove(self.netlist_filename)
        os.remove(self.raw_output_filename)


class SpectreSimulator(NGSpiceSimulator):
    stop: float
    strobefreq: float
    strobedelay: float
    skipdc: bool
    cmin: float

    def __init__(
        self,
        testbench: TestBench,
        stop: float,
        strobefreq: float,
        strobedelay: float,
        skipdc: bool = False,
        cmin: float = 0.0,
        netlist_filename: str = 'spectre_sim.cir',
        raw_output_filename: str = 'spectre_sim.raw',
    ):
        self.stop = stop
        self.strobedelay = strobedelay
        self.strobefreq = strobefreq
        self.testbench = testbench
        self.skipdc = skipdc
        self.cmin = cmin
        self.netlist_filename = netlist_filename
        self.raw_output_filename = raw_output_filename

        # Make netlist
        self.make_netlist()
        # simulate with ngspice
        self.run()
        # parse simulation result
        self.parse()

    def parse(self):
        self.control_vector = read_csv(
            self.testbench.control_signal_vector_filename
        ).to_numpy()[:, : self.control_vector.shape[1]]
        self._control_iterator = iter(self.control_vector)

    def make_netlist(self):
        testbench_netlist, verilog_ams = self.testbench.get_spectre()
        netlist = _template_env.get_template('spectre/simulator.cir.j2').render(
            {
                'testbench': testbench_netlist,
                'cmin': self.cmin,
                'stop': self.stop,
                'strobedelay': self.strobedelay,
                'strobefreq': self.strobefreq,
                'skipdc': ['no', 'yes'][self.skipdc],
            }
        )
        with open(self.netlist_filename, 'w') as f:
            f.write(netlist)
        with open(self.testbench.verilog_ams_library_name, 'w') as f:
            f.write(verilog_ams)

    def run(self):
        cmd = [
            'spectre',
            self.netlist_filename,
            '-format psfascii',
            '-raw',
            self.raw_output_filename,
            '++aps',
            '-ahdllibdir',
            os.path.abspath(self.testbench.verilog_ams_library_name),
            '-log',
        ]
        completed_process = subprocess.run(cmd, check=True, capture_output=True)
        logger.info(completed_process.args)
        logger.debug(completed_process.stdout)
        if completed_process.stderr:
            logger.error(completed_process.stderr)

    def cleanup(self):
        super().cleanup()
        os.remove(self.testbench.control_signal_vector_filename)
        os.remove(self.testbench.verilog_ams_library_name)


def _float_to_ternary(x: float) -> float:
    if x > 2.0 / 3.0:
        return 1.0
    elif x < 1.0 / 3.0:
        return 0.0
    else:
        return 0.5
