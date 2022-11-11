"""testbench implementations"""
from typing import List
from jinja2 import Environment, PackageLoader, select_autoescape
from cbadc.circuit.analog_frontend import AnalogFrontend as CircuitAnalogFrontend
from cbadc.circuit.analog_system import (
    AnalogSystemFirstOrderPoleOpAmp,
    AnalogSystemIdealOpAmp,
)
from cbadc.circuit.digital_control import DigitalControl
from cbadc.circuit.state_space_equations import AnalogSystem
from cbadc.analog_signal import Clock, Sinusoidal
from cbadc.simulator import SimulatorType, get_simulator
from datetime import datetime
from cbadc.__version__ import __version__
from cbadc.analog_frontend import AnalogFrontend as AnalogFrontend
import os.path
import subprocess
import logging
logger = logging.getLogger(__name__)
import tqdm

_env = Environment(
    loader=PackageLoader("cbadc", package_path="circuit/templates"),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


class TestBench:
    """Initialize a Cadence type testbench

    Parameters
    ----------
    analog_frontend: :py:class:`cbadc.circuit_level.analog_frontend.AnalogFrontend`
        an analog frontend to be simulated.
    input_signal: :py:class:`cbadc.analog_signal.sinusoidal.Sinusoidal`
        an input signal.
    clock: :py:class:`cbadc.analog_signal.clock.Clock`
        a simulation clock that determines the sample times of the simulation.
    name: `str`
        name of the testbench
    vdd: `float`
        positive supply voltage [V].
    vgd: `float`
        negative supply and ground voltage [V].
    vsgd: `float`
        signal ground voltage, defaults to centered between vdd and vgd.
    number_of_samples: `int`
        number of samples to simulate in the testbench.
    """

    strobe_freq: float
    strobe_delay: float
    analog_frontend: CircuitAnalogFrontend

    def __init__(
        self,
        analog_frontend: CircuitAnalogFrontend,
        input_signal_list: List[Sinusoidal],
        clock: Clock,
        name: str = "",
        vdd: float = 1.0,
        vgd: float = 0.0,
        vsgd: float = None,
        number_of_samples: int = 1 << 12,
        tran_options={},
        sim_options={},
    ):
        for input_signal in input_signal_list:
            if not isinstance(input_signal, Sinusoidal):
                raise NotImplementedError("Currently only supported for sinusodials.")
        self.analog_frontend = analog_frontend
        self._input_signal_list = input_signal_list
        self._simulation_clock = clock
        self.strobe_freq = 1 / clock.T
        # quarter clock phase delay until readout
        self.strobe_delay = clock.T / 4.0
        self._t_stop = (number_of_samples + 1) * clock.T
        self._name = name
        if vdd < vgd:
            raise Exception("Must be postive supply")
        self._vdd = vdd
        self._vgd = vgd
        if vsgd:
            if vsgd > self._vdd or vsgd < self._vgd:
                raise Exception("Signal ground must be in between supply rails")
            self._vsgd = vsgd
        else:
            self._vsgd = (self._vdd - self._vgd) / 2 + self._vgd

        self.tran_options = tran_options
        self.sim_options = sim_options

    def render(self, verilog_path: str = ".") -> str:
        """Generate a rendered testbench file

        Parameters
        ----------
        path: `str`
            a directory path to the verilog module location.

        Returns
        -------
        : `str`
            a string containing the rendered file.s
        """
        path = os.path.abspath(verilog_path)
        template = _env.get_template('testbench_cadence.txt')
        return template.render(
            {
                'includes': [os.path.join(path, self.analog_frontend._filename)],
                'options': [],
                'vsgd': self._vsgd,
                'vgd': self._vgd,
                'vdd': self._vdd,
                'clock': {
                    'period': self.analog_frontend.digital_control.digital_control.clock.T,
                    'rise_time': self.analog_frontend.digital_control.digital_control.clock.tt,
                    'fall_time': self.analog_frontend.digital_control.digital_control.clock.tt,
                },
                'input_signals': [
                    {
                        'offset': self._input_signal_list[i].offset,
                        'amplitude': self._input_signal_list[i].amplitude,
                        'freq': self._input_signal_list[i].frequency,
                        'phase': self._input_signal_list[i].phase,
                    }
                    for i in range(len(self._input_signal_list))
                ],
                'analog_frontend': {
                    'inputs': [inp.name for inp in self.analog_frontend.inputs],
                    'outputs': [out.name for out in self.analog_frontend.outputs],
                    'name': self.analog_frontend.module_name,
                },
                't_stop': self._t_stop,
                'strobefreq': self.strobe_freq,
                'strobedelay': self.strobe_delay,
                'save_variables': [
                    [
                        v.name
                        for v in [
                            *self.analog_frontend.inputs,
                            *self.analog_frontend.outputs,
                            *self.analog_frontend.analog_system._x,
                            *self.analog_frontend.analog_system.inputs,
                            *self.analog_frontend.analog_system.outputs,
                        ]
                    ],
                    [
                        v.name
                        for v in [
                            *self.analog_frontend.outputs,
                        ]
                    ],
                ],
                'tran_options': self.tran_options,
                'sim_options': self.sim_options,
            }
        )

    def get_simulator(self, simulator_type: SimulatorType, **kwargs):
        """Return an instantiated simulator

        Return a python simulator of the specified testbench.

        Paramters
        ---------
        simulator_type: :py:class:`cbadc.simulator.SimulatorType`
            indicates which simulator backend to be used.

        Returns
        -------
        : :py:class:`cbadc.simulator.get_simulator`
            an instantied simulator.
        """
        return get_simulator(
            self.analog_frontend.analog_system.analog_system,
            self.analog_frontend.digital_control.digital_control,
            [self._input_signal],
            self._simulation_clock,
            # self._t_stop,
            simulator_type=simulator_type,
            **kwargs,
        )

    def _run_spectre_simulation(self, filename: str, path: str = ".", raw_data_dir=".", log_file="spectre_sim.log"):
        """Simulate using Spectre
        """
        self.to_file(filename, path)
        log_file = os.path.abspath(log_file)

        popen_cmd = [f'spectre {os.path.join(path, filename)} -format psfascii -raw {os.path.abspath(raw_data_dir)} ++aps -ahdllibdir {os.path.abspath(raw_data_dir)} -log']

        logger.info(f'Starting spectre simulation.\nCommand: {popen_cmd}')
        with open(log_file, 'wb') as f:
            process = subprocess.Popen(popen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            p0 = 0
            line_s = None
            for line in iter(process.stdout.readline, b''):
                f.write(line)
                line_s = line.decode('ascii')

            if line_s is None:
                RuntimeError('Spectre simulation did not return any output')
            try:
                status_list = line_s.split(' ')
                err_idx = int([i for i in range(0, len(status_list)) if "error" in status_list[i]][0])-1
                errors = int(status_list[err_idx])
            except:
                errors = True
            if errors:
                raise RuntimeError(f'Error occurred. See: {log_file}')

        logger.info("SPECTRE SIMULATION COMPLETE")
        logger.info(f"Raw data directory: {raw_data_dir}")


    def to_file(self, filename: str, path: str = "."):
        """Write the testbench to file

        Parameters
        ----------
        filename: `str`
            the filename
        path: `str`
            the intended path for testbench and verilog-ams files,
            defaults to cwd.
        """
        path = os.path.abspath(path)
        preamble = _env.get_template('preamble.txt').render(
            {
                "datetime": datetime.isoformat(datetime.now()),
                "cbadc_version": __version__,
                "verilog_ams": False,
            }
        )
        self.analog_frontend.to_file(filename=os.path.join(path, "analog_frontend"))
        res = "\n\n\n".join([preamble, self.render()])

        with open(os.path.join(path, filename), 'w') as f:
            f.write(res)


def get_testbench(
    analog_frontend: AnalogFrontend,
    input_signal_list: List[Sinusoidal],
    clock: Clock = None,
    name: str = "",
    vdd: float = 1.0,
    vgd: float = 0.0,
    vsgd: float = None,
    save_all_variables: bool = False,
    save_to_filename: str = "observations.csv",
):
    """Return an ideal state space model testbench for the specified analog frontend

    Parameters
    ----------
    analog_frontend: :py:class:`cbadc.analog_frontend.AnalogFrontend`
        the analog frontend to be tested.
    input_signal_list: `List`[`Sinusoidal`]
        a list of input signals to be applied to the analog frontend.
    clock: :py:class:`cbadc.digital_control.Clock`, optional
        the clock to be used for the simulation, defaults to the clock
    name: `str`
        the name of the testbench, defaults to empty string.
    vdd: `float`
        the positive supply voltage, defaults to 1.0.
    vgd: `float`
        the ground voltage, defaults to 0.0.
    vsgd: `float`
        the signal ground voltage, defaults to None.
    save_all_variables: `bool`
        if True, all variables are saved to file, if False, only control
        signals are saved to file, defaults to False.
    save_to_filename: `str`
        the filename to save the observations to, defaults to "observations.csv".

    Returns
    -------
    : :py:class:`cbadc.circuit.testbench.Testbench`
        an instantiated testbench.
    """
    circuit_analog_frontend = CircuitAnalogFrontend(
        AnalogSystem(
            analog_frontend.analog_system,
        ),
        DigitalControl(analog_frontend.digital_control),
        save_all_variables=save_all_variables,
        save_to_filename=save_to_filename,
    )
    if clock is None:
        simulation_clock = Clock(analog_frontend.digital_control.clock.T)
    else:
        simulation_clock = clock
    return TestBench(
        circuit_analog_frontend,
        input_signal_list,
        simulation_clock,
        name=name,
        vdd=vdd,
        vgd=vgd,
        vsgd=vsgd,
    )


def get_opamp_testbench(
    analog_frontend: AnalogFrontend,
    input_signal_list: List[Sinusoidal],
    C: float,
    GBWP: float = None,
    A_DC: float = None,
    omega_p: float = None,
    clock: Clock = None,
    name: str = "",
    vdd: float = 1.0,
    vgd: float = 0.0,
    vsgd: float = None,
    save_all_variables: bool = False,
    save_to_filename: str = "observations.csv",
):
    """Return an op-amp model testbench for the specified analog frontend

    Parameters
    ----------
    analog_frontend: :py:class:`cbadc.analog_frontend.AnalogFrontend`
        the analog frontend to be tested.
    input_signal_list: `List`[`Sinusoidal`]
        a list of input signals to be applied to the analog frontend.
    C: `float`
        the capacitance of the time constant of the op-amp.
    GBWP: `float`, optional
        the GBWP of the op-amp, defaults to None.
    A_DC: `float`, optional
        the DC gain of the op-amp, defaults to None.
    omega_p: `float`, optional
        the pole frequency of the op-amp, defaults to None.
    clock: :py:class:`cbadc.digital_control.Clock`, optional
        the clock to be used for the simulation, defaults to the clock
    name: `str`
        the name of the testbench, defaults to empty string.
    vdd: `float`
        the positive supply voltage, defaults to 1.0.
    vgd: `float`
        the ground voltage, defaults to 0.0.
    vsgd: `float`
        the signal ground voltage, defaults to None.
    save_all_variables: `bool`
        if True, all variables are saved to file, if False, only control
        signals are saved to file, defaults to False.
    save_to_filename: `str`
        the filename to save the observations to, defaults to "observations.csv".


    Returns
    -------
    : :py:class:`cbadc.circuit.testbench.Testbench`
        an instantiated testbench.
    """
    if GBWP is None and A_DC is None and omega_p is None:
        circuit_analog_system = AnalogSystemIdealOpAmp(
            analog_system=analog_frontend.analog_system,
            C=C,
        )
    else:
        if GBWP is None:
            circuit_analog_system = AnalogSystemFirstOrderPoleOpAmp(
                analog_system=analog_frontend.analog_system,
                C=C,
                A_DC=A_DC,
                omega_p=omega_p,
            )
        elif A_DC is None:
            circuit_analog_system = AnalogSystemFirstOrderPoleOpAmp(
                analog_system=analog_frontend.analog_system,
                C=C,
                GBWP=GBWP,
                omega_p=omega_p,
            )
        elif omega_p is None:
            circuit_analog_system = AnalogSystemFirstOrderPoleOpAmp(
                analog_system=analog_frontend.analog_system,
                C=C,
                GBWP=GBWP,
                A_DC=A_DC,
            )
        else:
            raise ValueError("Only two of GBWP, A_DC and omega_p can be specified.")
    circuit_analog_frontend = CircuitAnalogFrontend(
        circuit_analog_system,
        DigitalControl(analog_frontend.digital_control),
        save_all_variables=save_all_variables,
        save_to_filename=save_to_filename,
    )
    if clock is None:
        simulation_clock = Clock(analog_frontend.digital_control.clock.T)
    else:
        simulation_clock = clock
    return TestBench(
        circuit_analog_frontend,
        input_signal_list,
        simulation_clock,
        name=name,
        vdd=vdd,
        vgd=vgd,
        vsgd=vsgd,
    )
