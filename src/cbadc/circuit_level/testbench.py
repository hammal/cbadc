"""testbench implementations"""
from jinja2 import Environment, PackageLoader, select_autoescape
from .analog_frontend import AnalogFrontend
from ..analog_signal import Clock, Sinusoidal
from ..simulator import SimulatorType, get_simulator
from datetime import datetime
from ..__version__ import __version__
import os.path


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
    t_stop: `float`
        end time [s] of simulation, the simulation always starts at 0.
    name: `str`
        name of the testbench
    vdd: `float`
        positive supply voltage [V].
    vgd: `float`
        negative supply and ground voltage [V].
    vsgd: `float`
        signal ground voltage, defaults to centered between vdd and vgd.

    """

    env: Environment
    strobe_freq: float
    strobe_delay: float
    analog_frontend: AnalogFrontend

    def __init__(
        self,
        analog_frontend: AnalogFrontend,
        input_signal: Sinusoidal,
        clock: Clock,
        t_stop: float,
        name: str = "",
        vdd: float = 1.0,
        vgd: float = 0.0,
        vsgd: float = None,
    ):
        self.env = Environment(
            loader=PackageLoader("cbadc", package_path="circuit_level/templates"),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        if not isinstance(input_signal, Sinusoidal):
            raise NotImplementedError("Currently only supported for sinusodials.")
        self.analog_frontend = analog_frontend
        self._input_signal = input_signal
        self._simulation_clock = clock
        self.strobe_freq = 1 / clock.T
        # quarter clock phase delay until readout
        self.strobe_delay = clock.T / 4.0
        self._t_stop = t_stop
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
        template = self.env.get_template('testbench_cadence.txt')
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
                'input': {
                    'offset': self._input_signal.offset,
                    'amplitude': self._input_signal.amplitude,
                    'freq': self._input_signal.frequency,
                },
                'analog_frontend': {
                    'inputs': [inp.name for inp in self.analog_frontend.inputs],
                    'outputs': [out.name for out in self.analog_frontend.outputs],
                    'name': self.analog_frontend.module_name,
                },
                't_stop': self._t_stop,
                'strobefreq': self.strobe_freq,
                'strobedelay': self.strobe_delay,
                'save_variables': [
                    v.name
                    for v in [
                        *self.analog_frontend.inputs,
                        *self.analog_frontend.outputs,
                        *self.analog_frontend.analog_system._x,
                        *self.analog_frontend.analog_system.inputs,
                        *self.analog_frontend.analog_system.outputs,
                    ]
                ],
            }
        )

    def get_simulator(self, simulator_type: SimulatorType):
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
            self._t_stop,
            simulator_type=simulator_type,
        )

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
        preamble = self.env.get_template('preamble.txt').render(
            {
                "datetime": datetime.isoformat(datetime.now()),
                "cbadc_version": __version__,
                "verilog_ams": False,
            }
        )
        self.analog_frontend.to_file(filename=os.path.join(path, "analog_frontend"))
        res = "\n\n\n".join([preamble, self.render()])
        if not filename.endswith('.txt'):
            filename = f"{filename}.txt"
        with open(os.path.join(path, filename), 'w') as f:
            f.write(res)
