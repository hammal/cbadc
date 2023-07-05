from typing import Dict, List, Tuple
from . import (
    _template_env,
    Ground,
    Terminal,
    SubCircuitElement,
)
from .components.sources import (
    DCVoltageSource,
    PulseVoltageSource,
    SinusoidalVoltageSource,
)
from .components.observer import Observer
from ..analog_signal import Clock, Sinusoidal
from ..__version__ import __version__

# from .state_space import StateSpaceFrontend
from datetime import datetime
from ..analog_frontend import AnalogFrontend
from .opamp import OpAmpFrontend
from .ota import GmCFrontend
from .lc_tank import LCFrontend
from .analog_frontend import CircuitAnalogFrontend
import logging

logger = logging.getLogger(__name__)


class TestBench(SubCircuitElement):
    """Testbench for a circuit.

    Parameters
    ----------
    input_signals : List[Sinusoidal]
        List of input signals.
    clock : Clock
        a simulation clock
    vdd_voltage : float
        Vdd voltage in volts. Vss is assumed to be 0V.
    number_of_control_signals : int
        Number of control signals.
    title : str, optional
        Title of the testbench, by default 'Testbench'
    control_signal_vector_name : str, optional
        Name of the control signal vector file, by default 'control_signals.out'
    verilog_ams_library_name : str, optional
        Name of the verilog ams library file, by default 'verilog_ams_library.vams'
    """

    title: str
    Xaf: CircuitAnalogFrontend
    input_signals: List[SinusoidalVoltageSource]
    highlighted_terminals: Dict[str, List[Terminal]]
    Vss: DCVoltageSource
    Vdd: DCVoltageSource
    Vclk: PulseVoltageSource
    verilog_ams_library_name: str
    Aobs: Observer

    def __init__(
        self,
        input_signals: List[Sinusoidal],
        clock: Clock,
        vdd_voltage: float,
        number_of_control_signals: int,
        *args,
        title='Testbench',
        control_signal_vector_name='control_signals.out',
        verilog_ams_library_name='verilog_ams_library.vams',
        **kwargs,
    ):
        self.title = title
        self.verilog_ams_library_name = verilog_ams_library_name
        super().__init__(
            'Xtb',
            'testbench',
            [
                Ground(),
                Terminal('VDD'),
                Terminal('CLK'),
                Terminal('VCM'),
            ]
            + [Terminal(f'IN{i}_P') for i in range(len(input_signals))]
            + [Terminal(f'IN{i}_N') for i in range(len(input_signals))]
            + [Terminal(f'OUT{i}_P') for i in range(number_of_control_signals)]
            + [Terminal(f'OUT{i}_N') for i in range(number_of_control_signals)],
            *args,
            **kwargs,
        )

        # Add power supplies
        self.Vdd = DCVoltageSource('Vdd', vdd_voltage / 2)
        self.Vss = DCVoltageSource('Vss', vdd_voltage / 2)

        # Connect power supplies to terminals
        self.connects(
            (self['VCM'], self.Vdd[1]),
            (self['VDD'], self.Vdd[0]),
            (self['0'], self.Vss[1]),
            (self['VCM'], self.Vss[0]),
        )

        # Add clock source
        self.T = clock.T
        self.Vclk = PulseVoltageSource(
            'Vclk',
            0.0,
            vdd_voltage,
            clock.T,
            clock.tt,
            clock.tt,
        )

        # Connect clock to terminals
        self.connects(
            (self['CLK'], self.Vclk[0]),
            (self['0'], self.Vclk[1]),
        )

        self.input_signals = []
        for l, input_signal in enumerate(input_signals):
            if not isinstance(input_signal, Sinusoidal):
                raise TypeError(f'Input signal {l} is not of type Sinusoidal')
            inp = SinusoidalVoltageSource(
                offset=input_signal.offset,
                amplitude=input_signal.amplitude / 2,
                frequency=input_signal.frequency,
                delay_time=0.0,
                phase=input_signal.phase,
                damping_factor=0.0,
                instance_name=f'Vin_p_{l}',
                ac_gain=vdd_voltage / 2.0,
            )
            inn = SinusoidalVoltageSource(
                offset=input_signal.offset,
                amplitude=input_signal.amplitude / 2,
                frequency=input_signal.frequency,
                delay_time=0.0,
                phase=input_signal.phase,
                damping_factor=0.0,
                instance_name=f'Vin_n_{l}',
                ac_gain=vdd_voltage / 2.0,
            )
            self.add(inp, inn)

            # Connect input signal to terminals
            self.connects(
                (self[f'IN{l}_P'], inp[0]),
                (self[f'IN{l}_N'], inn[1]),
                (self['VCM'], inp[1]),
                (self['VCM'], inn[0]),
            )

            self.input_signals.append(inp)
            self.input_signals.append(inn)

        # add observer
        self.Aobs = Observer(
            'Aobs',
            'observer',
            [f's_{index}' for index in range(number_of_control_signals)],
            trigger_offset=vdd_voltage / 2.0,
            filename=control_signal_vector_name,
        )
        self.connect(self['CLK'], self.Aobs[0])

        self.highlighted_terminals = {}
        # Highlight terminals for later plotting
        self.highlighted_terminals['control'] = [
            Terminal(f'OUT{i}_P') for i in range(number_of_control_signals)
        ] + [Terminal(f'OUT{i}_N') for i in range(number_of_control_signals)]

        self.highlighted_terminals['input'] = [
            Terminal(f'IN{i}_P') for i in range(len(input_signals))
        ] + [Terminal(f'IN{i}_N') for i in range(len(input_signals))]

    def _sanity_check(self):
        if self.Xaf is None:
            raise ValueError('Analog frontend is not set')
        for terminal in self.Aobs.get_terminals():
            if terminal not in self._internal_connections:
                raise ValueError(
                    f'Testbench is not connected to observer terminal {terminal.name}'
                )
        # Check analog frontend
        # This makes me mad.
        # self.Xaf.check_connections()
        self.Xaf.check_subckt_names()

    def get_ngspice(self, check=True) -> str:
        """returns the ngspice testbench as a string"""
        if check:
            self._sanity_check()
        return _template_env.get_template('ngspice/testbench.cir.j2').render(
            {
                'analog_frontend': self.Xaf,
                'input_signals': self.input_signals,
                'power_supplies': (self.Vdd, self.Vss),
                'clock': self.Vclk,
                'connections': self._internal_connections,
                'models': self._get_model_set(),
                'title': self.title,
                "datetime": datetime.isoformat(datetime.now()),
                "cbadc_version": __version__,
                "comment_symbol": "*",
            }
        )

    def get_spectre(self) -> Tuple[str, str]:
        """return the spectre testbench as a string"""
        raise NotImplementedError()
        self._sanity_check()
        models = self._get_model_set()
        spice_models = []
        verilog_models = []
        if models:
            for model in models:
                if model.verilog_ams and model not in verilog_models:
                    verilog_models.append(model)
                elif not model.verilog_ams and model not in spice_models:
                    spice_models.append(model)
        # if verilog_models:
        verilog_ams_text = _template_env.get_template(
            'verilog_ams/library.vams.j2'
        ).render(
            {
                'models': self._get_model_set(),
                "datetime": datetime.isoformat(datetime.now()),
                "cbadc_version": __version__,
                "comment_symbol": "//",
            }
        )
        spice_text = _template_env.get_template('spectre/testbench.cir.j2').render(
            {
                'analog_frontend': self.Xaf,
                'input_signals': self.input_signals,
                'power_supplies': (self.Vdd, self.Vss),
                'clock': self.Vclk,
                'connections': self._internal_connections,
                'title': self.title,
                "datetime": datetime.isoformat(datetime.now()),
                "cbadc_version": __version__,
                'includes': [f'ahdl_include {self.verilog_ams_library_name}'],
                'models': spice_models,
                "comment_symbol": "*",
                "observer": self.observer,
            }
        )
        return (spice_text, verilog_ams_text)


class StateSpaceTestBench(TestBench):
    def __init__(
        self,
        analog_frontend: AnalogFrontend,
        input_signals: List[Sinusoidal],
        clock: Clock,
        GBWP: float,
        DC_gain: float,
        vdd_voltage: float,
        C_int: float = 1e-14,
        C_amp: float = 1e-14,
        title="CBADC OpAmpTestbench",
        control_signal_vector_name='control_signals.out',
        verilog_ams_library_name='verilog_ams_library.vams',
    ):
        super().__init__(
            input_signals,
            clock,
            vdd_voltage,
            number_of_control_signals=analog_frontend.analog_system.M,
            control_signal_vector_name=control_signal_vector_name,
            title=title,
            verilog_ams_library_name=verilog_ams_library_name,
        )

        # in_high = vdd_voltage / 2.0
        # in_low = vdd_voltage / 2.0

        # self.Xaf = StateSpaceFrontend(analog_frontend, vdd_voltage, in_high, in_low)
        # Connect gnd, power supply, and clock to analog frontend
        self.connects(
            (self['0'], self.Xaf['VSS']),
            (self['VDD'], self.Xaf['VDD']),
            (self['CLK'], self.Xaf['CLK']),
            (self['VCM'], self.Xaf['VCM']),
        )

        for l in range(self.Xaf.analog_frontend.analog_system.L):
            # Connect input signal to terminals
            self.connects(
                (self[f'IN{l}_P'], self.Xaf[f'IN{l}_P']),
                (self[f'IN{l}_N'], self.Xaf[f'IN{l}_N']),
            )

        # Connect analog frontend to observer
        for m in range(self.Xaf.analog_frontend.analog_system.M):
            self.connects(
                (self[f'OUT{m}_P'], self.Aobs[1 + m]),
                (self[f'OUT{m}_P'], self.Xaf[f'OUT{m}_P']),
                (self[f'OUT{m}_N'], self.Xaf[f'OUT{m}_N']),
            )


class OpAmpTestBench(TestBench):
    """The OpAmpTestBench class is a testbench for the OpAmp class.

    Parameters
    ----------
    analog_frontend : AnalogFrontend
        The analog frontend state space model.
    input_signals : List[Sinusoidal]
        The input signals to the analog frontend.
    clock : Clock
        The clock signal.
    GBWP : float
        the gain-bandwdith product of the opamp.
    DC_gain : float
        the DC gain of the opamp.
    vdd_voltage : float
        the supply voltage of the opamp. Vss is assumed to be 0.
    C_int : float, optional
        the internal integration capacitance of the opamp, by default 1e-14
    C_amp : float, optional
        the first order pole capacitance of the opamp, by default 1e-14
    title : str, optional
        the title of the testbench, by default "CBADC OpAmpTestbench"
    control_signal_vector_name : str, optional
        the name of the control signal vector, by default 'control_signals.out'
    verilog_ams_library_name : str, optional
        the name of the verilog ams library, by default 'verilog_ams_library.vams'
    """

    def __init__(
        self,
        analog_frontend: AnalogFrontend,
        input_signals: List[Sinusoidal],
        clock: Clock,
        GBWP: float,
        DC_gain: float,
        vdd_voltage: float,
        C_int: float = 1e-14,
        C_amp: float = 1e-14,
        title="CBADC OpAmpTestbench",
        control_signal_vector_name='control_signals.out',
        verilog_ams_library_name='verilog_ams_library.vams',
    ):
        super().__init__(
            input_signals,
            clock,
            vdd_voltage,
            number_of_control_signals=analog_frontend.analog_system.M,
            control_signal_vector_name=control_signal_vector_name,
            title=title,
            verilog_ams_library_name=verilog_ams_library_name,
        )

        # in_high = vdd_voltage / 2.0
        in_high = 0.0
        # in_low = vdd_voltage / 2.0
        in_low = 0.0
        self.Xaf = OpAmpFrontend(
            analog_frontend, GBWP, DC_gain, vdd_voltage, in_high, in_low, C_int, C_amp
        )
        # Connect gnd, power supply, and clock to analog frontend
        self.connects(
            (self['0'], self.Xaf['VSS']),
            (self['VDD'], self.Xaf['VDD']),
            (self['CLK'], self.Xaf['CLK']),
            (self['VCM'], self.Xaf['VCM']),
        )

        for l in range(self.Xaf.analog_frontend.analog_system.L):
            # Connect input signal to terminals
            self.connects(
                (self[f'IN{l}_P'], self.Xaf[f'IN{l}_P']),
                (self[f'IN{l}_N'], self.Xaf[f'IN{l}_N']),
            )

        # Connect analog frontend to observer
        for m in range(self.Xaf.analog_frontend.analog_system.M):
            self.connects(
                (self[f'OUT{m}_P'], self.Aobs[1 + m]),
                (self[f'OUT{m}_P'], self.Xaf[f'OUT{m}_P']),
                (self[f'OUT{m}_N'], self.Xaf[f'OUT{m}_N']),
            )


class OTATestBench(TestBench):
    """The OTATestBench class is a testbench for the OTA class.

    Parameters
    ----------
    analog_frontend : AnalogFrontend
        The analog frontend state space model.
    input_signals : List[Sinusoidal]
        The input signals to the analog frontend.
    clock : Clock
        The clock signal.
    DC_gain : float
        the DC gain of the opamp.
    vdd_voltage : float
        the supply voltage of the opamp. Vss is assumed to be 0.
    C_int : float, optional
        the internal integration capacitance of the opamp, by default 1e-14
    title : str, optional
        the title of the testbench, by default "CBADC OTA Testbench"
    control_signal_vector_name : str, optional
        the name of the control signal vector, by default 'control_signals.out'
    verilog_ams_library_name : str, optional
        the name of the verilog ams library, by default 'verilog_ams_library.vams'

    """

    def __init__(
        self,
        analog_frontend: AnalogFrontend,
        input_signals: List[Sinusoidal],
        clock: Clock,
        DC_gain: float,
        vdd_voltage: float,
        C_int: float = 1e-14,
        title="CBADC OTA Testbench",
        control_signal_vector_name='control_signals.out',
        verilog_ams_library_name='verilog_ams_library.vams',
    ):
        super().__init__(
            input_signals,
            clock,
            vdd_voltage,
            number_of_control_signals=analog_frontend.analog_system.M,
            control_signal_vector_name=control_signal_vector_name,
            title=title,
            verilog_ams_library_name=verilog_ams_library_name,
        )

        # in_high = vdd_voltage / 2.0
        in_high = 0.0
        # in_low = vdd_voltage / 2.0
        in_low = 0.0
        self.Xaf = GmCFrontend(
            analog_frontend, vdd_voltage, in_high, in_low, C_int, DC_gain
        )
        self.add(self.Xaf)

        # Connect gnd, power supply, and clock to analog frontend
        self.connects(
            (self['0'], self.Xaf['VSS']),
            (self['VDD'], self.Xaf['VDD']),
            (self['CLK'], self.Xaf['CLK']),
            (self['VCM'], self.Xaf['VCM']),
        )

        for l in range(self.Xaf.analog_frontend.analog_system.L):
            # Connect input signal to terminals
            self.connects(
                (self[f'IN{l}_P'], self.Xaf[f'IN{l}_P']),
                (self[f'IN{l}_N'], self.Xaf[f'IN{l}_N']),
            )

        # Connect analog frontend to observer
        for m in range(self.Xaf.analog_frontend.analog_system.M):
            self.connects(
                (self[f'OUT{m}_P'], self.Aobs[1 + m]),
                (self[f'OUT{m}_P'], self.Xaf[f'OUT{m}_P']),
                (self[f'OUT{m}_N'], self.Xaf[f'OUT{m}_N']),
            )


class LCTestBench(TestBench):
    def __init__(
        self,
        input_signals: List[Sinusoidal],
        M: int,
        omega_p: float,
        clock: Clock,
        vdd_voltage: float,
        C: float = 10e-12,
        gm: float = 1e-3,
        Rin: float = 1e0,
        title="CBADC LC Testbench",
        control_signal_vector_name='control_signals.out',
        verilog_ams_library_name='verilog_ams_library.vams',
    ):
        super().__init__(
            input_signals,
            clock,
            vdd_voltage,
            number_of_control_signals=M,
            control_signal_vector_name=control_signal_vector_name,
            title=title,
            verilog_ams_library_name=verilog_ams_library_name,
        )
        in_high = 0.0
        in_low = 0.0

        L = 2 / (omega_p**2 * C)

        self.Xaf = LCFrontend(
            M, L, C, gm, 1 / clock.T, Rin, vdd_voltage, in_high, in_low
        )

        self.add(self.Xaf)

        # Connect gnd, power supply, and clock to analog frontend
        self.connects(
            (self['0'], self.Xaf['VSS']),
            (self['VDD'], self.Xaf['VDD']),
            (self['CLK'], self.Xaf['CLK']),
            (self['VCM'], self.Xaf['VCM']),
        )

        for l in range(self.Xaf.analog_frontend.analog_system.L):
            # Connect input signal to terminals
            self.connects(
                (self[f'IN{l}_P'], self.Xaf[f'IN{l}_P']),
                (self[f'IN{l}_N'], self.Xaf[f'IN{l}_N']),
            )

        # Connect analog frontend to observer
        for m in range(self.Xaf.analog_frontend.analog_system.M):
            self.connects(
                (self[f'OUT{m}_P'], self.Aobs[1 + m]),
                (self[f'OUT{m}_P'], self.Xaf[f'OUT{m}_P']),
                (self[f'OUT{m}_N'], self.Xaf[f'OUT{m}_N']),
            )
