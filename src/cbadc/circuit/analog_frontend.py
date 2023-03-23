from typing import List
from . import (
    Terminal,
    SubCircuitElement,
)
from ..analog_frontend import AnalogFrontend
from ..digital_control import DigitalControl as NominalDigitalControl
from ..digital_control import DitherControl as NominalDitherControl
from .digital_control import DigitalControl, DitherControl
from ..analog_system import AnalogSystem


class CircuitAnalogFrontend(SubCircuitElement):
    analog_frontend: AnalogFrontend
    xp: List[Terminal]
    xn: List[Terminal]

    def __init__(
        self,
        analog_frontend: AnalogFrontend,
        vdd_voltage: float = 1.2,
        in_high=0.0,
        in_low=0.0,
        subckt_name: str = 'analog_frontend',
        instance_name: str = 'Xaf',
    ):
        self.analog_frontend = analog_frontend
        self.xp = [Terminal(f'X{i}_P') for i in range(analog_frontend.analog_system.N)]
        self.xn = [Terminal(f'X{i}_N') for i in range(analog_frontend.analog_system.N)]
        super().__init__(
            terminals=[
                Terminal('VSS'),
                Terminal('VDD'),
                Terminal('CLK'),
                Terminal('VCM'),
            ]
            + [Terminal(f'IN{i}_P') for i in range(analog_frontend.analog_system.L)]
            + [Terminal(f'IN{i}_N') for i in range(analog_frontend.analog_system.L)]
            + [Terminal(f'OUT{i}_P') for i in range(analog_frontend.analog_system.M)]
            + [Terminal(f'OUT{i}_N') for i in range(analog_frontend.analog_system.M)],
            subckt_name=subckt_name,
            instance_name=instance_name,
        )

        self._generate_digital_control(
            analog_frontend.analog_system,
            analog_frontend.digital_control,
            in_high,
            in_low,
            vdd_voltage,
            0.0,
        )

    def _generate_digital_control(
        self,
        analog_system: AnalogSystem,
        digital_control: NominalDigitalControl,
        in_high: float,
        in_low: float,
        out_high: float,
        out_low: float,
    ):
        if isinstance(digital_control, NominalDitherControl):
            self.Xdc = DitherControl(
                'dc',
                analog_system,
                digital_control,
                in_high,
                in_low,
                out_high,
                out_low,
            )
        else:
            self.Xdc = DigitalControl(
                'dc',
                analog_system,
                digital_control,
                in_high,
                in_low,
                out_high,
                out_low,
            )

        self.connects(
            (self['VSS'], self.Xdc['VSS']),
            (self['VDD'], self.Xdc['VDD']),
            (self['CLK'], self.Xdc['CLK']),
            (self['VCM'], self.Xdc['VCM']),
        )

        # Connect States
        for n in range(analog_system.N):
            self.connects(
                (self.xp[n], self.Xdc[f'X{n}_P']),
                (self.xn[n], self.Xdc[f'X{n}_N']),
            )

        # Connect inputs
        for l in range(analog_system.L):
            self.connects(
                (self[f'IN{l}_P'], self.Xdc[f'IN{l}_P']),
                (self[f'IN{l}_N'], self.Xdc[f'IN{l}_N']),
            )

        # Connect outputs (control signals)
        for m in range(analog_system.M):
            self.connects(
                (self[f'OUT{m}_P'], self.Xdc[f'S{m}_P']),
                (self[f'OUT{m}_N'], self.Xdc[f'S{m}_N']),
            )
