from typing import Dict, List, Set
from .. import (
    SPICE_VALUE,
    DeviceModel,
    Port,
    Terminal,
    SubCircuitElement,
    CircuitElement,
    _template_env,
)
from ..models.voltage_buffer import VoltageBufferModel


class VoltageBuffer(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        # common_mode_output_voltage: float,
        model_name: str,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'E':
            instance_name = 'E' + instance_name

        super().__init__(
            instance_name,
            [
                Terminal('VDD'),
                Terminal('VSS'),
                Terminal('IN_P'),
                Terminal('IN_N'),
                Terminal('OUT_P'),
                Terminal('OUT_N'),
            ],
            # common_mode_output_voltage = common_mode_output_voltage
        )
        self.model = VoltageBufferModel(
            model_name,
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]) -> str:
        return _template_env.get_template('ngspice/voltage_buffer.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'comments': self.comments,
            }
        )

    # def get_ngspice(self, connections: Dict[Terminal, Port]) -> str:
    #     return _template_env.get_template('ngspice/xspice.cir.j2').render(
    #         {
    #             'instance_name': self.instance_name,
    #             'terminals': self._get_terminal_names(connections),
    #             'parameters': self.parameters_dict,
    #             'comments': self.comments,
    #             'model_instance_name': self.model_instance_name,
    #         }
    #     )

    def get_spectre(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('spectre/verilog_ams.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'comments': self.comments,
                'model_instance_name': self.model.model_name,
            }
        )
