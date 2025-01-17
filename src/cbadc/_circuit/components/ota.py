from typing import Dict
from .. import (
    SPICE_VALUE,
    Port,
    Terminal,
    CircuitElement,
    SubCircuitElement,
    _template_env,
)
from ..models.ota import OTAModel


class OTA(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        gm: SPICE_VALUE,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'G':
            instance_name = 'G' + instance_name

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
            model_name,
            gm=gm,
        )
        self.model = OTAModel(
            model_name,
            gm=float(gm),
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]) -> str:
        return _template_env.get_template('ngspice/ota.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'value': self._parameters_dict['gm'],
                'comments': self.comments,
            }
        )

    def get_spectre(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('spectre/verilog_ams.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'comments': self.comments,
                'model_instance_name': self.model.model_name,
            }
        )
