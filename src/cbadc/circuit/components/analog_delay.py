from typing import Dict, List, Set
from .. import (
    CircuitElement,
    SPICE_VALUE,
    DeviceModel,
    Port,
    Terminal,
    SubCircuitElement,
    _template_env,
)
from ..models.analog_delay import (
    AnalogDelayModel,
)
from . import ngspice_vector_terminal_vector_vector


class AnalogDelay(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        delay: SPICE_VALUE,
        buffer_size: int = 2048,
        comments: List[str] = [],
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'A':
            instance_name = 'A' + instance_name

        super().__init__(
            instance_name,
            [
                Terminal('IN'),
                Terminal('OUT'),
                Terminal('CNTRL'),
            ],
            comments=comments,
        )
        self.model = AnalogDelayModel(
            model_name,
            delay=delay,
            buffer_size=buffer_size,
            has_delay_cnt=False,
            comments=comments,
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('ngspice/xspice.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'parameters': self._parameters_dict,
                'comments': self.comments,
                'model_instance_name': self.model.model_name,
            }
        )
