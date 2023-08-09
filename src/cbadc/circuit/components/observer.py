from typing import Dict, List
from .. import (
    Port,
    Terminal,
    _template_env,
    CircuitElement,
    SPICE_VALUE,
)
from ..models.observer import ObserverModel


class Observer(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        input_signal_names: List[str],
        trigger_offset: SPICE_VALUE = 0.5,
        save_on_falling_edge: bool = True,
        comments: List[str] = [],
        filename: str = 'observations.csv',
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'A':
            instance_name = 'A' + instance_name

        super().__init__(
            instance_name,
            [Terminal() for _ in range(len(input_signal_names) + 1)],
        )
        self.model = ObserverModel(
            model_name,
            input_signal_names=input_signal_names,
            trigger_offset=trigger_offset,
            save_on_falling_edge=save_on_falling_edge,
            comments=comments,
            filename=filename,
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

    def get_spectre(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('spectre/verilog_ams.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'comments': self.comments,
                'model_instance_name': self.model.model_name,
            }
        )
