from typing import Dict
from .. import Port, Terminal, _template_env, CircuitElement
from ..models.integrator import IntegratorModel


class Integrator(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        input_offset: float = 0.0,
        gain: float = 1.0,
        out_lower_limit: float = -10.0,
        out_upper_limit: float = 10.0,
        limit_range: float = 1e-6,
        out_initial_condition: float = 0.0,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'A':
            instance_name = 'A' + instance_name

        super().__init__(
            instance_name,
            [Terminal(), Terminal()],
        )
        self.model = IntegratorModel(
            model_name,
            in_offset=input_offset,
            gain=gain,
            out_lower_limit=out_lower_limit,
            out_upper_limit=out_upper_limit,
            limit_range=limit_range,
            out_ic=out_initial_condition,
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
