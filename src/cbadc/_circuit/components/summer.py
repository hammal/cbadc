from typing import Dict, List
from .. import Port, Terminal, _template_env, CircuitElement
from . import ngspice_vector_terminal_vector_scalar
from ..models.summer import SummerModel


class Summer(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        number_of_inputs: int,
        input_offset: List[float],
        input_gain: List[float],
        output_offset: float = 0.0,
        output_gain: float = 1.0,
        comments: List[str] = [],
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'A':
            instance_name = 'A' + instance_name

        super().__init__(
            instance_name,
            [Terminal() for _ in range(number_of_inputs)]
            + [
                Terminal(hidden=True),
                Terminal('VCM'),
            ],
        )
        self.model = SummerModel(
            model_name,
            input_offset=input_offset,
            input_gain=input_gain,
            output_offset=output_offset,
            output_gain=output_gain,
            comments=comments,
        )

    def _ngspice_get_terminal_names(
        self, connections: Dict[Terminal, Port]
    ) -> List[str]:
        named_nodes = self._get_terminal_names(connections)
        input_vector = " ".join(
            [f"%vd({node},{named_nodes[-1]})" for node in named_nodes[:-2]]
        )
        return [f'[{input_vector}]', f'{named_nodes[-2]}']

    def get_ngspice(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('ngspice/xspice.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._ngspice_get_terminal_names(connections),
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


class DifferentialSummer(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        number_of_inputs: int,
        input_offset: List[float],
        input_gain: List[float],
        output_offset: float = 0.0,
        output_gain: float = 1.0,
        comments: List[str] = [],
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'A':
            instance_name = 'A' + instance_name

        if (
            len(input_offset) != len(input_gain)
            and len(input_offset) != number_of_inputs
        ):
            raise BaseException("Incorrect dimensions")

        terminals = [Terminal() for _ in range(2 * number_of_inputs)] + [
            Terminal(hidden=True),
        ]
        super().__init__(
            instance_name,
            terminals,
        )
        self._number_of_inputs = number_of_inputs
        self.model = SummerModel(
            model_name,
            input_offset=input_offset,
            input_gain=input_gain,
            output_offset=output_offset,
            output_gain=output_gain,
            comments=comments,
        )

    def _ngspice_get_terminal_names(
        self, connections: Dict[Terminal, Port]
    ) -> List[str]:
        named_nodes = self._get_terminal_names(connections)
        input_vector = " ".join(
            [
                f"%vd({a},{b})"
                for a, b in zip(
                    named_nodes[: self._number_of_inputs],
                    named_nodes[self._number_of_inputs : 2 * self._number_of_inputs],
                )
            ]
        )
        return [f'[{input_vector}]', f'{named_nodes[-1]}']

    def get_ngspice(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('ngspice/xspice.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._ngspice_get_terminal_names(connections),
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
