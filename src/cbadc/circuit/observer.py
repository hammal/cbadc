from typing import List
from .module import Module, Wire, Variable
from ..digital_control.digital_control import DigitalControl as IdealDigitalControl


class Observer(Module):
    """A verilog-ams module for capturing analog signals"""

    digital_control: IdealDigitalControl

    def __init__(self, inputs: List[Wire], filename: str = 'observations.csv') -> None:
        name = 'observer'
        self._vdd = Wire("vdd", True, False, True, comment="positive supply")
        self._gnd = Wire("vgd", True, False, True, comment="ground")
        self._sgd = Wire("vsgd", True, False, True, comment="signal ground")
        self._clk = Wire("clk", True, False, True, comment="clock signal")

        self._filename = filename

        self.inputs: List[Wire] = [
            self._vdd,
            self._gnd,
            self._sgd,
            self._clk,
            *[
                Wire(x.name, True, False, x.electrical, comment=x.comment)
                for x in inputs
            ],
        ]
        variables = [Variable('fp', real=False, comment='For observer')]

        self.outputs = []
        ports = [*self.inputs, *self.outputs]
        nets = [*ports]

        csv_header = ','.join([x.name for x in self.inputs[4:]])

        def print_format(value: Wire) -> str:
            if value.name == "clk":
                return "%t"
            else:
                return "%f"

        csv_format_type = ','.join([x for x in map(print_format, self.inputs[4:])])
        csv_value = ', '.join([f"V({x.name})" for x in self.inputs[4:]])
        analog_statements = [
            "@(initial_step) begin",
            f'\tfp=$fopen("{self._filename}","w");',
            f'\t$fwrite(fp,"{csv_header}\\n");',
            "end",
            "",
            "@(final_step) begin",
            "\t$fclose(fp);",
            "end",
            "",
            "@(cross(V(clk) - V(vsgd), 1)) begin",
            f'\t$fstrobe(fp, "{csv_format_type}", {csv_value});',
            "end",
        ]

        super().__init__(
            name,
            nets,
            ports,
            instance_name=name,
            analog_statements=analog_statements,
            # variables=variables
        )

    def _module_comment(self) -> List[str]:
        return [
            *super()._module_comment(),
            "",
            "Functional Description:",
            "",
            "A observer module for capturing signals to file.",
        ]
