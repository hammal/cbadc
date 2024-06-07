from typing import Dict
from .. import (
    SPICE_VALUE,
    Port,
    Terminal,
    CircuitElement,
    _template_env,
)
from ..models.vcvs import VoltageControlledVoltageSourceModel


class VoltageControlledVoltageSource(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        value: SPICE_VALUE,
        m: SPICE_VALUE = 1,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f"Expected str, got {type(instance_name)}")
        elif instance_name[0] != "E":
            instance_name = "E" + instance_name

        super().__init__(
            instance_name,
            [
                Terminal("VDD"),
                Terminal("VSS"),
                Terminal("IN_P"),
                Terminal("IN_N"),
                Terminal("OUT_P"),
                Terminal("OUT_N"),
            ],
            model_name,
            value=value,
            m=m,
        )
        self.model = VoltageControlledVoltageSourceModel(
            model_name,
            value=float(value),
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]) -> str:
        return _template_env.get_template("ngspice/vcvs.cir.j2").render(
            {
                "instance_name": self.instance_name,
                "terminals": self._get_terminal_names(connections),
                "value": self._parameters_dict["value"],
                "comments": self.comments,
            }
        )

    def get_spectre(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template("spectre/verilog_ams.cir.j2").render(
            {
                "instance_name": self.instance_name,
                "terminals": self._get_terminal_names(connections),
                "comments": self.comments,
                "model_instance_name": self.model.model_name,
            }
        )
