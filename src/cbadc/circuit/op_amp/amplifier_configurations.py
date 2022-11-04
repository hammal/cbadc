"""op-amp configurations"""
from typing import List, Union, Type
from cbadc.circuit.module import Module, Parameter, Wire, SubModules
from cbadc.circuit.state_space_equations import StateSpaceLinearSystem
from cbadc.circuit.op_amp.op_amp import FirstOrderPoleOpAmp, IdealOpAmp
import logging


logger = logging.getLogger(__name__)


class InvertingAmplifierCapacitiveFeedback(Module):
    def __init__(
        self,
        name: str,
        C: float,
        OpAmp: Union[
            Type[IdealOpAmp],
            # Type[FiniteGainOpAmp],
            Type[FirstOrderPoleOpAmp],
            Type[StateSpaceLinearSystem],
        ],
        **kwargs,
    ):
        instance_name = name
        self._vdd = Wire("vdd", True, False, True, comment="positive supply")
        self._gnd = Wire("vgd", True, False, True, comment="ground")
        ports = [
            self._vdd,
            self._gnd,
            Wire("p_in", True, False, True, comment="positive input"),
            Wire("n_in", True, True, True, comment="negative input"),
            Wire("out", False, True, True, comment="output"),
        ]
        nets = ports
        parameters = [Parameter("C", C, True)]
        analog_statements = [
            "I(out, n_in) <+ ddt(V(out, n_in)) * C;",
        ]
        unique_op_amp_name = f"op_amp_{name}"
        op_amp = OpAmp(unique_op_amp_name, instance_name=unique_op_amp_name, **kwargs)
        submodules = [SubModules(op_amp, [*ports])]
        super().__init__(
            f"inverting_amplifier_{name}",
            nets,
            ports,
            instance_name,
            analog_statements=analog_statements,
            parameters=parameters,
            submodules=submodules,
        )

    def _module_comment(self) -> List[str]:
        return [
            *super()._module_comment(),
            "",
            "Functional Description:",
            "",
            "Op-amp integrator configuration where",
            "a capacitor is connected as negative feedback",
            "i.e., between the output and negative input",
            "of the op-amp.",
            "",
            "The resulting differential equations are",
            "C ddt(V(out, n_in)) = I(out, n_in)",
        ]
