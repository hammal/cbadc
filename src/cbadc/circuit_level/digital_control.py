""""digital control implementations."""
from typing import List, Union
from .module import Module, Wire, Parameter, SubModules
from ..analog_signal.impulse_responses import StepResponse, RCImpulseResponse
from ..digital_control.digital_control import DigitalControl as IdealDigitalControl


class Comparator(Module):
    """A verilog-ams module representing a simple comparator

    Parameters
    ----------
    name: `str`
        the name of the instance to be used.
    """

    def __init__(self, name: str):
        instance_name = name
        self._vdd = Wire("vdd", True, False, True, comment="positive supply")
        self._gnd = Wire("vgd", True, False, True, comment="ground")
        self._sgd = Wire("vsgd", True, False, True, comment="signal ground")
        self._clk = Wire("clk", True, False, True, comment="clock signal")
        self.inputs = [
            self._vdd,
            self._gnd,
            self._sgd,
            self._clk,
            Wire("s_tilde", True, False, True),
        ]
        self.outputs = [Wire("s", False, True, True)]
        ports = [*self.inputs, *self.outputs]
        nets = [*ports]
        parameters = [
            Parameter('dly', 0.0),
            Parameter("ttime", "10p"),
        ]
        analog_statements = [
            "@(cross(V(clk) - V(sgd), -1)) begin",
            "\tif(V(s_tilde) > V(sgd))",
            "\t\tV(s, vgd) <+ V(vdd, vgd) * transition(1, dly, ttime);",
            "\telse",
            "\t\tV(s, vgd) <+ V(vdd, vgd) * transition(0, dly, ttime);",
            "end",
        ]
        analog_initial = ["V(s) = 0"]
        super().__init__(
            "comparator",
            nets,
            ports,
            instance_name=instance_name,
            parameters=parameters,
            analog_statements=analog_statements,
            analog_initial=analog_initial,
        )

    def _module_comment(self) -> List[str]:
        return [
            *super()._module_comment(),
            "",
            "Functional Description:",
            "",
            "A comparator implementation where",
            "the output signal s(t) is updated at the",
            "falling edge of the V(clk) signal depending",
            "on the input signal V(s_tilde) is above or",
            "below a given threshold.",
            "",
            "threshold determines the descision threshold.",
            "Furthermore, dly and ttime specifies how quickly the",
            "comparator can switch its output.",
        ]


class DigitalControl(Module):
    """A verilog-ams module representing a :py:class`cbadc.digital_control.DigitalControl`

    Specifically, with a step type impulse response

    Parameters
    ----------
    digital_control: :py:class`cbadc.digital_control.DigitalControl`
        the digital control from which the verilog-ams module will be constructed.

    Attributes
    ----------
    digital_control: :py:class`cbadc.digital_control.DigitalControl`
        the original digital control.
    """

    digital_control: IdealDigitalControl

    def __init__(self, digital_control: IdealDigitalControl) -> None:
        for imp in digital_control._impulse_response:
            if not isinstance(
                imp,
                StepResponse,
            ):
                raise Exception("This digital control only works for step responses.")
        self.digital_control = digital_control
        self._vdd = Wire("vdd", True, False, True, comment="positive supply")
        self._gnd = Wire("vgd", True, False, True, comment="ground")
        self._sgd = Wire("vsgd", True, False, True, comment="signal ground")
        self._clk = Wire("clk", True, False, True, comment="clock signal")
        self.inputs = [
            self._vdd,
            self._gnd,
            self._sgd,
            self._clk,
            *[
                Wire(f"s_tilde_{m}", True, False, True)
                for m in range(digital_control.M_tilde)
            ],
        ]
        self.outputs = [
            Wire(f"s_{m}", False, True, True) for m in range(digital_control.M)
        ]
        ports = [*self.inputs, *self.outputs]
        nets = [*ports]

        submodules = [
            SubModules(
                Comparator(f"q_{m}"),
                [
                    self._vdd,
                    self._gnd,
                    self._sgd,
                    self._clk,
                    self.inputs[m + 4],
                    self.outputs[m],
                ],
            )
            for m in range(digital_control.M)
        ]
        super().__init__("digital_control", nets, ports, submodules=submodules)

    def _module_comment(self) -> List[str]:
        return [
            *super()._module_comment(),
            "",
            "Functional Description:",
            "",
            "A digital control which mainly connects",
            "M comparators to the input and outputs of",
            "the module itself.",
        ]


class Switch(Module):
    def __init__(self, name: str):
        inputs = [Wire("s", True, False, True)]
        inouts = [
            Wire("p", True, True, True),
            Wire("n", True, True, True),
        ]
        ports = [*inputs, *inouts]
        nets = ports
        parameters = [
            Parameter("threshold", 0, True),
            Parameter("Gs", 1e0, True),
            Parameter("dly", 0, True),
            Parameter("ttime", 1e-12, True),
        ]
        analog_statements = [
            "if (V(s) > threshold)",
            "\tI(p, n) <+ Gs * V(p, n);",
            "else",
            "\tI(p, n) <+ transition(0, dly, ttime);",
            "end",
        ]
        super().__init__(
            "switch",
            nets,
            ports,
            name,
            parameters=parameters,
            analog_statements=analog_statements,
        )


class SwitchDAC(Module):
    def __init__(self, name: str, C: float):
        pass


class SwitchCapacitorControl(Module):
    """A verilog-ams module representing a :py:class`cbadc.digital_control.DigitalControl`

    Specifically, with a RC-type impulse response


    Parameters
    ----------
    digital_control: :py:class`cbadc.digital_control.DigitalControl`
        the digital control from which the verilog-ams module will be constructed.

    Attributes
    ----------
    digital_control: :py:class`cbadc.digital_control.DigitalControl`
        the original digital control.
    """

    digital_control: IdealDigitalControl

    def __init__(self, digital_control: IdealDigitalControl, C: float) -> None:
        for imp in digital_control._impulse_response:
            if not isinstance(
                imp,
                RCImpulseResponse,
            ):
                raise Exception("This digital control only works for step responses.")
        self.digital_control = digital_control
        self._vdd = Wire("vdd", True, False, True, comment="positive supply")
        self._gnd = Wire("vgd", True, False, True, comment="ground")
        self._sgd = Wire("vsgd", True, False, True, comment="signal ground")
        self._clk = Wire("clk", True, False, True, comment="clock signal")

        self.inputs = [
            self._vdd,
            self._gnd,
            self._sgd,
            self._clk,
            *[
                Wire(f"s_tilde_{m}", True, False, True)
                for m in range(digital_control.M_tilde)
            ],
        ]
        self.outputs = [
            Wire(f"s_{m}", False, True, True) for m in range(digital_control.M)
        ]
        ports = [*self.inputs, *self.outputs]
        nets = [*ports]

        submodules = [
            SubModules(
                Comparator(f"q_{m}"),
                [
                    self._vdd,
                    self._gnd,
                    self._sgd,
                    self._clk,
                    self.inputs[m + 4],
                    self.outputs[m],
                ],
            )
            for m in range(digital_control.M)
        ]
        super().__init__("digital_control", nets, ports, submodules=submodules)
