""""digital control implementations."""
from typing import List
from .module import Module, Variable, Wire, Parameter, SubModules
from ..analog_signal.impulse_responses import StepResponse, RCImpulseResponse
from ..digital_control.digital_control import DigitalControl as IdealDigitalControl
from ..digital_control.dither_control import DitherControl as IdealDitherControl


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
            Parameter("dly", 0.0),
            Parameter("ttime", "10p"),
        ]
        variables = [Variable("vout", real=True, comment="Output voltage value")]
        analog_statements = [
            "@(cross(V(clk) - V(vsgd), -1)) begin",
            "\tif(V(s_tilde) > V(vsgd)) begin",
            "\t\tvout=V(vdd, vgd);",
            "\tend",
            "\telse begin",
            "\t\tvout = V(vgd);",
            "\tend",
            "end",
            "V(s, vgd) <+ vout * transition(1.0, dly, ttime);",
        ]
        # analog_initial = ["V(s) = 0"]
        analog_initial = []
        super().__init__(
            "comparator",
            nets,
            ports,
            instance_name=instance_name,
            parameters=parameters,
            variables=variables,
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


class RandomControl(Module):
    """A verilog-ams module representing a random 1bit control

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
        ]
        self.outputs = [Wire("s", False, True, True)]
        ports = [*self.inputs, *self.outputs]
        nets = [*ports]
        parameters = [
            Parameter("dly", 0.0),
            Parameter("ttime", "10p"),
            Parameter("seed", 555),
        ]
        variables = [
            Variable("vout", real=True, comment="Output voltage value"),
            Variable("rval", real=True, comment="Random value"),
        ]
        analog_statements = [
            "@(cross(V(clk) - V(vsgd), -1)) begin",
            "\t// Generate random val",
            "\trval = $rdist_uniform(0, -1, 1);",
            "\tif (rval >= 0) begin",
            "\t\tvout=V(vdd, vgd);",
            "\tend",
            "\telse begin",
            "\t\tvout = V(vgd);",
            "\tend",
            "end",
            "V(s, vgd) <+ vout * transition(1.0, dly, ttime);",
        ]
        # analog_initial = ["V(s) = 0"]
        analog_initial = []
        super().__init__(
            "random_control",
            nets,
            ports,
            instance_name=instance_name,
            parameters=parameters,
            variables=variables,
            analog_statements=analog_statements,
            analog_initial=analog_initial,
        )

    def _module_comment(self) -> List[str]:
        return [
            *super()._module_comment(),
            "",
            "Functional Description:",
            "",
            "Random bit generator",
            # "the output signal s(t) is updated at the",
            # "falling edge of the V(clk) signal depending",
            # "on the input signal V(s_tilde) is above or",
            # "below a given threshold.",
            # "",
            # "threshold determines the descision threshold.",
            # "Furthermore, dly and ttime specifies how quickly the",
            # "comparator can switch its output.",
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
        self.number_of_random_control = 0
        if isinstance(digital_control, IdealDitherControl):
            self.number_of_random_control = digital_control.number_of_random_control

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

        # Instantiate dither control
        submodules = [
            SubModules(
                RandomControl(f"q_{m}"),
                [
                    self._vdd,
                    self._gnd,
                    self._sgd,
                    self._clk,
                    self.outputs[m],
                ],
            )
            for m in range(self.number_of_random_control)
        ]

        # Instantiate Comparators
        submodules += [
            SubModules(
                Comparator(f"q_{m}"),
                [
                    self._vdd,
                    self._gnd,
                    self._sgd,
                    self._clk,
                    self.inputs[m + 4 - self.number_of_random_control],
                    self.outputs[m],
                ],
            )
            for m in range(self.number_of_random_control, digital_control.M)
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
