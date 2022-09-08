"""op-amp implementations."""
from typing import List
from cbadc.circuit_level.module import Module, Wire, Parameter
import logging


logger = logging.getLogger(__name__)


class IdealOpAmp(Module):
    """An ideal op-amp implementation

    Parameters
    ----------
    name: `str`
        the instance name of the module.
    """

    def __init__(self, name: str, **kwargs):
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
        analog_statements = [
            "V(out): V(p_in, n_in) == 0;",
        ]
        super().__init__(
            "ideal_op_amp",
            nets,
            ports,
            instance_name,
            analog_statements=analog_statements,
        )

    def _module_comment(self) -> List[str]:
        return [
            *super()._module_comment(),
            "",
            "Functional Description:",
            "",
            "Ideal op-amp implementation.",
        ]


# class FiniteGainOpAmp(Module):
#     """A finite gain op-amp implementation

#     Specifically, the op-amps transfer function is
#     determined as

#     :math:`V_{\\text{out}} = A_{\\text{DC}} * V_{\\text{in}}`

#     Note that the default parameter values for A_DC and omega_0 are
#     typically overwritten by a parent module by using the same name for the parameter
#     in the same parent module.

#     Parameters
#     ----------
#     name: `str`
#         the instance name of the module.
#     A_DC: `float`
#         the finite gain of the op-amp.
#     """

#     def __init__(self, name: str, A_DC: float, **kwargs):
#         instance_name = name
#         self._vdd = Wire("vdd", True, False, True, comment="positive supply")
#         self._gnd = Wire("vgd", True, False, True, comment="ground")
#         ports = [
#             self._vdd,
#             self._gnd,
#             Wire("p_in", True, False, True, comment="positive input"),
#             Wire("n_in", True, True, True, comment="negative input"),
#             Wire("out", False, True, True, comment="output"),
#         ]
#         nets = ports
#         parameters = [Parameter('A_DC', A_DC, True)]
#         analog_statements = [
#             "V(out) <+  A_DC * V(p_in, n_in);",
#         ]
#         super().__init__(
#             "finite_gain_op_amp",
#             nets,
#             ports,
#             instance_name,
#             analog_statements=analog_statements,
#             parameters=parameters,
#         )

#     def _module_comment(self) -> List[str]:
#         return [
#             *super()._module_comment(),
#             "",
#             "Functional Description:",
#             "",
#             "A finite gain op-amp implementation",
#             "where",
#             "V(out) = A_DC * (V(p_in) - V(n_in))",
#         ]


class FirstOrderPoleOpAmp(Module):
    """An op-amp implementation including the first order pole

    Specifically, the op-amps transfer function is
    determined as

    :math:`V_{\\text{out}}(s) = \\frac{A_{\\text{DC}}}{1 + \\frac{s}{\omega_{p}}} * V_{\\text{in}}`

    Note that the default parameter values for A_DC and omega_0 are
    typically overwritten by a parent module by using the same name for the parameter
    in the same parent module.

    Parameters
    ----------
    name: `str`
        the instance name of the module.
    A_DC: `float`, `optional`
        the DC gain of the op-amp, defaults to 1e6.
    omega_p: `float`, `optional`
        the cutoff angular frequency of the op-amp, defaults to 100kHz.
    """

    def __init__(self, name: str, A_DC: float = 1e6, omega_p: float = 1e5, **kwargs):
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
        parameters = [
            Parameter("A_DC", A_DC, True),
            Parameter("omega_p", omega_p, True),
        ]
        analog_statements = [
            "V(out) <+ A_DC * laplace_qp(V(p_in, n_in), , {-omega_p, 0});",
        ]
        super().__init__(
            "first_order_pole_op_amp",
            nets,
            ports,
            instance_name,
            parameters=parameters,
            analog_statements=analog_statements,
        )

    def _module_comment(self) -> List[str]:
        return [
            *super()._module_comment(),
            "",
            "Functional Description:",
            "",
            "Op-amp implementation including a",
            "first order pole.",
            "",
            "i.e.,",
            "",
            "ddt(V(out)) = A_DC * omega_p * (V(p_in) - V(n_in)) - omega_p * V(out)",
        ]
