"""state space equations expressed in verilog-ams modules."""
from cbadc.circuit.module import Module, Wire
from cbadc.analog_system.analog_system import AnalogSystem
import numpy as np


def _signed_weight(x: float):
    magnitude = np.abs(x)
    if x >= 0:
        return f"+{magnitude}"
    else:
        return f"-{magnitude}"


class StateSpaceLinearSystem(Module):
    """State space linear system model


    Parameters
    ----------
    analog_system: :py:class`cbadc.analog_system.AnalogSystem`
        the analog system from which the verilog-ams module will be constructed.

    Attributes
    ----------
    analog_system: :py:class`cbadc.analog_system.AnalogSystem`
        the original analog system.
    """

    analog_system: AnalogSystem

    def __init__(
        self,
        module_name: str,
        amplifier: AnalogSystem,
        instance_name: str = "",
        **kwargs,
    ):
        if amplifier.L != 1:
            raise Exception("The amplifier system must be single input systems.")
        if amplifier.N_tilde != 1:
            raise Exception("The amplifiers must have precisely a single output")
        if (amplifier.Gamma is not None) or (amplifier.Gamma_tildeT is not None):
            raise Exception(
                "The amplifiers must be simple systems. No Gamma or Gamma_tildeTs allowed"
            )
        self.analog_system = amplifier
        self._vdd = Wire("vdd", True, False, True, comment="positive supply")
        self._gnd = Wire("vgd", True, False, True, comment="ground")
        self._sgd = Wire("vsgd", True, False, True, comment="signal ground")
        self._u = [
            Wire(f"u_{l}", True, False, True) for l in range(self.analog_system.L)
        ]
        self.inputs = [
            self._vdd,
            self._gnd,
            self._sgd,
            *self._u,
        ]
        self._x = [
            Wire(f"x_{n}", False, False, True, comment=f"internal_state {n}")
            for n in range(self.analog_system.N)
        ]
        self._y = [Wire("y", False, True, True, comment="Output")]
        self.outputs = [
            *self._y,
        ]
        ports = [*self.inputs, *self.outputs]
        nets = [*ports, *self._x]
        analog_statements = []
        # Differential Equations
        for n in range(self.analog_system.N):
            tmp = []
            for nn in range(self.analog_system.N):
                if self.analog_system.A[n, nn] != 0:
                    weight = self.analog_system.A[n, nn]
                    tmp.append(f"{_signed_weight(weight)}*V(x_{nn}, vsgd)")
            for l in range(self.analog_system.L):
                if self.analog_system.B[n, l] != 0:
                    weight = self.analog_system.B[n, l]
                    tmp.append(f"{_signed_weight(weight)}*V(u_{l}, vsgd)")
            if tmp:
                analog_statements.append(
                    f"V(x_{n}, vsgd) <+ idt(" + " ".join(tmp) + ");"
                )
        # OutputEquations
        for n_tilde in range(self.analog_system.N_tilde):
            tmp = []
            for n in range(self.analog_system.N):
                if self.analog_system.CT[n_tilde, n] != 0:
                    tmp.append(
                        f"{_signed_weight(self.analog_system.CT[n_tilde, n])}*V(x_{n}, vsgd)"
                    )
            if tmp:
                (
                    analog_statements.append(
                        f"V(y_{n_tilde}, vsgd) <+ " + " ".join(tmp) + ";"
                    )
                )
        super().__init__(
            module_name,
            nets,
            ports,
            analog_statements=analog_statements,
            instance_name=instance_name,
        )

    def _module_comment(self):
        return [
            *super()._module_comment(),
            "",
            "Functional Description",
            "",
            "A linear state space system directly modeled using differential",
            "equations.",
            "",
            "Specifically,",
            "",
            "ddt(x(t)) = A x(t) + B u(t)",
            "y(t) = C^T x(t)",
            "",
            "where",
            "",
            f"x(t) = [{', '.join([f'{v.name}' for v in self._x])}]^T",
            f"u(t) = [{', '.join([f'{v.name}' for v in self._u])}]^T",
            f"y(t) = [{', '.join([f'{v.name}' for v in self._y])}]^T",
            "",
            "A \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.A[i, :]])}]"
                for i in range(self.analog_system.N)
            ],
            "",
            "B \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.B[i, :]])}]"
                for i in range(self.analog_system.N)
            ],
            "",
            "",
            "CT \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.CT[i, :]])}]"
                for i in range(self.analog_system.N_tilde)
            ],
            "D \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.D[i, :]])}]"
                for i in range(self.analog_system.N_tilde)
            ],
        ]


class AnalogSystem(Module):
    """A verilog-ams module representing an :py:class`cbadc.analog_system.AnalogSystem`


    Parameters
    ----------
    analog_system: :py:class`cbadc.analog_system.AnalogSystem`
        the analog system from which the verilog-ams module will be constructed.

    Attributes
    ----------
    analog_system: :py:class`cbadc.analog_system.AnalogSystem`
        the original analog system.
    """

    analog_system: AnalogSystem

    def __init__(self, analog_system: AnalogSystem, **kwargs) -> None:
        if analog_system.Gamma is None or analog_system.Gamma_tildeT is None:
            raise Exception("both Gammas must be defined.")
        self.analog_system = analog_system
        self._u = [Wire(f"u_{l}", True, False, True) for l in range(analog_system.L)]
        self._s = [Wire(f"s_{m}", True, False, True) for m in range(analog_system.M)]
        self._vdd = Wire("vdd", True, False, True, comment="positive supply")
        self._gnd = Wire("vgd", True, False, True, comment="ground")
        self._sgd = Wire("vsgd", True, False, True, comment="signal ground")
        self.inputs = [
            self._vdd,
            self._gnd,
            self._sgd,
            *self._u,
            *self._s,
        ]
        self._x = [Wire(f"x_{n}", False, True, True) for n in range(analog_system.N)]
        self._s_tilde = [
            Wire(f"s_tilde_{m_tilde}", False, True, True)
            for m_tilde in range(analog_system.M_tilde)
        ]
        self.outputs = [
            *self._s_tilde,
        ]
        ports = [*self.inputs, *self.outputs]
        nets = [*ports, *self._x]
        analog_statements = []
        # Differential Equations
        for n in range(self.analog_system.N):
            tmp = []
            for nn in range(self.analog_system.N):
                if self.analog_system.A[n, nn] != 0:
                    tmp.append(
                        f"{_signed_weight(self.analog_system.A[n, nn])}*V(x_{nn}, vsgd)"
                    )
            for m in range(self.analog_system.M):
                if self.analog_system.Gamma[n, m] != 0:
                    tmp.append(
                        f"{_signed_weight(self.analog_system.Gamma[n,m])}*V(s_{m}, vsgd)"
                    )
            for l in range(self.analog_system.L):
                if self.analog_system.B[n, l] != 0:
                    tmp.append(
                        f"{_signed_weight(self.analog_system.B[n,l])}*V(u_{l}, vsgd)"
                    )
            if tmp:
                analog_statements.append(
                    f"V(x_{n}, vsgd) <+ idt(" + " ".join(tmp) + ");"
                )
        # OutputEquations
        for m_tilde in range(self.analog_system.M_tilde):
            tmp = []
            for n in range(self.analog_system.N):
                if self.analog_system.Gamma_tildeT[m_tilde, n] != 0:
                    tmp.append(
                        f"{_signed_weight(self.analog_system.Gamma_tildeT[m_tilde, n])}*V(x_{n}, vsgd)"
                    )
            if tmp:
                (
                    analog_statements.append(
                        f"V(s_tilde_{m_tilde}, vsgd) <+ " + " ".join(tmp) + ";"
                    )
                )
        super().__init__(
            "analog_system",
            nets,
            ports,
            analog_statements=analog_statements,
        )

    def _module_comment(self):
        return [
            *super()._module_comment(),
            "",
            "Functional Description",
            "",
            "The analog system directly modeled using differential",
            "equations.",
            "",
            "Specifically, we use the state space model equations",
            "",
            "ddt(x(t)) = A x(t) + B u(t) + Gamma s(t)",
            "s_tilde(t) = Gamma_tildeT x(t)",
            "",
            "where",
            "",
            f"x(t) = [{', '.join([f'{v.name}' for v in self._x])}]^T",
            f"u(t) = [{', '.join([f'{v.name}' for v in self._u])}]^T",
            f"s(t) = [{', '.join([f'{v.name}' for v in self._s])}]^T",
            f"s_tilde(t) = [{', '.join([f'{v.name}' for v in self._s_tilde])}]^T",
            "",
            "A \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.A[i, :]])}]"
                for i in range(self.analog_system.N)
            ],
            "",
            "B \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.B[i, :]])}]"
                for i in range(self.analog_system.N)
            ],
            "",
            "Gamma \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.Gamma[i, :]])}]"
                for i in range(self.analog_system.N)
            ],
            "",
            "Gamma_tildeT \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.Gamma_tildeT[i, :]])}]"
                for i in range(self.analog_system.M_tilde)
            ],
        ]
