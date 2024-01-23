from typing import Tuple
from . import Terminal, SubCircuitElement, Ground
from ..analog_frontend import AnalogFrontend
from ..analog_system import AnalogSystem
from ..digital_control import DigitalControl as NominalDigitalControl
from ..digital_control.dither_control import DitherControl as NominalDitherControl
from .components.passives import Resistor, Capacitor
import numpy as np
from .components.summer import DifferentialSummer
from .components.comparator import DifferentialOutputClockedComparator
from .components.opamp import OTA
from .digital_control import DigitalControl, DitherControl
from .analog_frontend import CircuitAnalogFrontend


class MultiInputOTA(SubCircuitElement):
    """A multi-input OTA.

    Parameters
    ----------
    instance_name : str
        The instance name of the OTA.
    analog_system : AnalogSystem
        The analog system to which this OTA belongs.
    index : int
        The index of the OTA. To determine between which states the integrating capacitor belongs.
    C : float
        The integration capacitance.
    DC_gain : float
        The DC gain of the OTA.

    """

    def __init__(
        self,
        instance_name: str,
        analog_system: AnalogSystem,
        index: int,
        C: float,
        DC_gain: float,
    ):
        super().__init__(
            instance_name,
            f"multi_input_ota_{index}",
            [Terminal("VSS"), Terminal("VDD"), Terminal("VCM")]
            + [Terminal(f"X{i}_P") for i in range(analog_system.N)]
            + [Terminal(f"X{i}_N") for i in range(analog_system.N)]
            + [Terminal(f"IN{i}_P") for i in range(analog_system.L)]
            + [Terminal(f"IN{i}_N") for i in range(analog_system.L)]
            + [Terminal(f"S{i}_P") for i in range(analog_system.M)]
            + [Terminal(f"S{i}_N") for i in range(analog_system.M)],
        )

        for n in range(analog_system.N):
            if analog_system.A[index, n] != 0.0:
                self._generate_ota_coupling(
                    analog_system.A[index, n] * C,
                    f"G_a_{index}_{n}",
                    (self[f"X{n}_P"], self[f"X{n}_N"]),
                    (self[f"X{index}_P"], self[f"X{index}_N"]),
                )

        for l in range(analog_system.L):
            if analog_system.B[index, l] != 0.0:
                self._generate_ota_coupling(
                    analog_system.B[index, l] * C,
                    f"G_b_{index}_{l}",
                    (
                        self[f"IN{l}_P"],
                        self[f"IN{l}_N"],
                    ),
                    (self[f"X{index}_P"], self[f"X{index}_N"]),
                )

        if analog_system.Gamma is None:
            raise ValueError("Gamma matrix must be defined")

        for m in range(analog_system.M):
            if analog_system.Gamma[index, m] != 0.0:
                self._generate_ota_coupling(
                    analog_system.Gamma[index, m] * C,
                    f"G_gamma_{index}_{m}",
                    (
                        self[f"S{m}_P"],
                        self[f"S{m}_N"],
                    ),
                    (self[f"X{index}_P"], self[f"X{index}_N"]),
                )

        # Finite DC gain
        if DC_gain is not np.inf:
            R_DC = DC_gain / (float(np.max(np.abs(analog_system.B))) * C)
            _RP = Resistor("Rp", 2 * R_DC)
            _RN = Resistor("Rn", 2 * R_DC)

            self.add(_RP, _RN)

            self.connects(
                (self[f"X{index}_P"], _RP[0]),
                (self[f"X{index}_N"], _RN[1]),
                (self["VCM"], _RP[1]),
                (self["VCM"], _RN[0]),
            )

    def _generate_ota_coupling(
        self,
        gm,
        instance_name: str,
        input_pair: Tuple[Terminal, Terminal],
        out_pair: Tuple[Terminal, Terminal],
    ):
        ota = OTA(
            instance_name,
            "ota",
            np.abs(gm),
        )
        self.add(ota)

        self.connects(
            (self["VSS"], ota["VSS"]),
            (self["VDD"], ota["VDD"]),
            (out_pair[0], ota["OUT_P"]),
            (out_pair[1], ota["OUT_N"]),
        )
        # Input polarity
        if gm < 0:
            self.connects((input_pair[0], ota["IN_P"]), (input_pair[1], ota["IN_N"]))
        else:
            self.connects((input_pair[1], ota["IN_P"]), (input_pair[0], ota["IN_N"]))


class GmCFrontend(CircuitAnalogFrontend):
    """A transconductance integrator based analog frontend.

    Parameters
    ----------
    analog_frontend : AnalogFrontend
        The analog frontend to which this circuit belongs.
    vdd_voltage : float, optional
        The supply voltage, by default 1.2
    in_high : float, optional
        The high input voltage threshold for the comparator, by default 0.0
    in_low : float, optional
        The low input voltage threshold for the comparator, by default 0.0
    C_int : float, optional
        The integration capacitance, by default 100e-15
    DC_gain : float, optional
        The DC gain of the OTA, by default 1e6
    """

    C: float

    def __init__(
        self,
        analog_frontend: AnalogFrontend,
        vdd_voltage: float = 1.2,
        in_high=0.0,
        in_low=0.0,
        C_int: float = 100e-15,
        DC_gain: float = 1e6,
    ):
        self.C = C_int
        super().__init__(
            analog_frontend,
            vdd_voltage,
            in_high,
            in_low,
            subckt_name="ota_frontend",
            instance_name="Xaf",
        )

        self._generate_integrators(analog_frontend.analog_system, C_int, DC_gain)

    def _generate_integrators(
        self, analog_system: AnalogSystem, C_int: float, DC_gain: float
    ):
        for n in range(analog_system.N):
            gmc = MultiInputOTA(f"gm{n}", analog_system, n, C_int, DC_gain)
            self.add(gmc)
            self.connects(
                (self["VSS"], gmc["VSS"]),
                (self["VDD"], gmc["VDD"]),
                (self["VCM"], gmc["VCM"]),
            )
            # States
            for nn in range(analog_system.N):
                self.connects(
                    (self.xp[nn], gmc[f"X{nn}_P"]),
                    (self.xn[nn], gmc[f"X{nn}_N"]),
                )
            # Inputs
            for l in range(analog_system.L):
                self.connects(
                    (self[f"IN{l}_P"], gmc[f"IN{l}_P"]),
                    (self[f"IN{l}_N"], gmc[f"IN{l}_N"]),
                )
            # Outputs
            for m in range(analog_system.M):
                self.connects(
                    (self[f"OUT{m}_P"], gmc[f"S{m}_P"]),
                    (self[f"OUT{m}_N"], gmc[f"S{m}_N"]),
                )

            # Integrating capacitors
            _cap_p = Capacitor(f"CP_{n}", 2 * C_int)
            _cap_n = Capacitor(f"CN_{n}", 2 * C_int)
            self.add(_cap_p, _cap_n)
            self.connects(
                (self.xp[n], _cap_p[0]),
                (self["VCM"], _cap_p[1]),
                (self.xn[n], _cap_n[1]),
                (self["VCM"], _cap_n[0]),
            )
