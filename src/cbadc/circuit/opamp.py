from typing import Tuple
from . import Terminal, SubCircuitElement, Ground
from ..analog_frontend import AnalogFrontend
from ..analog_filter import AnalogSystem
from ..digital_control import DigitalControl as NominalDigitalControl
from ..digital_control.dither_control import DitherControl as NominalDitherControl
from .components.passives import Resistor, Capacitor
import numpy as np
from .components.opamp import OpAmp
from .digital_control import DigitalControl, DitherControl
from .analog_frontend import CircuitAnalogFrontend


class OpAmpFrontend(CircuitAnalogFrontend):
    """An opamp-based analog frontend.

    Parameters
    ----------
    analog_frontend : AnalogFrontend
        The analog frontend to which this opamp-based analog frontend belongs.
    GBWP : float
        The GBWP of the opamp.
    DC_gain : float
        The DC gain of the opamp.
    vdd_voltage : float, optional
        The supply voltage of the opamp, by default 1.2.
    in_high : float, optional
        The high input voltage of the opamp, by default 0.0.
    in_low : float, optional
        The low input voltage of the opamp, by default 0.0.
    C_int : float, optional
        The integration capacitance, by default 100e-15.
    C_amp : float, optional
        The capacitance of the internal opamp pole, by default 10e-15.

    """

    C: float

    def __init__(
        self,
        analog_frontend: AnalogFrontend,
        GBWP: float,
        DC_gain: float,
        vdd_voltage: float = 1.2,
        in_high=0.0,
        in_low=0.0,
        C_int: float = 100e-15,
        C_amp: float = 10e-15,
    ):
        self.analog_frontend = analog_frontend

        self.vgndp = [
            Terminal(f"VGND_{i}_P") for i in range(analog_frontend.analog_filter.N)
        ]
        self.vgndn = [
            Terminal(f"VGND_{i}_N") for i in range(analog_frontend.analog_filter.N)
        ]

        super().__init__(
            analog_frontend,
            vdd_voltage,
            in_high,
            in_low,
            subckt_name="opamp_analog_frontend",
            instance_name="Xaf",
        )

        self._generate_state_resistor_network(analog_frontend, C_int)
        # self._generate_observation_network(analog_frontend.analog_filter)

        self._generate_integrators(
            analog_frontend.analog_filter, GBWP, DC_gain, C_amp, C_int
        )

    def _generate_integrators(
        self,
        analog_filter: AnalogSystem,
        GBWP: float,
        DC_gain: float,
        C_amp: float,
        C_int: float,
    ):
        opamps = [
            OpAmp(
                f"Xop_{n}",
                "opamp",
                GBWP=GBWP,
                DC_gain=DC_gain,
                C=C_amp,
            )
            for n in range(analog_filter.N)
        ]

        self.add(*opamps)

        for n in range(analog_filter.N):
            self.connects(
                (self.xp[n], opamps[n]["OUT_P"]),
                (self.xn[n], opamps[n]["OUT_N"]),
                (self.vgndp[n], opamps[n]["IN_P"]),
                (self.vgndn[n], opamps[n]["IN_N"]),
                (self["VSS"], opamps[n]["VSS"]),
                (self["VDD"], opamps[n]["VDD"]),
                (self["VCM"], opamps[n]["VCM"]),
            )

        _caps_p = [
            Capacitor(
                f"CP_{index}",
                C_int,
            )
            for index in range(analog_filter.N)
        ]
        _caps_n = [
            Capacitor(
                f"CN_{index}",
                C_int,
            )
            for index in range(analog_filter.N)
        ]

        self.add(*_caps_p, *_caps_n)

        for n in range(analog_filter.N):
            self.connects(
                (self.xp[n], _caps_p[n][1]),
                (self.vgndp[n], _caps_p[n][0]),
                (self.xn[n], _caps_n[n][1]),
                (self.vgndn[n], _caps_n[n][0]),
            )

    def _generate_resistor_pair(
        self,
        nominal_value: float,
        instance_name: str,
        t1: Tuple[Terminal, Terminal],
        t2: Tuple[Terminal, Terminal],
        C_int: float,
    ):
        _RP = Resistor(f"RP_{instance_name}", 1.0 / (1 * C_int * np.abs(nominal_value)))
        _RN = Resistor(f"RN_{instance_name}", 1.0 / (1 * C_int * np.abs(nominal_value)))
        self.add(_RP, _RN)
        if nominal_value < 0:
            # inverting connection
            self.connect(t1[0], _RP[0])
            self.connect(t1[1], _RN[0])
            self.connect(t2[0], _RP[1])
            self.connect(t2[1], _RN[1])
        else:
            # non-inverting connection
            self.connect(t1[0], _RP[0])
            self.connect(t1[1], _RN[0])
            self.connect(t2[0], _RN[1])
            self.connect(t2[1], _RP[1])

    def _generate_state_resistor_network(
        self, analog_frontend: AnalogFrontend, C_int: float
    ):
        if analog_frontend.analog_filter.Gamma is None:
            raise BaseException("Gamma must not be Nonw")

        # B Matrix
        for n in range(analog_frontend.analog_filter.N):
            for l in range(analog_frontend.analog_filter.L):
                if analog_frontend.analog_filter.B[n, l] != 0.0:
                    self._generate_resistor_pair(
                        analog_frontend.analog_filter.B[n, l],
                        f"b_{n}_{l}",
                        (
                            self[f"IN{l}_P"],
                            self[f"IN{l}_N"],
                        ),
                        (self.vgndp[n], self.vgndn[n]),
                        C_int,
                    )

        # A Matrix
        for n in range(analog_frontend.analog_filter.N):
            for nn in range(analog_frontend.analog_filter.N):
                if analog_frontend.analog_filter.A[n, nn] != 0.0:
                    self._generate_resistor_pair(
                        analog_frontend.analog_filter.A[n, nn],
                        f"a_{n}_{nn}",
                        (self.xp[nn], self.xn[nn]),
                        (self.vgndp[n], self.vgndn[n]),
                        C_int,
                    )

        # Gamma Matrix
        for n in range(analog_frontend.analog_filter.N):
            for m in range(analog_frontend.analog_filter.M):
                if analog_frontend.analog_filter.Gamma[n, m] != 0.0:
                    self._generate_resistor_pair(
                        analog_frontend.analog_filter.Gamma[n, m],
                        f"gamma_{n}_{m}",
                        (
                            self[f"OUT{m}_P"],
                            self[f"OUT{m}_N"],
                        ),
                        (self.vgndp[n], self.vgndn[n]),
                        C_int,
                    )
