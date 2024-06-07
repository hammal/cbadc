from typing import Dict, List
from .. import (
    Port,
    Terminal,
    CircuitElement,
    SPICE_VALUE,
    _template_env,
    SubCircuitElement,
)
from ..models.cmos import NMOSModel, PMOSModel


class NMOS(CircuitElement):
    """
    Default paramerization for 500nm technology from "Design of Analog CMOS Integrated Circuits" by Behzad Razavi.
    """

    def __init__(
        self,
        instance_name: str,
        model_name: str,
        W: SPICE_VALUE,
        L: SPICE_VALUE,
        VT0: SPICE_VALUE = 0.7,
        KP: SPICE_VALUE = 2.0e-5,
        GAMMA: SPICE_VALUE = 0.45,
        PHI: SPICE_VALUE = 0.9,
        LAMBDA: SPICE_VALUE = 0.1,
        RD: SPICE_VALUE = 0.0,
        RS: SPICE_VALUE = 0.0,
        CBD: SPICE_VALUE = 0.0,
        CBS: SPICE_VALUE = 0.0,
        IS: SPICE_VALUE = 0.0,
        PB: SPICE_VALUE = 0.9,
        CGSO: SPICE_VALUE = 0.0,
        CGDO: SPICE_VALUE = 4e-10,
        CGBO: SPICE_VALUE = 0.0,
        RSH: SPICE_VALUE = 0.0,
        CJ: SPICE_VALUE = 0.56e-3,
        MJ: SPICE_VALUE = 0.45,
        CJSW: SPICE_VALUE = 0.35e-11,
        MJSW: SPICE_VALUE = 0.2,
        JS: SPICE_VALUE = 1e-8,
        TOX: SPICE_VALUE = 9e-9,
        NSUB: SPICE_VALUE = 9e14,
        m: SPICE_VALUE = 1,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f"Expected str, got {type(instance_name)}")
        elif instance_name[0] != "M":
            instance_name = "M" + instance_name

        super().__init__(
            instance_name,
            [
                Terminal("D"),
                Terminal("G"),
                Terminal("S"),
                Terminal("B"),
            ],
            W=W,
            L=L,
            m=m,
        )
        self.W = W
        self.L = L
        self.m = m

        self.VT0 = VT0
        self.KP = KP
        self.GAMMA = GAMMA
        self.PHI = PHI
        self.LAMBDA = LAMBDA
        self.RD = RD
        self.RS = RS
        self.CBD = CBD
        self.CBS = CBS
        self.IS = IS
        self.PB = PB
        self.CGSO = CGSO
        self.CGDO = CGDO
        self.CGBO = CGBO
        self.RSH = RSH
        self.CJ = CJ
        self.MJ = MJ
        self.CJSW = CJSW
        self.MJSW = MJSW
        self.JS = JS
        self.TOX = TOX
        self.NSUB = NSUB

        self.model = NMOSModel(
            model_name,
            VT0=VT0,
            KP=KP,
            GAMMA=GAMMA,
            PHI=PHI,
            LAMBDA=LAMBDA,
            RD=RD,
            RS=RS,
            CBD=CBD,
            CBS=CBS,
            IS=IS,
            PB=PB,
            CGSO=CGSO,
            CGDO=CGDO,
            CGBO=CGBO,
            RSH=RSH,
            CJ=CJ,
            MJ=MJ,
            CJSW=CJSW,
            MJSW=MJSW,
            JS=JS,
            TOX=TOX,
            NSUB=NSUB,
        )

    def copy(self, instance_name: str, W: float = 1.0, L: float = 1.0, m: int = 1):
        return self.__class__(
            instance_name,
            self.model.model_name,
            W * self.W,
            L * self.L,
            self.VT0,
            self.KP,
            self.GAMMA,
            self.PHI,
            self.LAMBDA,
            self.RD,
            self.RS,
            CBD=self.CBD,
            CBS=self.CBS,
            IS=self.IS,
            PB=self.PB,
            CGSO=self.CGSO,
            CGDO=self.CGDO,
            CGBO=self.CGBO,
            RSH=self.RSH,
            CJ=self.CJ,
            MJ=self.MJ,
            CJSW=self.CJSW,
            MJSW=self.MJSW,
            JS=self.JS,
            TOX=self.TOX,
            NSUB=self.NSUB,
            m=self.m * m,
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]):
        named_nodes = self._get_terminal_names(connections)
        return _template_env.get_template("ngspice/mosfet.cir.j2").render(
            {
                "instance_name": self.instance_name,
                "terminals": named_nodes,
                "parameters": self._parameters_dict,
                "comments": self.comments,
                "model_instance_name": self.model.model_name,
            }
        )


class PMOS(NMOS):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        W: SPICE_VALUE,
        L: SPICE_VALUE,
        VT0: SPICE_VALUE = -0.8,
        KP: SPICE_VALUE = 2.0e-5,
        GAMMA: SPICE_VALUE = 0.4,
        PHI: SPICE_VALUE = 0.8,
        LAMBDA: SPICE_VALUE = 0.2,
        RD: SPICE_VALUE = 0.0,
        RS: SPICE_VALUE = 0.0,
        CBD: SPICE_VALUE = 0.0,
        CBS: SPICE_VALUE = 0.0,
        IS: SPICE_VALUE = 0.0,
        PB: SPICE_VALUE = 0.9,
        CGSO: SPICE_VALUE = 0.0,
        CGDO: SPICE_VALUE = 0.3e-9,
        CGBO: SPICE_VALUE = 0.0,
        RSH: SPICE_VALUE = 0.0,
        CJ: SPICE_VALUE = 0.94e-3,
        MJ: SPICE_VALUE = 0.5,
        CJSW: SPICE_VALUE = 0.32e-11,
        MJSW: SPICE_VALUE = 0.3,
        JS: SPICE_VALUE = 0.5e-8,
        TOX: SPICE_VALUE = 9e-9,
        NSUB: SPICE_VALUE = 5e14,
        m: SPICE_VALUE = 1,
    ):
        super().__init__(
            instance_name,
            model_name,
            W=W,
            L=L,
            VT0=VT0,
            GAMMA=GAMMA,
            PHI=PHI,
            KP=KP,
            LAMBDA=LAMBDA,
            RD=RD,
            RS=RS,
            CBD=CBD,
            CBS=CBS,
            IS=IS,
            PB=PB,
            CGSO=CGSO,
            CGDO=CGDO,
            CGBO=CGBO,
            RSH=RSH,
            CJ=CJ,
            MJ=MJ,
            CJSW=CJSW,
            MJSW=MJSW,
            JS=JS,
            TOX=TOX,
            NSUB=NSUB,
            m=m,
        )
        self.model = PMOSModel(
            model_name,
            VT0=VT0,
            GAMMA=GAMMA,
            PHI=PHI,
            KP=KP,
            LAMBDA=LAMBDA,
            RD=RD,
            RS=RS,
            CBD=CBD,
            CBS=CBS,
            IS=IS,
            PB=PB,
            CGSO=CGSO,
            CGDO=CGDO,
            CGBO=CGBO,
            RSH=RSH,
            CJ=CJ,
            MJ=MJ,
            CJSW=CJSW,
            MJSW=MJSW,
            JS=JS,
            TOX=TOX,
            NSUB=NSUB,
        )


class Inverter(SubCircuitElement):
    def __init__(
        self,
        instance_name: str,
        sub_ckt_name: str,
        nmos_template: NMOS,
        pmos_template: PMOS,
        m: SPICE_VALUE = 1,
    ):
        super().__init__(
            instance_name,
            sub_ckt_name,
            [
                Terminal("IN"),
                Terminal("OUT"),
                Terminal("VDD"),
                Terminal("GND"),
            ],
            m=m,
        )
        nmos = nmos_template.copy("M2")
        pmos = pmos_template.copy("M1")

        self.add(nmos)
        self.add(pmos)

        self.connects(
            (self["IN"], nmos["G"]),
            (nmos["G"], pmos["G"]),
            (self["OUT"], nmos["D"]),
            (nmos["D"], pmos["D"]),
            (self["VDD"], pmos["S"]),
            (self["VDD"], pmos["B"]),
            (self["GND"], nmos["S"]),
            (self["GND"], nmos["B"]),
        )


class RSLatch(SubCircuitElement):
    def __init__(
        self,
        instance_name: str,
        sub_ckt_name: str,
        nmos_template: NMOS,
        pmos_template: PMOS,
        m: SPICE_VALUE = 1,
    ):
        super().__init__(
            instance_name,
            sub_ckt_name,
            [
                Terminal("SET"),
                Terminal("RESET"),
                Terminal("Q"),
                Terminal("Q_BAR"),
                Terminal("VDD"),
                Terminal("GND"),
            ],
            m=m,
        )
        inv_1 = Inverter(
            "INV1", "INV", nmos_template.copy("M1"), pmos_template.copy("M2")
        )
        inv_2 = Inverter(
            "INV2", "INV", nmos_template.copy("M3"), pmos_template.copy("M4")
        )
        inv_3 = Inverter(
            "INV3", "INV", nmos_template.copy("M5"), pmos_template.copy("M6")
        )
        inv_4 = Inverter(
            "INV4", "INV", nmos_template.copy("M7"), pmos_template.copy("M8")
        )

        m1 = nmos_template.copy("M1")
        m2 = nmos_template.copy("M2")

        self.add(inv_1)
        self.add(inv_2)
        self.add(inv_3)
        self.add(inv_4)
        self.add(m1)
        self.add(m2)

        # connect power
        self.connects(
            (self["VDD"], inv_1["VDD"]),
            (self["VDD"], inv_2["VDD"]),
            (self["VDD"], inv_3["VDD"]),
            (self["VDD"], inv_4["VDD"]),
            (self["GND"], inv_1["GND"]),
            (self["GND"], inv_2["GND"]),
            (self["GND"], inv_3["GND"]),
            (self["GND"], inv_4["GND"]),
            (self["GND"], m1["S"]),
            (self["GND"], m1["B"]),
            (self["GND"], m2["S"]),
            (self["GND"], m2["B"]),
        )

        g1 = Terminal("G1")
        g2 = Terminal("G2")

        # connect signals
        self.connects(
            (self["SET"], inv_1["IN"]),
            (self["RESET"], inv_2["IN"]),
            (g1, inv_1["OUT"]),
            (g2, inv_2["OUT"]),
            (g1, m1["G"]),
            (g2, m2["G"]),
            (self["Q"], m1["D"]),
            (self["Q_BAR"], m2["D"]),
            (self["Q"], inv_3["IN"]),
            (self["Q"], inv_4["OUT"]),
            (self["Q_BAR"], inv_3["OUT"]),
            (self["Q_BAR"], inv_4["IN"]),
        )


class StrongARMLatch(SubCircuitElement):
    def __init__(
        self,
        instance_name: str,
        sub_ckt_name: str,
        nmos_template: NMOS,
        pmos_template: PMOS,
        input_weights: List[float],
        m: SPICE_VALUE = 1,
    ):
        super().__init__(
            instance_name,
            sub_ckt_name,
            [
                Terminal("VDD"),
                Terminal("GND"),
                Terminal("CLK"),
            ]
            + [Terminal(f"VIN_{n}_P") for n in range(len(input_weights))]
            + [Terminal(f"VIN_{n}_N") for n in range(len(input_weights))]
            + [Terminal("VOUT_P"), Terminal("VOUT_N")],
            m=m,
        )

        s1 = pmos_template.copy("S1")
        s2 = pmos_template.copy("S2")
        s3 = pmos_template.copy("S3")
        s4 = pmos_template.copy("S4")

        m3 = nmos_template.copy("M3")
        m4 = nmos_template.copy("M4")

        m5 = pmos_template.copy("M5")
        m6 = pmos_template.copy("M6")

        m7 = nmos_template.copy("M7")

        self.add(s1)
        self.add(s2)
        self.add(s3)
        self.add(s4)
        self.add(m3)
        self.add(m4)
        self.add(m5)
        self.add(m6)
        self.add(m7)

        m1 = []
        m2 = []
        for n in range(len(input_weights)):
            m1.append(nmos_template.copy(f"M1_{n}", W=input_weights[n]))
            self.add(m1[-1])
            m2.append(nmos_template.copy(f"M2_{n}", W=input_weights[n]))
            self.add(m2[-1])

        p = Terminal("P")
        q = Terminal("Q")

        x = Terminal("X")
        y = Terminal("Y")

        s = Terminal("S")

        # connect Externals
        self.connects(
            (self["VDD"], s1["S"]),
            (self["VDD"], s2["S"]),
            (self["VDD"], s3["S"]),
            (self["VDD"], s4["S"]),
            (self["VDD"], m5["S"]),
            (self["VDD"], m6["S"]),
            (self["VDD"], s1["B"]),
            (self["VDD"], s2["B"]),
            (self["VDD"], s3["B"]),
            (self["VDD"], s4["B"]),
            (self["VDD"], m5["B"]),
            (self["VDD"], m6["B"]),
            (self["GND"], m7["S"]),
            (self["GND"], m7["B"]),
            (self["CLK"], s1["G"]),
            (self["CLK"], s2["G"]),
            (self["CLK"], s3["G"]),
            (self["CLK"], s4["G"]),
            (self["CLK"], m7["G"]),
            (s, m7["D"]),
        )

        # connect input diff pairs
        for n in range(len(input_weights)):
            self.connects(
                (p, m1[n]["D"]),
                (q, m2[n]["D"]),
                (s, m1[n]["S"]),
                (s, m2[n]["S"]),
                # unsure where to put the bulk at S or GND?
                # (self["GND"], m1[n]["B"]),
                # (self["GND"], m2[n]["B"]),
                (s, m1[n]["B"]),
                (s, m2[n]["B"]),
                (self[f"VIN_{n}_P"], m1[n]["G"]),
                (self[f"VIN_{n}_N"], m2[n]["G"]),
            )

        self.connects(
            (p, s1["D"]),
            (q, s2["D"]),
            (x, s3["D"]),
            (y, s4["D"]),
            (x, m5["D"]),
            (y, m5["G"]),
            (y, m6["D"]),
            (x, m6["G"]),
            (x, m3["D"]),
            (y, m4["D"]),
            (x, m4["G"]),
            (y, m3["G"]),
            (p, m3["S"]),
            (q, m4["S"]),
            # unsure where to put the bulk at P or GND?
            # (self["GND"], m3["B"]),
            # (self["GND"], m4["B"]),
            (p, m3["B"]),
            (q, m4["B"]),
        )

        self.connects(
            (self["VOUT_P"], y),
            (self["VOUT_N"], x),
        )


class Comparator(SubCircuitElement):
    def __init__(
        self,
        instance_name: str,
        sub_ckt_name: str,
        nmos_template: NMOS,
        pmos_template: PMOS,
        input_weights: List[float],
        m: SPICE_VALUE = 1,
    ):
        super().__init__(
            instance_name,
            sub_ckt_name,
            [
                Terminal("VDD"),
                Terminal("GND"),
                Terminal("CLK"),
            ]
            + [Terminal(f"VIN_{n}_P") for n in range(len(input_weights))]
            + [Terminal(f"VIN_{n}_N") for n in range(len(input_weights))]
            + [Terminal("VOUT_P"), Terminal("VOUT_N")],
            m=m,
        )

        rs_latch = RSLatch("XRS", "rs_latch", nmos_template, pmos_template)
        sarmed_latch = StrongARMLatch(
            "XSAL", "strong_armed_latch", nmos_template, pmos_template, input_weights
        )

        self.add(rs_latch)
        self.add(sarmed_latch)

        # connect externals
        self.connects(
            (self["VDD"], rs_latch["VDD"]),
            (self["VDD"], sarmed_latch["VDD"]),
            (self["GND"], rs_latch["GND"]),
            (self["GND"], sarmed_latch["GND"]),
            (self["CLK"], sarmed_latch["CLK"]),
        )

        x = Terminal("X")
        y = Terminal("Y")

        # connect signals
        self.connects(
            (x, sarmed_latch["VOUT_P"]),
            (y, sarmed_latch["VOUT_N"]),
            (x, rs_latch["SET"]),
            (y, rs_latch["RESET"]),
            (self["VOUT_P"], rs_latch["Q"]),
            (self["VOUT_N"], rs_latch["Q_BAR"]),
        )

        for n in range(len(input_weights)):
            self.connects(
                (self[f"VIN_{n}_P"], sarmed_latch[f"VIN_{n}_P"]),
                (self[f"VIN_{n}_N"], sarmed_latch[f"VIN_{n}_N"]),
            )
