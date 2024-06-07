from typing import List
from .. import DeviceModel, _template_env


class NMOSModel(DeviceModel):
    ng_spice_model_name = "NMOS"

    def __init__(
        self,
        model_name: str,
        VT0: float,
        KP: float,
        GAMMA: float,
        PHI: float,
        LAMBDA: float,
        RD: float,
        RS: float,
        level: str = "1",
        CBD: float = 0.0,
        CBS: float = 0.0,
        IS: float = 0.0,
        PB: float = 0.8,
        CGSO: float = 0.0,
        CGDO: float = 0.0,
        CGBO: float = 0.0,
        RSH: float = 0.0,
        CJ: float = 0.56e-3,
        MJ: float = 0.45,
        CJSW: float = 0.35e-11,
        MJSW: float = 0.2,
        JS: float = 1e-8,
        TOX: float = 9e-9,
        NSUB: float = 9e14,
        comments: List[str] = ["A simplistic NMOS model"],
    ):
        super().__init__(
            model_name,
            comments=comments,
            VT0=VT0,
            KP=KP,
            GAMMA=GAMMA,
            PHI=PHI,
            LAMBDA=LAMBDA,
            RD=RD,
            RS=RS,
            level=level,
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

        # indicate that this model has a verilog-ams implementation
        self.verilog_ams = False

    def get_ngspice(self):
        return _template_env.get_template("ngspice/model.cir.j2").render(
            {
                "model_instance_name": self.model_name,
                "model_name": self.ng_spice_model_name,
                "description": self.comments,
                "parameters": {
                    "VT0": self.parameters["VT0"],
                    "KP": self.parameters["KP"],
                    "GAMMA": self.parameters["GAMMA"],
                    "PHI": self.parameters["PHI"],
                    "LAMBDA": self.parameters["LAMBDA"],
                    "RD": self.parameters["RD"],
                    "RS": self.parameters["RS"],
                    "level": self.parameters["level"],
                    "CBD": self.parameters["CBD"],
                    "CBS": self.parameters["CBS"],
                    "IS": self.parameters["IS"],
                    "PB": self.parameters["PB"],
                    "CGSO": self.parameters["CGSO"],
                    "CGDO": self.parameters["CGDO"],
                    "CGBO": self.parameters["CGBO"],
                    "RSH": self.parameters["RSH"],
                    "CJ": self.parameters["CJ"],
                    "MJ": self.parameters["MJ"],
                    "CJSW": self.parameters["CJSW"],
                    "MJSW": self.parameters["MJSW"],
                    "JS": self.parameters["JS"],
                    "TOX": self.parameters["TOX"],
                    "NSUB": self.parameters["NSUB"],
                },
            }
        )


class PMOSModel(NMOSModel):
    ng_spice_model_name = "PMOS"
