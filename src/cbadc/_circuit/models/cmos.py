from typing import List
from .. import DeviceModel, _template_env


class NMOSModel(DeviceModel):
    ng_spice_model_name = "NMOS"

    def __init__(
        self,
        model_name: str,
        VT0: float,
        KP: float,
        LAMBDA: float,
        RD: float,
        RS: float,
        level: str = "1",
        comments: List[str] = ["A simplistic NMOS model"],
    ):
        super().__init__(
            model_name,
            comments=comments,
            VT0=VT0,
            KP=KP,
            LAMBDA=LAMBDA,
            RD=RD,
            RS=RS,
            level=level,
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
                    "LAMBDA": self.parameters["LAMBDA"],
                    "RD": self.parameters["RD"],
                    "RS": self.parameters["RS"],
                    "level": self.parameters["level"],
                },
            }
        )


class PMOSModel(NMOSModel):
    ng_spice_model_name = "PMOS"
