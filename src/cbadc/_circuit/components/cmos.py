from typing import Dict
from .. import Port, Terminal, CircuitElement, SPICE_VALUE, _template_env
from ..models.cmos import NMOSModel, PMOSModel


class NMOS(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        W: SPICE_VALUE,
        L: SPICE_VALUE,
        VT0: SPICE_VALUE,
        KP: SPICE_VALUE,
        LAMBDA: SPICE_VALUE,
        RD: SPICE_VALUE,
        RS: SPICE_VALUE,
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
        )
        self.model = NMOSModel(
            model_name,
            VT0=VT0,
            KP=KP,
            LAMBDA=LAMBDA,
            RD=RD,
            RS=RS,
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
        VT0: SPICE_VALUE,
        KP: SPICE_VALUE,
        LAMBDA: SPICE_VALUE,
        RD: SPICE_VALUE,
        RS: SPICE_VALUE,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f"Expected str, got {type(instance_name)}")
        elif instance_name[0] != "M":
            instance_name = "M" + instance_name

        super().__init__(
            instance_name,
            model_name,
            W=W,
            L=L,
            VT0=VT0,
            KP=KP,
            LAMBDA=LAMBDA,
            RD=RD,
            RS=RS,
        )
        self.model = PMOSModel(
            model_name,
            VT0=VT0,
            KP=KP,
            LAMBDA=LAMBDA,
            RD=RD,
            RS=RS,
        )
