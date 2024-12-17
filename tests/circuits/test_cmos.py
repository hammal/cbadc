from cbadc.circuit import Terminal, SubCircuitElement
from cbadc.circuit.components.cmos import NMOS, PMOS


def test_nmos_model_ngspice():
    nmos = NMOS(
        instance_name="M1",
        model_name="nmos1",
        W=1e-6,
        L=2e-7,
        VT0=0.5,
        KP=0.5,
        LAMBDA=0.5,
        RD=0.5,
        RS=0.5,
    )

    terminals = [Terminal("D"), Terminal("G"), Terminal("S"), Terminal("B")]
    subckt = SubCircuitElement("Xsub", "subckt", terminals)
    subckt.add(nmos)
    subckt.connects(
        (subckt["D"], nmos["D"]),
        (subckt["G"], nmos["G"]),
        (subckt["S"], nmos["S"]),
        (subckt["B"], nmos["B"]),
    )
    component_string = nmos.get_ngspice(subckt._internal_connections)
    print(component_string)
    assert component_string == "M1 D G S B nmos1 W=1e-06 L=2e-07"
    model_string = [model.get_ngspice() for model in subckt._get_model_set()][0]
    print(model_string)
    assert (
        model_string
        == ".model nmos1 NMOS(VT0=0.5 KP=0.5 LAMBDA=0.5 RD=0.5 RS=0.5 level=1)"
    )


def test_pmos_model_ngspice():
    pmos = PMOS(
        instance_name="M1",
        model_name="pmos1",
        W=1e-6,
        L=2e-7,
        VT0=0.5,
        KP=0.5,
        LAMBDA=0.5,
        RD=0.5,
        RS=0.5,
    )

    terminals = [Terminal("D"), Terminal("G"), Terminal("S"), Terminal("B")]
    subckt = SubCircuitElement("Xsub", "subckt", terminals)
    subckt.add(pmos)
    subckt.connects(
        (subckt["D"], pmos["D"]),
        (subckt["G"], pmos["G"]),
        (subckt["S"], pmos["S"]),
        (subckt["B"], pmos["B"]),
    )
    component_string = pmos.get_ngspice(subckt._internal_connections)
    print(component_string)
    assert component_string == "M1 D G S B pmos1 W=1e-06 L=2e-07"
    model_string = [model.get_ngspice() for model in subckt._get_model_set()][0]
    print(model_string)
    assert (
        model_string
        == ".model pmos1 PMOS(VT0=0.5 KP=0.5 LAMBDA=0.5 RD=0.5 RS=0.5 level=1)"
    )
