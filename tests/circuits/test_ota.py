from cbadc.circuit.ota import GmCFrontend
from cbadc.synthesis.leap_frog import get_leap_frog


def test_ota_analog_frontend_ngspice():
    N = 4
    ENOB = 16
    BW = 1e6
    C_int = 1e-13
    vdd = 1.2
    in_high = vdd
    in_low = 0.0

    analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)

    ota_analog_frontend = GmCFrontend(analog_frontend, vdd, in_high, in_low, C_int)

    print('\n')
    print(ota_analog_frontend.get_ngspice())
    print('\n\n')

    # for subckt in opamp_analog_frontend.subckt_components:
    #     print(subckt.get_ngspice({}))

    for subckt_definition in ota_analog_frontend.get_sub_circuit_definitions():
        print(subckt_definition)
        print('\n')

    print('\n')
    for model in ota_analog_frontend._get_model_set():
        print(model.get_ngspice())
    assert True
