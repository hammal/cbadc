from cbadc.circuit.opamp import OpAmpFrontend
from cbadc.digital_control import DigitalControl
from cbadc.analog_signal import Clock
from cbadc.synthesis.leap_frog import get_leap_frog


def test_opamp_analog_frontend_ngspice():
    N = 4
    ENOB = 16
    BW = 1e6

    GBWP = 1e9
    DC_gain = 1e3
    vdd = 1.2
    in_high = vdd
    in_low = 0.0

    analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)

    opamp_analog_frontend = OpAmpFrontend(
        analog_frontend, GBWP, DC_gain, vdd, in_high, in_low
    )

    print('\n')
    print(opamp_analog_frontend.get_ngspice())
    print('\n\n')

    # for subckt in opamp_analog_frontend.subckt_components:
    #     print(subckt.get_ngspice({}))

    for subckt_definition in opamp_analog_frontend.get_sub_circuit_definitions():
        print(subckt_definition)
        print('\n')

    print('\n')
    for model in opamp_analog_frontend._get_model_set():
        print(model.get_ngspice())

    assert True


# def test_state_space_analog_frontend_spectre():
#     N = 4
#     ENOB = 14
#     BW = 1e6
#     vdd = 1.2
#     vgnd = 0.0

#     analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)

#     state_space_analog_frontend = StateSpaceAnalogFrontend(analog_frontend, vdd, vgnd)

#     print('\n')
#     print(state_space_analog_frontend.get_spectre())
#     print('\n\n')

#     for subckt_definition in state_space_analog_frontend.sub_circuit_definition_get_spectre():
#         print(subckt_definition)
#         print('\n')

#     print('\n')
#     for model in state_space_analog_frontend.get_model_set():
#         if model.verilog_ams:
#             print(model.get_verilog_ams())
#         else:
#             print(model.get_spectre())
#         print('\n')

#     assert True
