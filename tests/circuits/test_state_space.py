# from cbadc.circuit.state_space_model import DigitalControl, IntegratorAnalogSystem, StateSpaceAnalogFrontend
# from cbadc.digital_control import DigitalControl
# from cbadc.analog_signal import Clock
# from cbadc.synthesis.leap_frog import get_leap_frog

# def test_digital_control_ngspice():
#     vdd = 1.2
#     vgnd = 0.0
#     T = 1e-9
#     M = 4

#     digital_control = DigitalControl(Clock(T), M)

#     clocked_comparator_digital_control = DigitalControl(
#         digital_control, vdd, vgnd
#     )
#     print('\n')
#     print(clocked_comparator_digital_control.get_ngspice())
#     print('\n\n')

#     for subckt_definition in clocked_comparator_digital_control.sub_circuit_definition_get_ngspice():
#         print(subckt_definition)
#         print('\n')

#     print('\n')
#     for model in clocked_comparator_digital_control.get_model_set():
#         print(model.get_ngspice())
#         print('\n')

#     assert True


# def test_digital_control_spectre():
#     vdd = 1.2
#     vgnd = 0.0
#     T = 1e-9
#     M = 4

#     digital_control = DigitalControl(Clock(T), M)

#     clocked_comparator_digital_control = DigitalControl(
#         digital_control, vdd, vgnd
#     )
#     print('\n')
#     print(clocked_comparator_digital_control.get_spectre())
#     print('\n\n')

#     for subckt_definition in clocked_comparator_digital_control.sub_circuit_definition_get_spectre():
#         print(subckt_definition)
#         print('\n')

#     print('\n')
#     for model in clocked_comparator_digital_control.get_model_set():
#         if model.verilog_ams:
#             print(model.get_verilog_ams())
#         else:
#             print(model.get_spectre())
#         print('\n')

#     assert True


# def test_analog_system_ngspice():
#     N = 4
#     ENOB = 14
#     BW = 1e6
#     vdd = 1.2
#     vgd = 0.0
#     analog_system = get_leap_frog(N=N, ENOB=ENOB, BW=BW)

#     integrator_analog_system = IntegratorAnalogSystem(analog_system.analog_system, vdd, vgd)

#     print('\n')
#     print(integrator_analog_system.get_ngspice())
#     print('\n\n')

#     for subckt_definition in integrator_analog_system.get_sub_circuit_definitions():
#         print(subckt_definition)
#         print('\n')

#     print('\n')
#     for model in integrator_analog_system._get_model_set():
#         print(model.get_ngspice())

#     assert True

# def test_analog_system_spectre():
#     N = 4
#     ENOB = 14
#     BW = 1e6
#     vdd = 1.2
#     vgd = 0.0

#     analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)

#     integrator_analog_system = IntegratorAnalogSystem(analog_frontend.analog_system, vdd, vgd)

#     print('\n')
#     print(integrator_analog_system.get_spectre())
#     print('\n\n')

#     for subckt_definition in integrator_analog_system._get_spectre_sub_circuit_definition():
#         print(subckt_definition)
#         print('\n')

#     print('\n')
#     for model in integrator_analog_system._get_model_set():
#         if model.verilog_ams:
#             print(model.get_verilog_ams())
#         else:
#             print(model.get_spectre())
#         print('\n')
#     assert True


# def test_state_space_analog_frontend_ngspice():
#     N = 4
#     ENOB = 14
#     BW = 1e6
#     vdd = 1.2
#     vgnd = 0.0

#     analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)

#     state_space_analog_frontend = StateSpaceAnalogFrontend(analog_frontend, vdd, vgnd)

#     print('\n')
#     print(state_space_analog_frontend.get_ngspice())
#     print('\n\n')

#     for subckt_definition in state_space_analog_frontend.get_sub_circuit_definitions():
#         print(subckt_definition)
#         print('\n')

#     print('\n')
#     for model in state_space_analog_frontend._get_model_set():
#         print(model.get_ngspice())

#     assert True


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

#     for subckt_definition in state_space_analog_frontend._get_spectre_sub_circuit_definition():
#         print(subckt_definition)
#         print('\n')

#     print('\n')
#     for model in state_space_analog_frontend._get_model_set():
#         if model.verilog_ams:
#             print(model.get_verilog_ams())
#         else:
#             print(model.get_spectre())
#         print('\n')

#     assert True
