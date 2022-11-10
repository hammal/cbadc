import cbadc

# def test_AnalogFrontend(get_simulator):
#     digital_control_module = cbadc.circuit_level.digital_control.DigitalControl(
#         get_simulator.digital_control
#     )
#     ADC = 1e5
#     C = 1e-12
#     analog_system_module = (
#         cbadc.circuit_level.op_amp.analog_system.AnalogSystemFiniteGainOpAmp(
#             analog_system=get_simulator.analog_system, C=C, A_DC=ADC
#         )
#     )
#     analog_frontend_module = cbadc.circuit_level.analog_frontend.AnalogFrontend(
#         analog_system_module, digital_control_module
#     )
#     clock = Clock(digital_control_module.digital_control.clock.T * 1e-2)
#     vsgd = 400e-3
#     vdd = 800e-3
#     vgd = 0.0
#     # 10000 control cycles
#     t_stop = digital_control_module.digital_control.clock.T * 1e5
#     sinusoidal = Sinusoidal(400e-3, 50e0, 0, vsgd)
#     testbench = cbadc.circuit_level.testbench.TestBench(
#         analog_frontend_module, sinusoidal, clock, t_stop, "my_testbench", vdd, vgd
#     )
#     path_here = './test_bench'
#     if not os.path.exists(path_here):
#         os.mkdir(path_here)
#     testbench.to_file("this_file", path=path_here)


def test_get_testbench():
    ENOB = 12
    N = 5
    BW = 1e6
    analog_frontend_target = cbadc.synthesis.get_leap_frog(ENOB=ENOB, N=N, BW=BW)
    amplitude = 1.0
    frequency = 1.0 / analog_frontend_target.digital_control.clock.T
    while frequency > BW:
        frequency /= 2
    input_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency)
    testbench = cbadc.circuit.get_testbench(analog_frontend_target, [input_signal])


def test_get_opamp_testbench():
    ENOB = 12
    N = 5
    BW = 1e6
    analog_frontend_target = cbadc.synthesis.get_leap_frog(ENOB=ENOB, N=N, BW=BW)
    amplitude = 1.0
    frequency = 1.0 / analog_frontend_target.digital_control.clock.T
    while frequency > BW:
        frequency /= 2
    input_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency)
    C = 1e-12
    A_DC = 1e2
    GBWP = BW * A_DC

    testbench = cbadc.circuit.get_opamp_testbench(
        analog_frontend_target, [input_signal], C
    )
    testbench = cbadc.circuit.get_opamp_testbench(
        analog_frontend_target, [input_signal], C, GBWP=GBWP, A_DC=A_DC
    )
