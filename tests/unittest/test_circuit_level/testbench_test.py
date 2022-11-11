import cbadc
import os
from tests.fixture.chain_of_integrators import get_simulator, chain_of_integrators


def test_AnalogFrontend(get_simulator):
    digital_control_module = cbadc.circuit.DigitalControl(get_simulator.digital_control)
    ADC = 1e5
    C = 1e-12
    analog_system_module = cbadc.circuit.op_amp.analog_system.AnalogSystemIdealOpAmp(
        analog_system=get_simulator.analog_system, C=C
    )
    analog_frontend_module = cbadc.circuit.analog_frontend.AnalogFrontend(
        analog_system_module, digital_control_module
    )
    clock = cbadc.analog_signal.Clock(
        digital_control_module.digital_control.clock.T * 1e-2
    )
    vsgd = 400e-3
    vdd = 800e-3
    vgd = 0.0
    # 10000 control cycles
    t_stop = digital_control_module.digital_control.clock.T * 1e5
    sinusoidal = cbadc.analog_signal.Sinusoidal(400e-3, 50e0, 0, vsgd)
    testbench = cbadc.circuit.testbench.TestBench(
        analog_frontend_module, [sinusoidal], clock, "my_testbench", vdd, vgd
    )
    path_here = './test_bench'
    if not os.path.exists(path_here):
        os.mkdir(path_here)
    testbench.to_file("this_file", path=path_here)


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
    testbench.to_file('testbench')


def test_testbench_save_all():
    ENOB = 12
    N = 5
    BW = 1e6
    analog_frontend_target = cbadc.synthesis.get_leap_frog(ENOB=ENOB, N=N, BW=BW)
    amplitude = 1.0
    frequency = 1.0 / analog_frontend_target.digital_control.clock.T
    while frequency > BW:
        frequency /= 2
    input_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency)
    testbench = cbadc.circuit.get_testbench(
        analog_frontend_target, [input_signal], save_all_variables=True
    )
    testbench.to_file('testbench')


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
    testbench.to_file('testbench1')
    testbench = cbadc.circuit.get_opamp_testbench(
        analog_frontend_target, [input_signal], C, GBWP=GBWP, A_DC=A_DC
    )
    testbench.to_file('testbench2')


def test_opamp_testbench_save_all():
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
        analog_frontend_target, [input_signal], C, save_all_variables=True
    )
    testbench.to_file('testbench1')
    testbench = cbadc.circuit.get_opamp_testbench(
        analog_frontend_target,
        [input_signal],
        C,
        GBWP=GBWP,
        A_DC=A_DC,
        save_all_variables=True,
    )
    testbench.to_file('testbench2')
