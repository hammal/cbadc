from tests.fixture.chain_of_integrators import get_simulator, chain_of_integrators
import cbadc.circuit_level.digital_control
import cbadc.circuit_level.op_amp
import cbadc.circuit_level.analog_frontend
from cbadc.analog_signal import Clock, Sinusoidal
import cbadc.circuit_level.testbench
import os


def test_AnalogFrontend(get_simulator):
    digital_control_module = cbadc.circuit_level.digital_control.DigitalControl(
        get_simulator.digital_control
    )
    ADC = 1e5
    C = 1e-12
    analog_system_module = (
        cbadc.circuit_level.op_amp.analog_system.AnalogSystemFiniteGainOpAmp(
            get_simulator.analog_system, C, ADC
        )
    )
    analog_frontend_module = cbadc.circuit_level.analog_frontend.AnalogFrontend(
        analog_system_module, digital_control_module
    )
    clock = Clock(digital_control_module.digital_control.clock.T * 1e-2)
    vsgd = 400e-3
    vdd = 800e-3
    vgd = 0.0
    # 10000 control cycles
    t_stop = digital_control_module.digital_control.clock.T * 1e5
    sinusoidal = Sinusoidal(400e-3, 50e0, 0, vsgd)
    testbench = cbadc.circuit_level.testbench.TestBench(
        analog_frontend_module, sinusoidal, clock, t_stop, "my_testbench", vdd, vgd
    )
    path_here = './test_bench'
    if not os.path.exists(path_here):
        os.mkdir(path_here)
    testbench.to_file("this_file", path=path_here)
