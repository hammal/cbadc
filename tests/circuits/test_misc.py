from cbadc.circuit import Terminal
from cbadc.circuit.testbench import OpAmpTestBench
from cbadc.synthesis.leap_frog import get_leap_frog
from cbadc.analog_signal import Sinusoidal

two_terminals = [Terminal() for _ in range(2)]
three_terminals = two_terminals + [Terminal()]
N = 4
ENOB = 14
BW = 1e6
amplitude = 0
frequency = 1e3
delay = 1e-6
offset = 0.0
damping_factor = 0.0
phase = 0.0
vdd = 1.2
vgnd = 0.0
t_end = 1e-6


def test__dict__property():
    analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW)
    vdd = 1.8
    GBWP = 1e8
    DC_gain = 1e2
    testbench = OpAmpTestBench(
        analog_frontend,
        input_signals=[Sinusoidal(amplitude, frequency, phase, offset)],
        clock=analog_frontend.digital_control.clock,
        vdd_voltage=vdd,
        GBWP=GBWP,
        DC_gain=DC_gain,
    )

    testbench.__dict__['test'] = 1

    assert testbench.test == 1
