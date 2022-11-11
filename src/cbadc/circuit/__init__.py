"""modules relating to turning analog-frontends into Verilog-ams models."""

from cbadc.circuit import module
from cbadc.circuit import analog_frontend
from cbadc.circuit import op_amp
from cbadc.circuit import state_space_equations
from cbadc.circuit import testbench
from cbadc.circuit import noise_models

from cbadc.circuit.module import Module, Wire, Parameter
from cbadc.circuit.state_space_equations import (
    AnalogSystem as AnalogSystemStateSpaceEquations,
)
from cbadc.circuit.analog_frontend import AnalogFrontend
from cbadc.circuit.analog_system import (
    AnalogSystemFirstOrderPoleOpAmp,
    AnalogSystemIdealOpAmp,
    # AnalogSystemHigherOrderOpAmp,
)
from cbadc.circuit.digital_control import DigitalControl
from cbadc.circuit.testbench import TestBench, get_opamp_testbench, get_testbench
from cbadc.circuit.observer import Observer
