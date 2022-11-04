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
from cbadc.circuit.testbench import TestBench

# TODO
#
# - OP-amp analog_system
# - switched-cap control
# - gm-C analog_system
# - get_simulator_function
# - get_estimator_function
# - get_analog_system, digital_control
# - Spice testbench
# - Documentation
# - Tutorial
#
