"""modules relating to turning analog-frontends into Verilog-ams models."""

from cbadc.circuit_level import module
from cbadc.circuit_level import analog_frontend
from cbadc.circuit_level import op_amp
from cbadc.circuit_level import state_space_equations
from cbadc.circuit_level import testbench
from cbadc.circuit_level import noise_models

from cbadc.circuit_level.module import Module, Wire, Parameter
from cbadc.circuit_level.state_space_equations import (
    AnalogSystem as AnalogSystemStateSpaceEquations,
)
from cbadc.circuit_level.analog_frontend import AnalogFrontend
from cbadc.circuit_level.analog_system import (
    AnalogSystemFirstOrderPoleOpAmp,
    AnalogSystemIdealOpAmp,
    # AnalogSystemHigherOrderOpAmp,
)
from cbadc.circuit_level.digital_control import DigitalControl
from cbadc.circuit_level.testbench import TestBench

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
