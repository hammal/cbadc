"""modules relating to turning analog-frontends into Verilog-ams models."""

from . import module
from . import analog_frontend
from . import op_amp
from . import state_space_equations
from . import testbench
from . import noise_models

from .module import Module, Wire, Parameter
from .state_space_equations import AnalogSystem as AnalogSystemStateSpaceEquations
from .analog_frontend import AnalogFrontend
from .op_amp import (
    AnalogSystemFiniteGainOpAmp,
    AnalogSystemFirstOrderPoleOpAmp,
    AnalogSystemIdealOpAmp,
    AnalogSystemStateSpaceOpAmp,
)
from .digital_control import DigitalControl
from .testbench import TestBench

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
