"""Digital controls

This module provides a general :py:class:`cbadc.digital_control.DigitalControl`
class to enabled the creation of a general independently controlled digital
control system.
"""
from typing import Union
from .digital_control import DigitalControl
from .multi_level_digital_control import MultiLevelDigitalControl
from .multi_phase_control import MultiPhaseDigitalControl
from .conservative_control import ConservativeControl
from .utilities import overcomplete_set, unit_element_set
from .dither_control import DitherControl
from .modulator import ModulatorControl

from . import calibration_control
from . import conservative_control
from . import digital_control
from . import dither_control
from . import multi_level_digital_control
from . import multi_phase_control
from . import modulator

# from .switch_capacitor_control import SwitchedCapacitorControl

_valid_digital_control_types = Union[
    DigitalControl,
    MultiPhaseDigitalControl,
    ConservativeControl,
    MultiLevelDigitalControl,
    DitherControl,
    ModulatorControl,
]
