"""Digital controls

This module provides a general :py:class:`cbadc.digital_control.DigitalControl`
class to enabled the creation of a general independently controlled digital
control system.
"""
from typing import Union
from .digital_control import DigitalControl
from .multi_level_digital_control import MultiLevelDigitalControl
from .dither_control import DitherControl
from .utilities import overcomplete_set, unit_element_set

from . import digital_control

# from .switch_capacitor_control import SwitchedCapacitorControl

_valid_digital_control_types = Union[DigitalControl, MultiLevelDigitalControl]
