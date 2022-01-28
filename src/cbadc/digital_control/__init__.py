"""Digital controls

This module provides a general :py:class:`cbadc.digital_control.DigitalControl`
class to enabled the creation of a general independently controlled digital
control system.
"""
from typing import Union
from .digital_control import DigitalControl
from .multi_phase_control import MultiPhaseDigitalControl
from .conservative_control import ConservativeControl
from .utilities import overcomplete_set, unit_element_set

_valid_digital_control_types = Union[
    DigitalControl, MultiPhaseDigitalControl, ConservativeControl
]
