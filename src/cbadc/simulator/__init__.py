"""Analog frontend simulators

This module provides simulator tools to simulate the hardware
interaction between an analog system and digital control.
These are mainly intended to produce control signals
:math:`\mathbf{s}[k]` and evaluate state vector trajectories
:math:`\mathbf{x}(t)` for various Analog system
:py:class:`cbadc.analog_system.AnalogSystem` and
:py:class:`cbadc.digital_control.DigitalControl` interactions.
"""

from typing import Union
from .numerical_simulator import (
    FullSimulator,
    PreComputedControlSignalsSimulator,
)
from .numerical_simulator import PreComputedControlSignalsSimulator as Simulator
from .numpy_simulator import NumpySimulator
from .utilities import extended_simulation_result

from ..simulation_event import SimulationEvent, TimeEvent, OutOfBounds

_valid_simulators = Union[FullSimulator, PreComputedControlSignalsSimulator]
