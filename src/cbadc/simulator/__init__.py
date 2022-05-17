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
    NonLinearSimulator,
)
from .analytical_simulator import AnalyticalSimulator
from .mp_simulator import MPSimulator
from .numpy_simulator import NumpySimulator
from .utilities import extended_simulation_result
from .wrapper import get_simulator, SimulatorType


_valid_simulators = Union[
    FullSimulator, PreComputedControlSignalsSimulator, AnalyticalSimulator, MPSimulator
]
