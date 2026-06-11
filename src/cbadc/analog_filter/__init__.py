"""Analog systems

A selection of pre-configures and general analog system structures.
"""

from typing import Union

from . import analog_system, chain_of_integrators, filters, leap_frog, topology
from .analog_system import AnalogSystem, InvalidAnalogSystemError
from .chain_of_integrators import ChainOfIntegrators
from .filters import ButterWorth, Cauer, ChebyshevI, ChebyshevII, IIRDesign
from .leap_frog import LeapFrog
from .modulator import SineWaveModulator, SquareWaveModulator
from .topology import chain, sos2abcd, stack, tf2abcd, zpk2abcd

_valid_analog_filter_types = Union[AnalogSystem, ChainOfIntegrators, LeapFrog]
_valid_filter_types = Union[ButterWorth, ChebyshevI, ChebyshevII, IIRDesign]
