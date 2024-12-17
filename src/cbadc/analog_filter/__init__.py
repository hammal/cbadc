"""Analog systems

A selection of pre-configures and general analog system structures.
"""

from typing import Union
from .analog_system import AnalogSystem, InvalidAnalogSystemError
from .chain_of_integrators import ChainOfIntegrators
from .leap_frog import LeapFrog
from .filters import ButterWorth, Cauer, ChebyshevI, ChebyshevII, IIRDesign
from .topology import chain, stack, sos2abcd, tf2abcd, zpk2abcd
from .modulator import SineWaveModulator, SquareWaveModulator

from . import analog_system
from . import chain_of_integrators
from . import filters
from . import leap_frog
from . import topology

_valid_analog_filter_types = Union[AnalogSystem, ChainOfIntegrators, LeapFrog]
_valid_filter_types = Union[ButterWorth, ChebyshevI, ChebyshevII, IIRDesign]
