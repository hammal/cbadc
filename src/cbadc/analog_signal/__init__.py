"""Predefined common analog signals

This module focuses on representing analog signals, i.e., mappings from the
time :math:`t` to a signal value :math:`u(t)`. Typically for signal processing
algorithms, we are used to handeling discrete-time signals, i.e. samples of
signals. However, since the control-bounded A/D converters are converting
continuous-time signals we need tools to define signals that can be evaluated
over their whole continuous domain.
"""

from typing import Union
from ._analog_signal import _AnalogSignal
from .constant_signal import ConstantSignal
from .ramp import Ramp
from .sinc_pulse import SincPulse
from .sinusoidal import Sinusoidal
from .clock import Clock
from .impulse_responses import StepResponse, RCImpulseResponse
from .quadrature import get_quadrature_signal_pair, QuadratureSignal, SineWaveModulator
from .zero_order_hold import ZeroOrderHold, RaisedCosine, BandLimitedSignal
from .random import (
    BinaryReferenceSignal,
    TernaryReferenceSignal,
    GaussianReferenceSignal,
    UniformReferenceSignal,
)

from . import clock
from . import constant_signal
from . import impulse_responses
from . import ramp
from . import sinc_pulse
from . import sinusoidal
from . import quadrature
from . import zero_order_hold
from . import random

_valid_input_signal_types = Union[
    ConstantSignal,
    Ramp,
    SincPulse,
    Sinusoidal,
    Clock,
    ZeroOrderHold,
    RaisedCosine,
    BandLimitedSignal,
    QuadratureSignal,
    SineWaveModulator,
    BinaryReferenceSignal,
    TernaryReferenceSignal,
    GaussianReferenceSignal,
]

_valid_impulse_response_types = Union[StepResponse, RCImpulseResponse]

_valid_clock_types = Clock
