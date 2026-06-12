"""The control-bounded converter toolbox.

This module provides a set of tools for the design and analysis of
continuous-time and discrete-time analog-to-digital converters (ADCs).

In particular, this module targets the high-to-medium level of abstraction
where the we investigate the performance of...

Note
----
The module is still under development and the API is subject to change.

"""

# Set logging level
import logging as _logging

_logging.basicConfig(level=_logging.INFO)

# Set version variable
# from . import digital_estimator
# from . import circuit
from . import (
    analog_filter,
    analog_frontend,
    analog_signal,
    delsig,
    digital_backend,
    digital_control,
    fom,
    noise,
    snr,
    utilities,
)
from .__version__ import __version__

# Commonly used classes
from .analog_frontend import ActiveRC, AnalogFrontend, GmC
from .analog_signal import AnalogSignal, Sinusoidal, ZeroOrderHold
from .digital_backend import AdaptiveFIRFilter, DataAidedEstimator, WienerFilter
from .digital_control import DigitalControl
