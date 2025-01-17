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
from .__version__ import __version__

from . import analog_signal
from . import analog_filter
from . import digital_control
from . import analog_frontend
from . import digital_backend

# from . import digital_estimator
from . import utilities
from . import delsig

# from . import circuit
from . import fom

# Commonly used classes
from .analog_frontend import AnalogFrontend, GmC, ActiveRC
from .analog_signal import AnalogSignal, Sinusoidal, ZeroOrderHold
from .digital_control import DigitalControl
from .digital_backend import WienerFilter, AdaptiveFIRFilter
