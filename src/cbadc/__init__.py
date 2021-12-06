"""
The control-bounded converter toolbox allows simulation and reconstruction
of control-bounded converters.
"""
import logging
from importlib.metadata import version
from . import analog_signal
from . import analog_system
from . import digital_control
from . import digital_estimator
from . import simulator
from . import utilities

# Set logging level
logging.basicConfig(level=logging.INFO)

# Set version variable
__version__ = version("cbadc")
