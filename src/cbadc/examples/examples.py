"""Control-bounded ADC examples

This module provides several control-bounded converter examples.
Many of them are directly taken from, or inspired by, the examples
provided in the PhD dissertation
`Control-Bounded Converters <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf>`_.

These examples servers not only as inspiration but also for easily
benchmarking. Furthermore, since simulating is significantly slower than
digital estimation, these examples also provide large pre-simulated control
signal sequences thus circumventing the time consuming process of long
simulations.
"""
import numpy as np
from math import log2
import scipy.linalg
from cbadc.digital_control import DigitalControl
from cbadc.analog_system import AnalogSystem
from typing import Dict
import cbadc
import requests
import json


class PreSimulation:

    def __init__(self, **kwargs):
        self.analog_system = cbadc.analog_system.AnalogSystem(**kwargs)
        self.digital_control = cbadc.digital_control.DigitalControl(**kwargs)
        self.simulator = cbadc.simulator.StateSpaceSimulator(**kwargs)

    def parameters_to_json(self):
        return json.dump({
            'beta_vector': self.beta_vector,
            'rho_vector': self.rho_vector,
            'kappa_vector': self.kappa_vector,
            'T': self.T,
            'N': self.N,
            'M': self.M
        })


def load_simulation(configuration: Dict):
    """Load a pre-computed simulation from configuration dict.

    Parameters
    ----------
    configuration: { description: str, parameters: str, control_signal_sequence: Union[str, list], class: Class}
        a configuration dictionary containing the necessary information
        to load and configure a previous simulation

    Returns
    -------
    description: str
        simulation description.
    parameters: Dict
        system parameters.
    control_signal_sequence: Iterator[ndarray, None, None]
        a control_signal_sequence.
    system: Union[ChainOfIntegrators]
        an instantiated example object.
    """
    if 'description' not in configuration:
        raise BaseException("configuration must have description")
    if 'parameters' not in configuration:
        raise BaseException("configuration must have parameters")
    if 'control_signal_sequence' not in configuration:
        raise BaseException("configuration must have control_signal_sequence")
    description = configuration['description']
    # retrive and load parameters
    params = requests.get(configuration['parameters']).json()
    # Instantiate class with parameters
    system = PreSimulation(**params)
    M = system["M"]
    control_signal_sequence = cbadc.utilities.byte_stream_2_control_signal(
        cbadc.utilities.read_byte_stream_from_url(
            configuration['control_signal_sequence'], M), M)
    return description, params, control_signal_sequence, system


# A dictionary containing all pre computed simulations
pre_computed_simulations = {
    'chain-of-integrators': {
        '32OSRSineFSInput': {
            "description": "A sinusoid input signal at 1/256 T",
            "parameters": "https://adfjkhaskf.json",
            "control_signal_sequence": ["https://alfkjdaslkj.adcs"],
        }
    },
    'leap-frog': {}
}
