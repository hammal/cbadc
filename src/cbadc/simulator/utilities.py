"""Simulation utilities."""
from typing import Dict, Generator
import numpy as np
from ._base_simulator import _BaseSimulator


def extended_simulation_result(
    simulator: _BaseSimulator,
) -> Generator[Dict[str, np.ndarray], None, None]:
    """Extended simulation output

    Used to also pass the state vector from a
    simulator generator.

    Parameters
    ----------
    simulator : :py:class:`cbadc.simulator.StateSpaceSimulator`
        a iterable simulator instance.

    Yields
    ------
    { 'control_signal', 'analog_state', 't' } : { (array_like, shape=(M,)), (array_like, shape=(N,)) }
        an extended output including the analog state vector.
    """
    for control_signal in simulator:
        analog_state = simulator.state_vector()
        yield {
            "control_signal": np.array(control_signal),
            "analog_state": np.array(analog_state),
            "t": simulator.t,
        }
