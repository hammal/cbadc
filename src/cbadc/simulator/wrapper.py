from typing import List
import cbadc
import logging
import math
import enum

from cbadc.simulator import (
    FullSimulator,
    PreComputedControlSignalsSimulator,
    AnalyticalSimulator,
    MPSimulator,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimulatorType(enum.Enum):
    full_numerical = 1
    pre_computed_numerical = 2
    analytical = 3
    mpmath = 4


def get_simulator(
    analog_system: cbadc.analog_system._valid_analog_system_types,
    digital_control: cbadc.digital_control._valid_digital_control_types,
    input_signal: List[cbadc.analog_signal._valid_input_signal_types],
    clock: cbadc.analog_signal._valid_clock_types = None,
    t_stop: float = math.inf,
    initial_state_vector=None,
    atol: float = 1e-9,
    rtol: float = 1e-6,
    simulator_type: SimulatorType = None,
):
    if isinstance(simulator_type, SimulatorType):
        simulator_type = simulator_type
    else:
        simulator_type = SimulatorType.pre_computed_numerical

    if SimulatorType.full_numerical == simulator_type:
        logger.info("FullSimulator used for simulation.")
        return FullSimulator(
            analog_system,
            digital_control,
            input_signal,
            clock,
            t_stop,
            initial_state_vector,
            atol,
            rtol,
        )
    if SimulatorType.pre_computed_numerical == simulator_type:
        logger.info("PreComputedControlSignalSimulator used for simulation.")
        return PreComputedControlSignalsSimulator(
            analog_system,
            digital_control,
            input_signal,
            clock,
            t_stop,
            initial_state_vector,
            atol,
            rtol,
        )
    if SimulatorType.analytical == simulator_type:
        logger.info("AnalyticalSimulator used for simulation.")
        return AnalyticalSimulator(
            analog_system,
            digital_control,
            input_signal,
            clock,
            t_stop,
            initial_state_vector,
        )
    if SimulatorType.mpmath == simulator_type:
        logger.info("MPSimulator used for simulation.")
        return MPSimulator(
            analog_system,
            digital_control,
            input_signal,
            clock,
            t_stop,
            initial_state_vector,
            tol=min(atol, rtol),
        )
    raise NotImplementedError
