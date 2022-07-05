"""Simulation event tools"""
from typing import List
import numpy as np


class SimulationEvent:
    """The base class for simulation events.

    The simulation event class implements functionality to
    inform the simulator of certain events related to time and or
    state values.

    This is done by calling the internal call method with the
    current time and state upon which a float value is returned.
    The event is symbolized by the zero crossing of the call function.

    Parameters
    ----------
    name: `str`
        the name of an event.
    terminal: `bool`
        if the event should terminate the simulation, defaults to False.
    direction: `float`
        the direction of the zero crossing to trigger, defaults to 1 which is from negative to positive.
    """

    def __init__(self, name: str, terminal: bool = False, direction: float = 1):
        self.name = name
        self.terminal = terminal
        self.direction = direction

    def __call__(self, t: float, x: np.ndarray) -> float:
        return self.event_trigger(t, x)

    def event_trigger(self, t: float, x: np.ndarray) -> float:
        """The event trigger function

        evaluates the closeness to the event by finding a zero crossing
        of this function.

        See also
        ---------
        :py:class:`cbadc.simulator.OutOfBounds`
        :py:class:`cbadc.simulator.TimeEvent`

        Parameters
        ----------
        t: `float`
            the current time of evaluation
        x: `array_like`,
            the current state vector

        Note
        ----
        This is a base class and the given event trigger function will
        never produce a zero crossing, thus never trigger an actual event.

        Returns
        -------
        : `float`
            the current event trigger value, event is identified by a zero crossing
            of this function. In this class this event never triggers as 1.0 is always
            returned.
        """
        return 1.0


class OutOfBounds(SimulationEvent):
    """Out-of-bounds simulation event


    This event triggers when an out-of-bounds
    event occur

    Parameters
    ----------
    bound: `float`
        the bound for which, when state exceeds this, an event will be triggered.
    dim: `int`
        the state dimension that the bound applies to.
    name: `str`
        the name of an event.
    terminal: `bool`
        if the event should terminate the simulation, defaults to False.
    direction: `float`
        the direction of the zero crossing to trigger, defaults to 1 which is from negative to positive.
    """

    def __init__(self, bound: float, dim: int, *args, **kwargs):
        self.bound = bound
        self.dim = dim
        super().__init__(*args, **kwargs)

    def event_trigger(self, t: float, x: np.ndarray) -> float:
        """The event trigger function

        evaluates the closeness to the event by finding a zero crossing
        of this function.

        Parameters
        ----------
        t: `float`
            the current time of evaluation
        x: `array_like`,
            the current state vector

        Returns
        -------
        : `float`
            the current event trigger value, event is identified by a zero crossing
            of this function.

        """
        return x[self.dim] - self.bound


def out_of_bounds_factory(
    N: int,
    state_bounds: np.ndarray,
    name: str = 'x',
    terminal=False,
) -> List[SimulationEvent]:
    """
    a factory function for generating out-ouf-bounds events.

    Parameters
    ----------
    N: `int`
        number of state dimensions
    state_bounds: `array_like`, shape=(N, 2)
        the state bounds where for each dimension a lower followed by an upper
        bound are states, for example state_bounds = numpy.array([[x1_low, x1_high], [x2_low, x2_high], ... ])
    name: `str`
        the parent name of an event, defaults to x.
    terminal: `bool`
        if the event should terminate the simulation, defaults to False.
    """
    event_list = []
    for n in range(N):
        # Lower bound
        event_list.append(
            OutOfBounds(
                bound=state_bounds[n, 0],
                dim=n,
                name=f"ofb_{name}_{n}",
                terminal=terminal,
                direction=-1,
            )
        )
        # Upper bound
        event_list.append(
            OutOfBounds(
                bound=state_bounds[n, 1],
                dim=n,
                name=f"ofb_{name}_{n}",
                terminal=terminal,
                direction=1,
            )
        )
    return event_list


class TimeEvent(SimulationEvent):
    """time-event simulation event


    This event triggers at the specified time.

    Parameters
    ----------
    t: `float`
        the time at which the event triggers.
    name: `str`
        the name of an event.
    terminal: `bool`
        if the event should terminate the simulation, defaults to False.
    """

    def __init__(self, t: float, *args, **kwargs):
        self.t = t
        if 'direction' in kwargs:
            raise Exception(
                "time events can only be of positive direction as time progresses in a non-decreasing way."
            )
        super().__init__(*args, direction=1, **kwargs)

    def event_trigger(self, t: float, x: np.ndarray) -> float:
        """The event trigger function

        evaluates the closeness to the event by finding a zero crossing
        of this function.

        See also
        ---------
        :py:class:`cbadc.simulator.OutOfBounds`
        :py:class:`cbadc.simulator.TimeEvent`

        Parameters
        ----------
        t: `float`
            the current time of evaluation
        x: `array_like`,
            the current state vector

        Returns
        -------
        : `float`
            the current event trigger value, event is identified by a zero crossing
            of this function.

        """
        return t - self.t
