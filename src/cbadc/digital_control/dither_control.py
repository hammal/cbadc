"""the calibration control"""
from typing import List, Union
import numpy as np
from copy import copy
from . import DigitalControl
from .conservative_control import ConservativeControl
from .multi_phase_control import MultiPhaseDigitalControl
from .switch_capacitor_control import SwitchedCapacitorControl
from ..analog_signal.impulse_responses import _ImpulseResponse
from ..analog_signal import StepResponse


class DitherControl(DigitalControl):
    """A control with dither signal.

    An extension to the traditional digital controls,
    that stabilize the digital dimension, where the dither
    control adds a randomly generated dithering signal to
    the mix.

    Note that the dithering controls are prepended those
    of the initalizing digital control.

    Parameters
    ----------
    number_of_random_controls: `int`
        number of dithering controls
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        a digital control to be extended

    """

    def __init__(
        self,
        number_of_random_controls: int,
        digital_control: Union[
            DigitalControl,
            ConservativeControl,
            MultiPhaseDigitalControl,
            SwitchedCapacitorControl,
        ],
        t0: float = 0.0,
        impulse_response: _ImpulseResponse = StepResponse(),
        ternary: bool = False,
        index: int = 0,
    ):
        self._deterministic_control = copy(digital_control)
        self.number_of_random_control = number_of_random_controls
        self.clock = copy(digital_control.clock)
        self.M = digital_control.M + self.number_of_random_control
        self.M_tilde = self._deterministic_control.M_tilde
        self._t_next = t0
        self._t_last_update = self._t_next * np.ones(self.M)
        self._s = np.zeros(self.M, dtype=np.int8)
        self._s[:] = False
        self._control_decisions = np.zeros(self.M, dtype=np.double)
        self.ternary = bool(ternary)
        # initialize dac values

        self._deterministic_control_index: List[int] = []
        self._random_control_index: List[int] = []

        self._impulse_response = []

        for i in range(self.M):
            if i >= index and i < index + self.number_of_random_control:
                self._random_control_index.append(i)
                self._impulse_response.append(copy(impulse_response))
            else:
                self._deterministic_control_index.append(i)
                self._impulse_response.append(
                    copy(
                        self._deterministic_control._impulse_response[
                            len(self._deterministic_control_index) - 1
                        ]
                    )
                )

        self.control_update(
            self._t_next, np.zeros(self.M - self.number_of_random_control)
        )

    def jitter(self, t: float):
        "Jitter the phase by t"
        self._deterministic_control.jitter(t)

    def reset(self, t0: float = 0.0):
        """Reset the digital control clock

        Parameters
        ----------
        t0: `float`, `optional`
            time to set next update at, defaults to 0.
        """
        self._t_next = t0
        self._t_last_update = t0 * np.ones(self.M)
        self._s = np.zeros(self.M, dtype=np.int8)
        self._s[:] = False
        self._control_decisions = np.zeros(self.M, dtype=np.double)

        self._deterministic_control.reset(t0)

    def control_update(self, t: float, s_tilde: np.ndarray):
        """Updates the control at time t if valid.

        Parameters
        ----------
        t : `float`
            time at which the digital control i evaluated.
        s_tilde : `array_like`, shape=(M_tilde,)
            state vector evaluated at time t
        """
        # Deterministic controls
        self._deterministic_control.control_update(t, s_tilde)
        self._s[self._deterministic_control_index] = self._deterministic_control._s[:]
        # Check if time t has passed the next control update
        if np.allclose(t, self._t_next, atol=self.clock._tt_2) or t > self._t_next:
            # Random controls
            if self.ternary:
                self._s[self._random_control_index] = (
                    np.random.randint(3, size=(self.number_of_random_control,)) / 2.0
                )
            else:
                self._s[self._random_control_index] = np.random.randint(
                    2, size=(self.number_of_random_control,)
                )

            self._t_last_update[:] = t
            self._t_next = self._t_next + self.clock.T
        self._control_decisions = np.asarray(2 * self._s - 1, dtype=np.double)

    # def control_contribution(self, t: float) -> np.ndarray:
    #     """Evaluates the control contribution at time t.

    #     Parameters
    #     ----------
    #     t : `float`
    #         time at which the digital control i evaluated.

    #     Returns
    #     -------
    #     `array_like`, shape=(M,)
    #         the control signal :math:`\mathbf{s}(t)`

    #     """
    #     # Check if time t has passed the next control update
    #     impulse_response = np.zeros(self.M)
    #     impulse_response[
    #         self._deterministic_control_index
    #     ] = self._deterministic_control.control_contribution(t)
    #     for m in range(self.number_of_random_control):
    #         impulse_response[self._random_control_index[m]] = self._control_decisions[
    #             m
    #         ] * self._impulse_response[m](t - self._t_last_update[m])

    #     return impulse_response

    def impulse_response(self, m: int, t: float) -> np.ndarray:
        """The impulse response of the corresponding DAC waveform

        Parameters
        ----------
        m : `int`
            determines which :math:`m\in\{0,\dots,M-1\}` control dimension
            which is triggered.
        t : `float`
            evaluate the impulse response at time t.

        Returns
        -------
        `array_like`, shape=(M,)
            the dac waveform of the digital control system.

        """
        temp = np.zeros(self.M, dtype=np.double)
        temp[m] = self._impulse_response[m](t)
        return temp
