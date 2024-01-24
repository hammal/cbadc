"""the calibration control"""
from typing import List, Union
import numpy as np
from copy import copy
from . import DigitalControl
from cbadc.analog_signal.impulse_responses import StepResponse
from ..analog_signal.impulse_responses import _ImpulseResponse


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
        digital_control: DigitalControl,
        impulse_response: List[_ImpulseResponse] = None,
        dithering=True,
        random_sequence_length=1 << 22,
        t_delay: float = 0.0,
    ):
        if t_delay < 0:
            raise Exception("t_delay must be non-negative")
        else:
            self.t_delay = t_delay
        self._deterministic_control = copy(digital_control)
        self.number_of_random_control = number_of_random_controls
        self.clock = copy(digital_control.clock)
        self.M = digital_control.M + self.number_of_random_control
        self.M_tilde = self._deterministic_control.M_tilde
        self._s = np.zeros(self.M)
        self._s[:] = False
        self._control_decisions = np.zeros(self.M, dtype=np.double)
        # initialize dac values

        self.dithering = dithering

        rng = np.random.default_rng()
        self._pseudo_random_sequence = rng.integers(
            2, size=(random_sequence_length, number_of_random_controls)
        )
        # self._pseudo_random_sequence = rng.normal(
        #     0.5, 0.1, size=(random_sequence_length, number_of_random_controls)
        # )

        self._pseudo_random_index = 0
        self._pseudo_random_sequence_size = random_sequence_length

        self._initialize_impulse_response(digital_control.clock, impulse_response)

    def turn_on_dither(self):
        self.dithering = True

    def turn_off_dither(self):
        self.dithering = False

    def _initialize_impulse_response(self, clock, impulse_response):
        self._random_control_index = [i for i in range(self.number_of_random_control)]
        self._deterministic_control_index: List[int] = []
        self._impulse_response = []
        if impulse_response is not None:
            if len(impulse_response) != self.number_of_random_control:
                raise Exception(
                    "must be number_of_random_control speciefied impulse responses"
                )
            t0s = np.array([x.t0 for x in impulse_response])
            if (
                (t0s < 0.0).any()
                or (t0s > self.clock.T).any()
                or (t0s[:-1] > t0s[1:]).any()
            ):
                raise Exception("Invalid impulse respones")

            self._mulit_phase = (
                np.sum(t0s) > 0.0 or self._deterministic_control._mulit_phase
            )

            for i in range(self.number_of_random_control):
                self._impulse_response.append(copy(impulse_response[i]))
        else:
            for i in range(self.number_of_random_control):
                self._impulse_response.append(StepResponse(0.0))
            self._mulit_phase = self._deterministic_control._mulit_phase

        for i in range(self._deterministic_control.M):
            self._deterministic_control_index.append(i + self.number_of_random_control)
            self._impulse_response.append(
                copy(self._deterministic_control._impulse_response[i])
            )
        self._setup_clock_phases()

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
        for i, m in enumerate(self._deterministic_control_index):
            self._s[m] = self._deterministic_control._s[i]
            # DAC
            self._control_decisions[m] = 2.0 * self._s[m] - 1.0

        # Random controls
        for m in self._random_control_index:
            if np.allclose(t, self._impulse_response[m].t0, atol=self.clock._tt_2):
                if self.dithering:
                    self._s[m] = self._pseudo_random_sequence[
                        self._pseudo_random_index, m
                    ]
                    self._pseudo_random_index = (
                        self._pseudo_random_index + 1
                    ) % self._pseudo_random_sequence_size
                else:
                    self._s[m] = 0.5
                # DAC
                self._control_decisions[m] = 2.0 * self._s[m] - 1.0

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
