from .digital_control import (
    DigitalControl,
    _ImpulseResponse,
    StepResponse,
    _valid_clock_types,
)
from typing import List
import numpy as np


class MultiLevelDigitalControl(DigitalControl):
    """Multi-level digital control system.

    Parameters
    ----------
    clock : :py:class:`cbadc.analog_signal.clock.Clock`
        the clock to which the digital control synchronizes its updates.
    M : `int`
        number of controls.
    number_of_levels: `[int]`
        number of levels for each of the M quantizers.
    t0 : `float`, `optional`
        determines initial time, defaults to 0.
    impulse_response : :py:class:`cbadc.analog_signal.AnalogSignal`, optional
        the digital control's impulse response.
    offsets: [float], `optional`
        a M sized list with offsets for each control, defaults to all 0.

    Attributes
    ----------
    clock : :py:class:`cbadc.analog_signal.clock.Clock`
        the digital control system clock.
    M : `int`
        number of controls :math:`M`.
    M_tilde : `int`
        number of control observations :math:`\\tilde{M}`.

    Note
    ----
    For this digital control system :math:`M=\\tilde{M}`.
    Also, the number of levels should equal M_tilde.
    """

    def __init__(
        self,
        clock: _valid_clock_types,
        M: int,
        number_of_levels: List[int],
        t0: float = 0.0,
        impulse_response: _ImpulseResponse = StepResponse(),
        offsets: List[float] = [],
    ):
        self.number_of_levels = number_of_levels
        self._levels = []
        self._references = []

        if not offsets:
            self.offsets = np.zeros(M)
        elif offsets and len(offsets) == M:
            self.offsets = np.array(offsets)
        else:
            raise Exception("offsets must be empty or M sized list of floats.")
        if len(number_of_levels) != M:
            raise Exception("Must have M number of levels")

        for m in range(M):
            smallest_step = 1.0 / number_of_levels[m]
            self._levels.append(
                np.linspace(-1 + smallest_step, 1 - smallest_step, number_of_levels[m])
            )
            self._references.append(np.linspace(0, 1, number_of_levels[m] + 1))
        super().__init__(clock, M, t0, impulse_response)
        self._s = np.zeros(self.M, dtype=np.double)
        self.control_update(self._t_next, np.zeros(self.M))

    def control_update(self, t: float, s_tilde: np.ndarray):
        """Updates the control at time t if valid.

        Parameters
        ----------
        t : `float`
            time at which the digital control i evaluated.
        s_tilde : `array_like`, shape=(M_tilde,)
            state vector evaluated at time t
        """
        if np.allclose(t, self._t_next, atol=self.clock._tt_2) or t > self._t_next:
            for m_tilde in range(self.M_tilde):
                index_mask = s_tilde[m_tilde] >= self._levels[m_tilde]
                for level in range(self.number_of_levels[m_tilde]):
                    if not index_mask[level]:
                        self._s[m_tilde] = self._references[m_tilde][level]
                        break
                    elif level == (self.number_of_levels[m_tilde] - 1):
                        self._s[m_tilde] = self._references[m_tilde][-1]

            self._t_last_update[:] = t
            self._t_next += self.clock.T
            self._control_descisions = (
                np.asarray(2.0 * self._s - 1.0, dtype=np.double) + self.offsets
            )
