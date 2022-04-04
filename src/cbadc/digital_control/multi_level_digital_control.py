from .digital_control import (
    DigitalControl,
    _ImpulseResponse,
    StepResponse,
    _valid_clock_types,
)
from typing import List
import numpy as np


class MultiLevelDigitalControl(DigitalControl):
    """
    Number of levels should equal M_tilde!
    """

    def __init__(
        self,
        clock: _valid_clock_types,
        M: int,
        number_of_levels: List[int],
        t0: float = 0.0,
        impulse_response: _ImpulseResponse = StepResponse(),
    ):
        self.number_of_levels = number_of_levels
        self._levels = []
        self._references = []
        for m in range(M):
            smallest_step = 1.0 / number_of_levels[m]
            self._levels.append(
                np.linspace(-1 + smallest_step, 1 - smallest_step, number_of_levels[m])
            )
            self._references.append(np.linspace(0, 1, number_of_levels[m] + 1))
        super().__init__(clock, M, t0, impulse_response)
        self._s = np.zeros(self.M, dtype=np.double)
        self.control_update(self._t_next, np.zeros(self.M))
        print("initial digital control", self._s, self.control_contribution(0))

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
            self._control_descisions = np.asarray(2.0 * self._s - 1.0, dtype=np.double)
