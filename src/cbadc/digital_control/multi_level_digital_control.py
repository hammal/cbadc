from cbadc.digital_control.digital_control import (
    DigitalControl,
    _ImpulseResponse,
    _valid_clock_types,
)
from cbadc.analog_signal import StepResponse
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
        impulse_response: List[_ImpulseResponse] = None,
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
        super().__init__(clock, M, impulse_response)
        self._s = np.zeros(self.M, dtype=np.double)

    def control_update(self, t: float, s_tilde: np.ndarray):
        """Updates the control at time t if valid.

        Parameters
        ----------
        t : `float`
            time at which the digital control i evaluated.
        s_tilde : `array_like`, shape=(M_tilde,)
            state vector evaluated at time t
        """
        for m_tilde in range(self.M_tilde):
            if np.allclose(
                t, self._impulse_response[m_tilde].t0, atol=self.clock._tt_2
            ):
                # quantize
                self._s[m_tilde] = self.quantize(m_tilde, s_tilde[m_tilde])
                # update control decisions
                self._control_decisions[m_tilde] = (
                    np.asarray(2.0 * self._s[m_tilde] - 1.0, dtype=np.double)
                    + self.offsets[m_tilde]
                )

    def quantize(self, m_tilde: int, s_tilde_m: np.array):
        """Quantizes the state vector.

        Parameters
        ----------
        m_tilde : `int`
            index of the control.
        s_tilde_m : `array_like`, shape=(M_tilde,)
            state vector evaluated at time t

        Returns
        -------
        `int`
            quantized state vector.
        """
        index_mask = s_tilde_m >= self._levels[m_tilde]
        for level in range(self.number_of_levels[m_tilde]):
            if not index_mask[level]:
                return self._references[m_tilde][level]
            elif level == (self.number_of_levels[m_tilde] - 1):
                return self._references[m_tilde][-1]
