import numpy as np
from .digital_control import DigitalControl


class CalibrationControl(DigitalControl):
    def control_contribution(self, t: float, s_tilde: np.ndarray) -> np.ndarray:
        """Evaluates the control contribution at time t given a control observation
        s_tilde.

        Parameters
        ----------
        t : `float`
            time at which the digital control i evaluated.
        s_tilde : `array_like`, shape=(M_tilde,)
            state vector evaluated at time t

        Returns
        -------
        `array_like`, shape=(M,)
            the control signal :math:`\mathbf{s}(t)`

        """
        # Check if time t has passed the next control update
        if t >= self._t_next:
            # if so update the control signal state
            self._s[1:] = s_tilde[1:] >= 0
            # randomize first bit
            self._s[0] = np.random.randint(2)
            self._t_next += self.T
            # DAC
            self._dac_values = np.asarray(2 * self._s - 1, dtype=np.double)
        return self._dac_values
