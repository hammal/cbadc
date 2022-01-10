import numpy as np
from ..analog_signal.impulse_responses import StepResponse
from ..analog_signal import _valid_clock_types
from .digital_control import DigitalControl


class MultiPhaseDigitalControl(DigitalControl):
    """Represents a digital control system that switches controls individually
    sequentially.

    This digital control updates the :math:`m`-th control signals as

    :math:`s_m[k] = \\tilde{s}((k+m)T)`

    except for this it works similarly to
    :py:class`cbadc.digital_control.DigitalControl`

    Parameters
    ----------
    T : `float`
        clock period at which the digital control updates.
    T1 : `array_like`, shape=(M,)
        time at which the digital control empties the capacitor into the
        system.
    t_tolerance: `float`, `optional`
        determines with which time tolerance the digital control
        updated, defaults to 1e-12.
    t0 : `float`: optional
        determines initial time, defaults to 0.

    Attributes
    ----------
    T : `float`
        clock period :math:`T` of digital control system.
    T1 : `array_like`, shape=(M,)
        discharge phase time
    M : `int`
        number of controls :math:`M`.
    M_tilde : `int`
        number of control observations :math:`\\tilde{M}`.

    Note
    ----
    For this digital control system :math:`M=\\tilde{M}`.
    """

    def __init__(
        self,
        clock: _valid_clock_types,
        phi_1: np.ndarray,
        t0: float = 0,
        t_tolerance=1e-12,
        impulse_response=None,
    ):
        M = phi_1.size
        self._phi_1 = phi_1
        self._t_next = t0
        self._t_next_phase = self._t_next + self._phi_1
        self._t_last_update = t0 * np.ones(M)
        self._t_tolerance = t_tolerance
        DigitalControl.__init__(self, clock, M, t0=t0)
        if (phi_1 < 0).any() or (phi_1 > self.clock.T).any():
            raise Exception(f"Invalid phi_1 ={phi_1}")

        if impulse_response is not None:
            if len(impulse_response) != self.M:
                raise Exception("must be M impulse responses for M phases")
            self._impulse_response = impulse_response
        else:
            self._impulse_response = [StepResponse() for _ in range(self.M)]

        self._dac_values = np.zeros(self.M, dtype=np.double)
        # initialize dac values
        self.control_update(self._t_next, np.zeros(self.M))

    def _next_update(self):
        t_next = np.inf
        for m in range(self.M):
            if self._t_next_phase[m] < t_next:
                t_next = self._t_next_phase[m]
        return t_next

    def jitter(self, t: float):
        "Jitter the phase by t"
        for m in range(self.M):
            self._t_next_phase[m] += t
        self._t_next = self._next_update()

    def control_update(self, t: float, s_tilde: np.ndarray):
        """Updates the control descisions at time t given a control observation
        s_tilde.

        Parameters
        ----------
        t : `float`
            time at which the digital control i evaluated.
        s_tilde : `array_like`, shape=(M_tilde,)
            state vector evaluated at time t

        """
        # Check if time t has passed the next control update
        if t >= self._t_next:
            for m in range(self.M):
                if t >= self._t_next_phase[m]:
                    # if so update the control signal state
                    self._s[m] = s_tilde[m] >= 0
                    self._t_next_phase[m] += self.clock.T
                    self._t_next = self._next_update()
                    self._t_last_update[m] = t
                    # DAC
                    self._dac_values = np.asarray(2 * self._s - 1, dtype=np.double)
                    # print(f"m = {m}, t = {t}, s = {self._dac_values}")

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
        if t >= 0 and t <= self.clock.T:
            temp[m] = self._impulse_response[m](t - self._phi_1[m])
        return temp
