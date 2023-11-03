"""The default digital control."""
import numpy as np
from cbadc.analog_signal import StepResponse, _valid_clock_types, Clock
from cbadc.analog_signal.impulse_responses import _ImpulseResponse
from .digital_control import DigitalControl


def _rotation_matrix(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


class ModulatorControl(DigitalControl):
    """Represents a modulated digital control system.

    Parameters
    ----------
    clock : :py:class:`cbadc.analog_signal.clock.Clock`
        the clock to which the digital control synchronizes its updates.
    M : `int`
        number of controls.
    fc: `float`
        modulation frequency in Hz.
    fs: `float`, optional
        sampling frequency in Hz of output control signals. If not specified, fs = 2 * fc.
    t0 : `float`, `optional`
        determines initial time, defaults to 0.
    impulse_response : :py:class:`cbadc.analog_signal.AnalogSignal`, optional
        the digital control's impulse response.

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

    See also
    ---------
    :py:class:`cbadc.simulator.Simulator`
    """

    def __init__(
        self,
        clock: _valid_clock_types,
        M: int,
        fc: float,
        fs: float = None,
        t0: float = 0.0,
        impulse_response: _ImpulseResponse = StepResponse(),
    ):
        raise DeprecationWarning("ModulatorControl is deprecated.")
        if not isinstance(clock, Clock):
            raise Exception("Clock must derive from cbadc.analog_signal.Clock")
        self.fc = fc
        if fs is None:
            self.omega_fs = 2 * fc * np.pi
        else:
            self.omega_fs = fs * 2 * np.pi
        self.omega_c = 2 * np.pi * fc
        super().__init__(clock, M, t0, impulse_response)

    def control_update(self, t: float, s_tilde: np.ndarray):
        """Updates the control at time t if valid.

        Parameters
        ----------
        t : `float`
            time at which the digital control i evaluated.
        s_tilde : `array_like`, shape=(M_tilde,)
            state vector evaluated at time t
        """
        # print(f"check closeness ({t}, {self._t_next})")
        # Check if time t has passed the next control update
        if np.allclose(t, self._t_next, atol=self.clock._tt_2) or t > self._t_next:
            # Down-modulate the control decisions
            modulation_matrix = np.kron(
                _rotation_matrix(-self.omega_c * t), np.eye(self.M // 2)
            )
            s_tilde = np.dot(modulation_matrix, s_tilde)
            # Update the control decisions
            self._s = s_tilde >= 0
            self._t_last_update[:] = t
            self._t_next += self.clock.T
            # DAC
            self._control_decisions = np.asarray(2 * self._s - 1, dtype=np.double)
        # return self._dac_values * self._impulse_response(t - self._t_next + self.T)

    def event_list(self):
        """
        Return the event list of the digital control.

        Returns
        -------
        : [(t, x)->r]
            the list of event functions.
        """

        # The next control update
        control_update = TimeEvent(self._t_next, name="control_update", terminal=True)

        event_list = [control_update]

        # Add modulation ticks
        if self.clock.T > 1 / self.fc:
            for index in range(int(np.ceil(self.clock.T * self.fc * 2))):
                event_list.append(
                    TimeEvent(
                        index / self.fc / 2 + self._t_last_update[0],
                        name="modulation_frequency_tick",
                        terminal=False,
                    )
                )

        # The start of delayed impulse responses
        for m in range(self.M):
            response = TimeEvent(
                self._t_last_update[m] - self._impulse_response[m].t0,
                name=f"impulse_response_{m}",
                terminal=False,
            )
            event_list.append(response)

        return event_list

    def control_contribution(self, t: float) -> np.ndarray:
        """Evaluates the control contribution at time t.

        Parameters
        ----------
        t : `float`
            time at which the digital control i evaluated.

        Returns
        -------
        `array_like`, shape=(M,)
            the control signal :math:`\mathbf{s}(t)`

        """
        # Check if time t has passed the next control update
        impulse_response = np.zeros(self.M)

        modulation_matrix = np.kron(
            _rotation_matrix(self.omega_c * t), np.eye(self.M // 2)
        )
        # print(f"modulation matrix:\n{modulation_matrix}")
        for m in range(self.M):
            impulse_response[m] = self._impulse_response[m](t - self._t_last_update[m])
        modulated = np.dot(
            modulation_matrix, self._control_decisions * impulse_response
        )
        # print(modulated)
        return modulated

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

        modulation_matrix = np.kron(
            _rotation_matrix(self.omega_c * (t - self.clock.T)), np.eye(self.M // 2)
        )
        # modulation_matrix = np.kron(_rotation_matrix(self.omega_c * (t)), np.eye(self.M // 2))

        if t >= 0 and t <= self.clock.T:
            temp[m] = self._impulse_response[m](t)
        return np.dot(modulation_matrix, temp)
        # return temp

    def control_signal(self) -> np.ndarray:
        """Returns the current control state, i.e, :math:`\mathbf{s}[k]`.

        Returns
        -------
        `array_like`, shape=(M,), dtype=numpy.int8
            current control state.
        """
        kT = self._t_last_update[0]
        modulation_matrix = np.kron(
            _rotation_matrix(self.omega_fs * kT), np.eye(self.M // 2)
        )
        # return self._s[:]
        return np.dot(modulation_matrix, self._s)
