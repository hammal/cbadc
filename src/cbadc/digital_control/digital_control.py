import numpy as np
from ..analog_signal import StepResponse, _valid_clock_types
from ..analog_signal.impulse_responses import _ImpulseResponse


class DigitalControl:
    """Represents a digital control system.

    This is the simplest digital control where
    :math:`M=\\tilde{M}` and each control signal is updated
    independently. Furthermore, the DAC waveform is a constant signal
    as :math:`\mathbf{s}(t)=\mathbf{s}[k]` for :math:`t\in[k T, (k+1)T)`.

    Parameters
    ----------
    clock : :py:class:`cbadc.analog_signal.clock.Clock`
        the clock to which the digital control synchronizes its updates.
    M : `int`
        number of controls.
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

    Examples
    --------
    >>> from cbadc.digital_control import DigitalControl
    >>> from cbadc.analog_signal import Clock
    >>> T = 1e-6
    >>> clock = Clock(T)
    >>> M = 4
    >>> dc = DigitalControl(clock, M)
    >>> print(dc)
    The Digital Control is parameterized as:
    T = 1e-06,
    M = 4, and next update at
    t = 1e-06
    """

    def __init__(
        self,
        clock: _valid_clock_types,
        M: int,
        t0: float = 0.0,
        impulse_response: _ImpulseResponse = StepResponse(),
    ):
        self.clock = clock
        self.M = M
        self.M_tilde = M
        self._t_next = t0
        self._t_last_update = self._t_next * np.ones(self.M)
        self._s = np.zeros(self.M, dtype=np.int8)
        self._s[:] = False
        self._impulse_response = [impulse_response for _ in range(self.M)]
        self._control_descisions = np.zeros(self.M, dtype=np.double)
        # initialize dac values
        self.control_update(self._t_next, np.zeros(self.M))

    def jitter(self, t: float):
        "Jitter the phase by t"
        self._t_next += t

    def control_update(self, t: float, s_tilde: np.ndarray):
        """Updates the control at time t if valid.

        Parameters
        ----------
        t : `float`
            time at which the digital control i evaluated.
        s_tilde : `array_like`, shape=(M_tilde,)
            state vector evaluated at time t
        """
        # Check if time t has passed the next control update
        if np.allclose(t, self._t_next, atol=self.clock._tt_2):
            # if so update the control signal state
            # print(f"digital_control set for {t} and {s_tilde}")
            self._s = s_tilde >= 0
            self._t_last_update[:] = t
            self._t_next += self.clock.T
            # DAC
            self._control_descisions = np.asarray(2 * self._s - 1, dtype=np.double)
        # return self._dac_values * self._impulse_response(t - self._t_next + self.T)

    def control_signal(self) -> np.ndarray:
        """Returns the current control state, i.e, :math:`\mathbf{s}[k]`.

        Examples
        --------
        >>> from cbadc.digital_control import DigitalControl
        >>> import numpy as np
        >>> T = 1e-6
        >>> M = 4
        >>> dc = DigitalControl(T, M)
        >>> _ = dc.control_contribution(T, np.array([-0.1, -0.2, 0.3, 99]))
        >>> res = dc.control_signal()
        >>> print(np.array(res))
        [0 0 1 1]


        Returns
        -------
        `array_like`, shape=(M,), dtype=numpy.int8
            current control state.
        """
        return self._s[:]

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
        for m in range(self.M):
            impulse_response[m] = self._impulse_response[m](t - self._t_last_update[m])
        return self._control_descisions * impulse_response

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
            temp[m] = self._impulse_response[m](t)
        return temp

    def __str__(self):
        return f"""{80 * '='}

The Digital Control is parameterized as:

{80 * '-'}

clock:
{self.clock}

M:
{self.M}
{80 * '='}
        """
