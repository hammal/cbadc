"""The default digital control."""
from typing import List
import numpy as np
from cbadc.analog_signal import _valid_clock_types, Clock
from cbadc.analog_signal.impulse_responses import StepResponse
from cbadc.analog_signal.impulse_responses import _ImpulseResponse


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
    """

    def __init__(
        self,
        clock: _valid_clock_types,
        M: int,
        impulse_response: List[_ImpulseResponse] = None,
        t_delay: float = 0.0,
    ):
        if not isinstance(clock, Clock):
            raise Exception("Clock must derive from cbadc.analog_signal.Clock")
        self.clock = clock
        self.M = M
        self.M_tilde = M
        if t_delay < 0:
            raise Exception("t_delay must be non-negative")
        else:
            self.t_delay = t_delay
        self._s = np.zeros(self.M, dtype=float)
        self._s[:] = False
        self._initialize_impulse_response(clock, impulse_response)
        self._control_decisions = np.zeros(self.M, dtype=np.double)
        self._old_control_decisions = np.zeros_like(self._control_decisions)
        self._setup_clock_phases()

    def _setup_clock_phases(self):
        self.clock_pulses = [0.0, self.clock.T]
        # this adds unique clock pulses in sorted order
        [
            self.clock_pulses.append(imp.t0)
            for imp in self._impulse_response
            if imp.t0 not in self.clock_pulses
        ]

    def _initialize_impulse_response(self, clock, impulse_response):
        if impulse_response is not None:
            if len(impulse_response) != self.M:
                raise Exception("must be M speciefied impulse responses")
            t0s = np.array([x.t0 for x in impulse_response])
            if (
                (t0s < 0.0).any()
                or (t0s > self.clock.T).any()
                or (t0s[:-1] > t0s[1:]).any()
            ):
                raise Exception("Invalid impulse respones")
            self._mulit_phase = np.sum(t0s) > 0.0
            self._impulse_response = impulse_response
        else:
            self._impulse_response = [StepResponse(0.0) for _ in range(self.M)]
            self._mulit_phase = False

    def reset(self, t0: float = 0.0):
        """Reset the digital control clock

        Parameters
        ----------
        t0: `float`, `optional`
            time to set next update at, defaults to 0.
        """
        if t0 != 0.0:
            raise DeprecationWarning("t0 will be removed in future version.")
        self._s = np.zeros(self.M, dtype=np.int8)
        self._control_decisions = np.zeros(self.M, dtype=np.double)

    def control_update(self, t: float, s_tilde: np.ndarray):
        """Updates the control at time t if valid.

        Parameters
        ----------
        t : `float`
            time at which the digital control i evaluated.
            For t > clock.T the control is updated.
        s_tilde : `array_like`, shape=(M_tilde,)
            state vector evaluated at time t
        """
        for m in range(self.M):
            # Comparator
            if np.allclose(t, self._impulse_response[m].t0, atol=self.clock._tt_2):
                self._s[m] = s_tilde[m] >= 0.0
                # DAC
                self._old_control_decisions[m] = self._control_decisions[m]
                self._control_decisions[m] = 2.0 * self._s[m] - 1.0

    def control_signal(self) -> np.ndarray:
        """Returns the current control state, i.e, :math:`\mathbf{s}[k]`.

        Examples
        --------
        >>> from cbadc.digital_control import DigitalControl
        >>> from cbadc.analog_signal import Clock
        >>> import numpy as np
        >>> T = 1e-6
        >>> M = 4
        >>> dc = DigitalControl(Clock(T), M)
        >>> _ = dc.control_contribution(T)
        >>> dc.control_signal()
        array([0., 0., 0., 0.])


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
        temp = np.zeros(self.M)
        for m in range(self.M):
            if t < self._impulse_response[m].t0:
                temp[m] = self._impulse_response[m](t + self.clock.T)
            else:
                temp[m] = self._impulse_response[m](t)

            if t <= self.t_delay:
                temp[m] *= self._old_control_decisions[m]
            else:
                temp[m] *= self._control_decisions[m]

        return temp

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
