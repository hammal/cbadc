from typing import List, Tuple
import numpy as np
from ..analog_signal import StepResponse, _valid_clock_types
from ..analog_signal.impulse_responses import _ImpulseResponse
from .digital_control import DigitalControl


class ConservativeControl(DigitalControl):
    """Represents a digital control system.

    This is the conservative digital control where
    :math:`M=\\tilde{M}` and each control signal is updated
    independently. Furthermore, the DAC waveform is a constant signal
    as :math:`\mathbf{s}(t)=\mathbf{s}[k]` for :math:`t\in[k T, (k+1)T)`.

    What differentiates the conservative digital control from the
    regular digital control is the fact that it has a third 0 state
    namely if the input signal does not exceed :math:`[-b, b]` the control
    descision is 0.

    Parameters
    ----------
    clock : :py:class:`cbadc.analog_signal.clock.Clock`
        the clock to which the digital control synchronizes its updates.
    M : `int`
        number of controls.
    t0 : `float`, `optional`
        determines initial time, defaults to 0.
    bounds: [(`float`,`float`)], `optional`
        a M sized list of tuples corresponding to the negative and positive bound.,
        defaults to [(-0.25, 0.25),...].
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
    bounds: [(float,float)]


    """

    def __init__(
        self,
        clock: _valid_clock_types,
        M: int,
        t0: float = 0.0,
        bounds: List[Tuple[float, float]] = None,
        impulse_response: _ImpulseResponse = StepResponse(),
    ):
        if bounds is None:
            self.bounds = [(-0.5e-3, 0.5e-3) for _ in range(M)]
        else:
            if len(bounds) != M:
                raise BaseException("Must be M bounds")
            for b in bounds:
                if len(b) != 2:
                    raise BaseException("each of the M bounds must be tuples of size 2")
                if b[0] > b[1]:
                    raise BaseException(
                        "the bounds must be organized as (lower bound, upper bound)"
                    )
                self.bounds = bounds
        self._b_lower = np.array([b[0] for b in self.bounds])
        self._b_upper = np.array([b[1] for b in self.bounds])
        super().__init__(clock, M, t0, impulse_response)
        self._s = self._s.astype(np.double)

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
            for m in range(self.M):
                if s_tilde[m] < self._b_upper[m] and s_tilde[m] > self._b_lower[m]:
                    self._s[m] = 0.5
            self._t_last_update[:] = t
            self._t_next += self.clock.T
            # DAC
            self._control_descisions = np.asarray(2 * self._s - 1, dtype=np.double)

    def __str__(self):
        return f"""{super().__str__()}
{80 * '-'}

bounds:
{self.bounds}

"""
