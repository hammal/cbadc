"""Digital controls

This module provides a general :py:class:`cbadc.digital_control.DigitalControl`
class to enabled the creation of a general independently controlled digital
control system.
"""
import numpy as np


class DigitalControl:
    """Represents a digital control system.

    This is the simplest digital control where
    :math:`M=\\tilde{M}` and each control signal is updated
    independently. Furthermore, the DAC waveform is a constant signal
    as :math:`\mathbf{s}(t)=\mathbf{s}[k]` for :math:`t\in[k T, (k+1)T)`.

    Parameters
    ----------
    T : `float`
        clock period at which the digital control updates.
    M : `int`
        number of controls.
    t0 : `float`: optional
        determines initial time, defaults to 0.

    Attributes
    ----------
    T : `float`
        clock period :math:`T` of digital control system.
    M : `int`
        number of controls :math:`M`.
    M_tilde : `int`
        number of control observations :math:`\\tilde{M}`.

    Note
    ----
    For this digital control system :math:`M=\\tilde{M}`.

    See also
    ---------
    :py:class:`cbadc.simulator.StateSpaceSimulator`

    Examples
    --------
    >>> from cbadc.digital_control import DigitalControl
    >>> T = 1e-6
    >>> M = 4
    >>> dc = DigitalControl(T, M)
    >>> print(dc)
    The Digital Control is parameterized as:
    T = 1e-06,
    M = 4, and next update at
    t = 1e-06
    """

    def __init__(self, T: float, M: int, t0: float = 0.0):
        self.T = T
        self.M = M
        self.M_tilde = M
        self._t_next = t0 + self.T
        self._s = np.zeros(self.M, dtype=np.int8)
        self._dac_values = np.zeros(self.M, dtype=np.double)

    def control_contribution(self, t: float, s_tilde: np.ndarray) -> np.ndarray:
        """Evaluates the control contribution at time t given a control observation
        s_tilde.

        Parameters
        ----------
        t : `float`
            time at which the digital control i evaluated.
        s_tilde : `array_like`, shape=(M_tilde,)
            state vector evaluated at time t

        Examples
        --------
        >>> from cbadc.digital_control import DigitalControl
        >>> import numpy as np
        >>> T = 1e-6
        >>> M = 4
        >>> dc = DigitalControl(T, M)
        >>> res = dc.control_contribution(T + 1e-100, np.array([0.1, -0.2, 0.3, -99]))
        >>> print(np.array(res))
        [ 1. -1.  1. -1.]



        Returns
        -------
        `array_like`, shape=(M,)
            the control signal :math:`\mathbf{s}(t)`

        """
        # Check if time t has passed the next control update
        if t >= self._t_next:
            # if so update the control signal state
            self._s = s_tilde >= 0
            self._t_next += self.T
            # DAC
            self._dac_values = np.asarray(2 * self._s - 1, dtype=np.double)
        return self._dac_values

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
        return self._s

    def impulse_response(self, m: int, t: float):
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
        temp[m] = 1
        return temp

    def __str__(self):
        return f"The Digital Control is parameterized as:\nT = {self.T},\nM = {self.M}, and next update at\nt = {self._t_next}"


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
            self._dac_values = np.asarray(
                2 * self._s - 1, dtype=np.double)
        return self._dac_values
