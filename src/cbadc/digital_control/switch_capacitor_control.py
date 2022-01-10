import numpy as np
from .digital_control import DigitalControl


class SwitchedCapacitorControl(DigitalControl):
    """Represents a digital control system that uses switched capacitor control


    Parameters
    ----------
    T : `float
        total clock period
    T1 : `float`, `array_like`, shape=(M,)
        time at which the digital control empties the capacitor into the
        system. Can be either float or array of float.
    T2 : `float`, `array_like`, shape=(M,)
        time at which the switched capacitor is re-charged and disconnected
        from the analog system.
    M : `int`
        number of controls.
    A: array_like, shape(M,M), optional
        dynamical system model.
    t0 : `float`, optional
        determines initial time, defaults to 0.
    VCap: `float`, optional
        the voltage stored on each capacitor before discharge,
        defaults to 1.


    Attributes
    ----------
    T : `float`
        total clock period :math:`T` of digital control system.
    T1 : `array_like`, shape=(M,)
        discharge phase time
    T2 : `float`
        charge phase time
    M : `int`
        number of controls :math:`M`.
    M_tilde : `int`
        number of control observations :math:`\\tilde{M}`.

    Note
    ----
    For this digital control system :math:`M=\\tilde{M}`.
    """

    def __init__(self, T, T1, T2, M, A, t0=0, VCap=1.0):
        if isinstance(T1, (list, tuple, np.ndarray)):
            self.T1 = np.array(T1, dtype=np.double)
        else:
            self.T1 = T1 * np.ones(M, dtype=np.double)

        self._T1_next = t0 + self.T1

        if isinstance(T2, (list, tuple, np.ndarray)):
            self.T2 = np.array(T2, dtype=np.double)
        else:
            self.T2 = T2 * np.ones(T2, dtype=np.double)
        self._T2_next = t0 + self.T2

        self.M = M

        # Check for invalid period times.
        for m in range(self.M):
            if self.T2[m] <= self.T1[m] or (self.T2[m] + T == self.T1[m]):
                raise Exception(
                    f"Invalid T1={self.T1[m]} and T2={self.T2[m]} for m={m}"
                )

        self.T = T
        if (self.T < self.T1).any() or (2 * self.T < self.T2).any():
            raise Exception("T1 cannot exceed T and T2 cannot exceed 2T.")

        self.phase = np.zeros(M, dtype=int)

        if not isinstance(self.M, int):
            raise Exception("M must be an integer.")

        self._s = np.zeros(self.M, dtype=int)
        self.A = np.array(A, dtype=np.double)
        self.VCap = VCap

    def next_update(self):
        t_next = np.inf
        for m in range(self.M):
            if self._T1_next[m] < t_next:
                t_next = self._T1_next[m]
            if self._T2_next[m] < t_next:
                t_next = self._T2_next[m]
        return t_next

    def control_update(self, t, s_tilde: np.ndarray):
        # Check if time t has passed the next control update\
        reset = np.zeros(self.M, dtype=bool)
        for m in range(self.M):
            if not t < self._T2_next[m]:
                self.phase[m] = 1
                reset[m] = True
                self._T2_next[m] += self.T
            elif not t < self._T1_next[m]:
                self._s[m] = s_tilde[m] > 0
                self.phase[m] = 0
                reset[m] = True
                self._T1_next[m] += self.T
        return self.phase, reset, self._s

    def control_signal(self) -> np.ndarray:
        """Returns the current control state, i.e, :math:`\mathbf{s}[k]`.

        Returns
        -------
        `array_like`, shape=(M,), dtype=numpy.int8
            current control state.
        """
        return self._s
