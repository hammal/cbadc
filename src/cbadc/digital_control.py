"""Digital controls

This module provides a general :py:class:`cbadc.digital_control.DigitalControl`
class to enabled the creation of a general independently controlled digital
control system.
"""
import numpy as np
import scipy.optimize
import itertools


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
        if t >= 0 and t <= self.T:
            temp[m] = 1
        return temp

    def __str__(self):
        return f"The Digital Control is parameterized as:\nT = {self.T},\nM = {self.M}, and next update at\nt = {self._t_next}"


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
    M : `int`
        number of controls.
    phases: array_like, shape=(M,), dtype=int
        the phase number associated with each phase,
        for example phases = [0, 1, 2, 0] would assign first
        and forth control to phase 0 and the second to phase 1
        etc.
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
    """

    def __init__(self, T: float, M: int, phases: np.ndarray, t0: float = 0):
        DigitalControl.__init__(self, T, M, t0=t0)
        self._phases = phases
        if self._phases.size != self.M:
            raise BaseException("phase must be an M sized numpy array")
        self._t_next = t0 + self._phases * T
        self.number_of_phases = np.max(self._phases) + 1

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
        for m in range(self.M):
            if t >= self._t_next[m]:
                # if so update the control signal state
                self._s[m] = s_tilde[m] >= 0
                self._t_next[m] += self.T * self.number_of_phases
                # DAC
                self._dac_values = np.asarray(2 * self._s - 1, dtype=np.double)
                # print(f"m = {m}, t = {t}, s = {self._dac_values}")
        return self._dac_values


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


def overcomplete_set(Gamma: np.ndarray, M: int):
    """
    Construct a overcomplete set of normalized column vectors

    Parameters
    ----------
    Gamma: array_like
        the initial set of vectors
    M : `int`
        the desired number of column vectors

    Returns
    -------
    array_like
        the resulting set of column vectors.
    """
    T = np.copy(Gamma.transpose())
    for dim in range(T.shape[0]):
        T[dim, :] /= np.linalg.norm(T[dim, :], ord=2)
    number_of_candidates_per_new_vector = 100
    while T.shape[0] < M:
        candidate_set = np.random.randn(T.shape[1], number_of_candidates_per_new_vector)
        candidate_set /= np.linalg.norm(candidate_set, ord=2, axis=0)

        cost = np.zeros(number_of_candidates_per_new_vector)

        def cost_function(alpha):
            return np.linalg.norm(np.dot(T, alpha), ord=2) / np.linalg.norm(
                alpha, ord=2
            )

        for index in range(number_of_candidates_per_new_vector):
            sol = scipy.optimize.minimize(cost_function, candidate_set[:, index])
            cost[index] = sol.fun
            candidate_set[:, index] = sol.x / np.linalg.norm(sol.x, ord=2)

        T = np.vstack(
            (T, candidate_set[:, best_candidate_index].reshape((1, T.shape[1])))
        )
    return T.transpose()


def unit_element_set(N: int, M: int):
    """
    Construct an overcomplete set of vectors only using a single element, i.e.,

    :math:`\mathbf{v} \in \\{ - \\alpha, \\alpha , 0 \\}^{N \\times M}`

    where duplicates and the :math:`\\begin{pmatrix}0, \dots, 0 \\end{pmatrix}`
    is excluded from the set.

    Parameters
    ----------
    N: `int`
        the length of the vectors
    M: `int`
        the number of unique vectors

    Returns
    -------
    array_like, shape=(N, M)
        a matrix containing the unique vectors as column vectors.
    """
    candidate_set = []
    for item in itertools.product(*[[-1, 1, 0] for _ in range(N)]):
        duplicate = False
        sum = np.sum(np.abs(np.array(item)))
        if sum == 0:
            break
        candidate = np.array(item)
        for item in candidate_set:
            s1 = np.sum(np.abs(np.array(item) - candidate))
            s2 = np.sum(np.abs(np.array(item) + candidate))
            if s1 == 0 or s2 == 0:
                # print(item, candidate)
                duplicate = True
        if not duplicate:
            candidate_set.append(candidate)

    candidate_set = np.array(candidate_set)  # [
    #        np.random.permutation(len(candidate_set)), :
    # ]
    # print(candidate_set)
    if candidate_set.shape[0] < M:
        raise BaseException("Not enough unique combinations; M is set to large.")
    set = candidate_set[0, :].reshape((N, 1))
    candidate_set = np.delete(candidate_set, 0, 0)

    while set.shape[1] < M:
        costs = np.linalg.norm(
            np.dot(candidate_set, set), ord=2, axis=1
        ) / np.linalg.norm(candidate_set, axis=1, ord=2)
        next_index = np.argmin(costs)
        # print(candidate_set, costs, next_index)
        set = np.hstack((set, candidate_set[next_index, :].reshape((N, 1))))
        candidate_set = np.delete(candidate_set, next_index, 0)
    # return np.array(set)[:, np.random.permutation(set.shape[1])]
    return np.array(set)


class SwitchedCapacitorControl:
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
    A: array_like, shape(M,M)
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
                raise BaseException(
                    f"Invalid T1={self.T1[m]} and T2={self.T2[m]} for m={m}"
                )

        self.T = T
        if (self.T < self.T1).any() or (2 * self.T < self.T2).any():
            raise BaseException("T1 cannot exceed T and T2 cannot exceed 2T.")

        self.phase = np.zeros(M, dtype=int)

        if not isinstance(self.M, int):
            raise BaseException("M must be an integer.")

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
