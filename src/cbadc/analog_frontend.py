"""The analog fronted module."""

import logging
from copy import deepcopy as _deepcopy
from typing import Optional, Union

import numpy as np
import scipy.integrate as _integrate
import scipy.linalg as _linalg
import sympy as sp
from scipy.linalg import block_diag
from scipy.signal import StateSpace, freqresp

from .analog_filter import ChainOfIntegrators, LeapFrog
from .analog_filter.analog_system import AnalogSystem, as2af
from .analog_signal import (
    AnalogSignal,
    ConcatenatedSignals,
    PartitionedSignal,
    Sinusoidal,
)
from .delsig import partitionABCD, simulateDSM
from .digital_control import DigitalControl
from .fom import enob_to_snr, snr_from_dB, snr_to_enob
from .noise import discrete_process_noise_cov as _discrete_process_noise_cov
from .noise import psd_factor as _psd_factor

logger = logging.getLogger(__name__)


def _g_i_chain_of_integrators(N: int):
    """Compute the integration factor g_i

    Parameters
    ----------
    N: `int`
        the system order
    Returns
    -------
    :  `float`
        the computed integration factor.
    """
    return 2.0 * N + 1.0


def _g_i_leapfrog(N: int):
    """Compute the integration factor g_i

    Parameters
    ----------
    N: `int`
        the system order
    Returns
    -------
    :  `float`
        the computed integration factor.
    """
    omega, omega_p, gamma = sp.symbols("w w_p, g", real=True, positive=True)
    n = sp.symbols("n", integer=True, positive=True)
    determinant = sp.Product(
        sp.I * (omega + omega_p * sp.cos(n * sp.pi / (N + 1))), (n, 1, N)
    )
    H = determinant / ((gamma * omega_p) ** N)
    H2 = sp.Abs(H) ** 2
    LF_int = sp.integrate(H2, (omega, 0, omega_p))
    # g_i = sp.simplify(omega_p / (LF_int * gamma ** (2 * N)))
    g_i = omega_p / (LF_int * gamma ** (2 * N))
    return np.float64(g_i.subs(omega_p, 1e0).evalf())


def _leapfrog_3dB_factor(alpha: float, beta: float, N: int, BW: float) -> float:
    """Factor to scale ``omega_p`` (i.e. ``alpha``, ``beta``) so the leap-frog
    open-loop signal transfer function has its 3 dB edge at ``BW``.

    ``|G(w)|`` is invariant in ``w/omega_p``, so the 3 dB frequency scales
    linearly with ``omega_p`` and the factor is the single ratio
    ``BW / f_3dB(current)``. ``f_3dB`` is read off ``G(w) = (jw I - A)^-1 B_u``
    (norm over the observed states, relative to the low-frequency passband)."""
    A0 = np.diag(alpha * np.ones(N - 1), 1) + np.diag(beta * np.ones(N - 1), -1)
    Bu = np.zeros(N)
    Bu[0] = beta
    fs = np.logspace(np.log10(BW / 1000), np.log10(BW * 20), 50000)
    jw = 1j * 2 * np.pi * fs
    G = np.linalg.solve(jw[:, None, None] * np.eye(N) - A0,
                        np.broadcast_to(Bu[:, None], (fs.size, N, 1)))  # (size, N, 1)
    g = np.linalg.norm(G[:, :, 0], axis=1)
    below = np.where(g < g[0] / np.sqrt(2.0))[0]
    return BW / fs[below[0]] if below.size else 1.0


class CyclicStateSpace(StateSpace):
    """A cyclic state space system

    A cyclic state space system is a state space system where
    the state matrix is cyclic, i.e. A[k] = A[k % K] for some K.

    Parameters
    ----------
    A: `np.ndarray`, shape=(K, N, N)
        the state matrix
    B: `np.ndarray`, shape=(K, N, L + M)
        the input matrix
    C: `np.ndarray`, shape=(H, M, N)
        the output matrix
    D: `np.ndarray`, shape=(H, M, L + M)
        the feedthrough matrix
    dt: `float`, optional
        the sampling period, defaults to None (continuous-time system)

    Attributes
    ----------
    A: `np.ndarray`, shape=(K, N, N)
        the state matrix
    B: `np.ndarray`, shape=(K, N, L + M)
        the input matrix
    C: `np.ndarray`, shape=(H, M, N)
        the output matrix
    D: `np.ndarray`, shape=(H, M, L + M)
        the feedthrough matrix
    dt: `float`
        the sampling period

    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        dt: Optional[float] = None,
    ):
        if not isinstance(A, np.ndarray):
            raise ValueError("A must be a numpy array")
        if not isinstance(B, np.ndarray):
            raise ValueError("B must be a numpy array")
        if not isinstance(C, np.ndarray):
            raise ValueError("C must be a numpy array")
        if not isinstance(D, np.ndarray):
            raise ValueError("D must be a numpy array")

        if A.ndim != 3:
            raise ValueError("A must be a 3D numpy array")
        if B.ndim != 3:
            raise ValueError("B must be a 3D numpy array")
        if C.ndim != 3:
            raise ValueError("C must be a 3D numpy array")
        if D.ndim != 3:
            raise ValueError("D must be a 3D numpy array")

        if A.shape[1] != A.shape[2]:
            raise ValueError("A must be square")
        if A.shape[0] != B.shape[0]:
            raise ValueError("A and B must have the same number of slices")
        if C.shape[0] != D.shape[0]:
            raise ValueError("C and D must have the same number of slices")
        if A.shape[1] != B.shape[1]:
            raise ValueError("A and B must have the same number of rows")
        if B.shape[2] != D.shape[2]:
            raise ValueError("B and D must have the same number of columns")
        if C.shape[1] != D.shape[1]:
            raise ValueError("C and D must have the same number of rows")
        if A.shape[1] != C.shape[2]:
            raise ValueError("A and C must have the same number of columns")
        if dt is not None and (not isinstance(dt, (float, int)) or dt <= 0):
            raise ValueError("dt must be a positive float or None")
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self._dt = dt
        # super().__init__(A[0], B[0], C[0], D[0])

        self._K = A.shape[0]
        self._H = C.shape[0]
        self._N = A.shape[1]
        self._M = D.shape[1]
        self._L = B.shape[2] - self._M
        self._is_discrete_time = dt is not None

    def get_state_space(self, k: int = 0, h: int = 0) -> StateSpace:
        """Get the state space representation at time index k

        Parameters
        ----------
        k: `int`, optional
            the time index, defaults to 0

        Returns
        -------
        :py:class:`scipy.signal.StateSpace`
            the state space representation at time index k
        """
        if not isinstance(k, int):
            raise ValueError("k must be an integer")

        if not isinstance(h, int):
            raise ValueError("h must be an integer")

        print(
            self.A.shape,
            self.B[k % self._K].shape,
            self.C[h % self._H].shape,
            self.D[h % self._H].shape,
        )

        return StateSpace(
            self.A[k % self._K],
            self.B[k % self._K],
            self.C[h % self._H],
            self.D[h % self._H],
        )


class AnalogFrontend:
    """An analog frontend

    As a cornerstone of the control-bounded conversion theory,
    an analog frontend is a analog-to-digital conversion system
    which operates by controlling an analog filter using digital
    control.

    In particular, an analog filter is specified by a
    [[A B], [C D]] state-space representation, where
    - A is the state matrix of shape ((K, N, N)), where K is the sequence length of the system, and N is the number of analog state variables.
    - B is the input matrix of shape ((K, N, L + M)), where L is the number of analog signals and M is the number of control signals.
    Furthermore, B is partitioned into B = [B_L B_M],
    where B_0 is the corresponding analog signal input matrix of shape ((N, L))
    and B_1 is the digital control matrix of shape ((N, M)).
    - C is the output matrix (control input matrix) of shape ((H, M, N)), and
    - D is the feedthrough matrix of shape ((H, M, L + M)) where
    D is partitioned into D = [D_L D_M], where
    D_L is the feedthrough matrix for the analog signals and
    D_M is the feedthrough matrix for the control signals.
    H is the number of control sequences where a [C D][k, i, :] = [0, ..., 0] would result in a no operation for the i-th control signal at the k-th sequence.


    Parameters
    ----------
    analog_filter: :py:class:`scipy.signal.StateSpace`
        an analog system instance
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        a digital control instance
    analog_signal: :py:class:`cbadc.analog_signal._AnalogSignal`
        an analog signal
    state_covariance: `np.ndarray`, optional
        the state covariance matrix, defaults to None.
    output_covariance: `np.ndarray`, optional
        the output covariance matrix, defaults to None.
    slew_rate: `np.ndarray`, optional
        the slew rate, defaults to np.inf * np.ones(N).
    v_o_max: `np.ndarray`, optional
        the maximum output voltage, defaults to np.inf * np.ones(M).
    v_o_min: `np.ndarray`, optional
        the minimum output voltage, defaults to -np.inf * np.ones(M).
    seed: `int`, optional
        the random seed, defaults to 98123591265830293457639481.

    Attributes
    ----------
    analog_filter: :py:class:`scipy.signal.StateSpace`
        the analog frontend's analog system instance
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        the analog frontend's digital control instance
    analog_signal: :py:class:`cbadc.analog_signal._AnalogSignal`
        the analog signal.
    A: :py:class:`numpy.ndarray`
        the state matrix.
    B: :py:class:`numpy.ndarray`
        the input matrix.
    C: :py:class:`numpy.ndarray`
        the output matrix.
    D: :py:class:`numpy.ndarray`
        the feedthrough matrix.
    L: `int`
        the number of analog signals.
    M: `int`
        the number of control signals.
    N: `int`
        the number of analog state variables.


    """

    def __init__(
        self,
        analog_filter: StateSpace,
        digital_control: DigitalControl,
        analog_signal: Optional[AnalogSignal] = None,
        state_covariance: Optional[np.ndarray] = None,
        output_covariance: Optional[np.ndarray] = None,
        slew_rate: Optional[np.ndarray] = None,
        state_max: Optional[np.ndarray] = None,
        state_min: Optional[np.ndarray] = None,
        seed: int = 98123591265830293457639481,
    ):

        self.analog_filter = analog_filter
        self.digital_control = digital_control
        self.analog_signal = analog_signal

        self.A = self.analog_filter.A
        self.B = self.analog_filter.B
        self.C = self.analog_filter.C
        self.D = self.analog_filter.D

        self.state_covariance = state_covariance
        self.output_covariance = output_covariance
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        if slew_rate is None:
            # V/s
            self.slew_rate = np.inf * np.ones((self.N))
        elif isinstance(slew_rate, np.ndarray):
            if slew_rate.size != self.N:
                raise ValueError(
                    f"slew_rate must have size {self.N}, got {slew_rate.size}"
                )
            self.slew_rate = slew_rate.flatten()
        else:
            raise ValueError(f"slew_rate must be None or a {self.N}-sized numpy array")

        if state_max is None:
            self.state_max = np.inf * np.ones((self.N))
        elif isinstance(state_max, np.ndarray):
            if state_max.size != self.N:
                raise ValueError(
                    f"state_max must have size {self.N}, got {state_max.size}"
                )
            self.state_max = state_max.flatten()
        else:
            raise ValueError(f"state_max must be None or a {self.N}-sized numpy array")

        if state_min is None:
            self.state_min = -np.inf * np.ones((self.N))
        elif isinstance(state_min, np.ndarray):
            if state_min.size != self.N:
                raise ValueError(
                    f"state_min must have size {self.N}, got {state_min.size}"
                )
            self.state_min = state_min.flatten()
        else:
            raise ValueError(f"state_min must be None or a {self.N}-sized numpy array")

    def __str__(self) -> str:
        return (
            f"AnalogFrontend(\n"
            "  analog_filter=StateSpace(\n"
            f"A=\n{self.A},\n"
            f"B=\n{self.B},\n"
            f"C=\n{self.C},\n"
            f"D=\n{self.D}),\n"
            ")\n"
            f"digital_control={self.digital_control},\n"
            f"analog_signal={self.analog_signal},\n"
            f"state_covariance={self.state_covariance},\n"
            f"output_covariance={self.output_covariance},\n"
            f"N={self.N},\n"
            f"L={self.L},\n"
            f"M={self.M},\n"
            f"dt={self.dt}\n"
            f")"
        )

    @property
    def A(self) -> np.ndarray:
        """The state matrix

        Returns
        -------
        : numpy.ndarray
            the analog filter's state matrix
        """
        return self._A

    @A.setter
    def A(self, A: Optional[np.ndarray] = None):
        if not isinstance(A, np.ndarray):
            raise ValueError("A must be a numpy array")
        if A.shape[-2:] != (self.N, self.N):
            raise ValueError(
                f"A.shape[-2:] must have shape {(self.N, self.N)}, got {A.shape}"
            )
        self._A: np.ndarray = np.asarray(A).reshape((-1, self.N, self.N))
        # self.analog_filter.A = self._A[0, :, :]

    @property
    def B(self) -> np.ndarray:
        """The input matrix

        Returns
        -------
        : numpy.ndarray
            the analog filter's input matrix
        """
        return self._B

    @B.setter
    def B(self, B: Optional[np.ndarray] = None):
        if not isinstance(B, np.ndarray):
            raise ValueError("B must be a numpy array")
        if B.shape[-2:] != (self.N, self.L + self.M):
            raise ValueError(
                f"B.shape[-2:] must have shape {(self.N, self.L + self.M)}, got {B.shape[-2:]}"
            )
        self._B: np.ndarray = np.asarray(B).reshape((-1, self.N, self.L + self.M))
        # self.analog_filter.B = self._B[0, :, :]

    @property
    def C(self) -> np.ndarray:
        """The output matrix

        Returns
        -------
        : numpy.ndarray
            the analog filter's output matrix
        """
        return self._C

    @C.setter
    def C(self, C: Optional[np.ndarray] = None):
        if not isinstance(C, np.ndarray):
            raise ValueError("C must be a numpy array")
        if C.shape[-2:] != (self.M, self.N):
            raise ValueError(
                f"C.shape[-2:] must have shape {(self.M, self.N)}, got {C.shape}"
            )
        self._C = np.asarray(C).reshape((-1, self.M, self.N))
        # self.analog_filter.C = self._C[0, :, :]

    @property
    def D(self) -> np.ndarray:
        """The feedthrough matrix

        Returns
        -------
        : numpy.ndarray
            the analog filter's feedthrough matrix
        """
        return self._D

    @D.setter
    def D(self, D: Optional[np.ndarray] = None):
        if not isinstance(D, np.ndarray):
            raise ValueError("D must be a numpy array")
        if D.shape[-2:] != (self.M, self.L + self.M):
            raise ValueError(
                f"D must have shape {(self.M, self.L + self.M)}, got {D.shape}"
            )
        self._D = np.asarray(D).reshape((-1, self.M, self.L + self.M))
        # self.analog_filter.D = self._D[0, :, :]

    @property
    def ABCD(self):
        """The ABCD matrix

        Returns
        -------
        : numpy.ndarray, shape=(M + N, L + M + N)
            the analog filter's ABCD matrix
        """
        return np.vstack(
            (
                np.hstack((self.A[0, :, :], self.B[0, :, :])),
                np.hstack((self.C[0, :, :], self.D[0, :, :])),
            )
        )

    @property
    def analog_filter(self) -> StateSpace:
        """The analog filter

        Returns
        -------
        :py:class:`scipy.signal.StateSpace`
            the analog filter
        """
        return self._analog_filter

    @analog_filter.setter
    def analog_filter(self, analog_filter: StateSpace):
        if not isinstance(analog_filter, StateSpace):
            if isinstance(analog_filter, AnalogSystem):
                analog_filter = as2af(analog_filter)
            else:
                raise ValueError("analog_filter must be a StateSpace instance")
        self._analog_filter = analog_filter

    @property
    def digital_control(self) -> DigitalControl:
        """The digital control

        Returns
        -------
        :py:class:`cbadc.digital_control.DigitalControl`
            the digital control
        """
        return self._digital_control

    @digital_control.setter
    def digital_control(self, control: DigitalControl):
        if not isinstance(control, DigitalControl):
            raise ValueError("digital_control must be a DigitalControl instance")
        self._digital_control = control

    @property
    def analog_signal(self) -> AnalogSignal:
        """The analog signal

        Returns
        -------
        :py:class:`cbadc.analog_signal.AnalogSignal`
            the analog signal
        """
        return self._analog_signal

    @analog_signal.setter
    def analog_signal(self, signal: Optional[AnalogSignal]):
        if signal is None:
            L = self.analog_filter.B.shape[-1] - self.M
            M = 1
            offset = np.zeros((L, M))
            signal = AnalogSignal(offset)
        if not isinstance(signal, AnalogSignal):
            raise ValueError(
                f"analog_signal {signal} must be an AnalogSignal or derived instance"
            )
        self._analog_signal = signal

    @property
    def L(self) -> int:
        """The number of analog signals

        Returns
        -------
        : int
            the number of analog signals
        """
        return self.analog_signal.L

    @property
    def J(self) -> int:
        """The number of parallel input signals

        Used for parallel simulations

        Returns
        -------
        : int
            the number of parallel inputs
        """
        return self.analog_signal.J

    @property
    def M(self) -> int:
        """The number of control signals

        Returns
        -------
        : int
            the number of control signals
        """
        return self.digital_control.M

    @property
    def N(self) -> int:
        """The number of analog state variables

        Returns
        -------
        : int
            the number of analog state variables"""
        return self.analog_filter.A.shape[-1]

    @property
    def loop_order(self) -> int:
        """The loop order of the analog frontend

        Note
        ----
        This is equivalent to :math:`N`.

        Returns
        -------
        : int
            the loop order of the analog frontend
        """
        return self.N

    @property
    def dt(self) -> float:
        """The digital control sampling period

        Returns
        -------
        : float
            the digital control sampling period
        """
        return self.digital_control.dt

    @dt.setter
    def dt(self, dt: float):
        ds_old = self.dt
        self.digital_control.dt = dt
        fs_ds_old = self.fs * ds_old
        self.analog_filter.A *= fs_ds_old
        self.analog_filter.B *= fs_ds_old
        self._slew_rate *= fs_ds_old

    @property
    def fs(self) -> float:
        """The digital control sampling rate

        Note
        ----
        Same as :math:`f_s = 1 / \text{dt}`.

        Returns
        -------
        : float
            the digital control sampling rate

        """
        return 1.0 / self.dt

    @property
    def quantization_level(self):
        """The digital control quantization level

        Returns
        -------
        `np.ndarray`, shape=(M, 1)
            the number of quantization levels for each digital control.
        """
        return self.digital_control.quantization_level

    @property
    def slew_rate(self) -> np.ndarray:
        """The per state slew rate of the analog frontend

        Returns
        -------
        : numpy.ndarray, shape (N,)
            the slew rate, in V/s, of the analog frontend

        """
        return self._slew_rate

    @slew_rate.setter
    def slew_rate(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("slew_rate must be a numpy array")
        if value.size != self.N:
            raise ValueError(f"slew_rate must have size {self.N}, got {value.size}")
        if (value <= 0.0).any():
            raise ValueError("slew_rate must be positive")
        self._slew_rate = value

    @property
    def state_min(self):
        """The minimum output voltage

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N,)
            the minimum output voltage matrix.
        """
        return self._v_o_min

    @state_min.setter
    def state_min(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("v_o_min must be a numpy array")
        if value.size != self.N:
            raise ValueError(f"v_o_min must have size {self.N}, got {value.size}")
        self._v_o_min = value.flatten()

    @property
    def state_max(self):
        """The maximum output voltage

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N,)
            the maximum output voltage matrix.
        """
        return self._v_o_max

    @state_max.setter
    def state_max(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("v_o_max must be a numpy array")
        if value.size != self.N:
            raise ValueError(f"v_o_max must have size {self.N}, got {value.size}")
        self._v_o_max = value.flatten()

    @property
    def is_discrete_time(self) -> bool:
        """Whether the analog frontend is discrete-time

        Returns
        -------
        : bool
            whether the analog frontend is discrete-time
        """
        return hasattr(self.analog_filter, "dt") and self.analog_filter.dt is not None

    @property
    def state_covariance(self) -> Union[None, np.ndarray]:
        """The state covariance matrix

        Returns
        -------
        : numpy.ndarray
            the state covariance matrix
        """
        return self._state_covariance

    @state_covariance.setter
    def state_covariance(self, state_covariance: Optional[np.ndarray] = None):
        if state_covariance is None:
            self._state_covariance = None
        else:
            if not isinstance(state_covariance, np.ndarray):
                raise ValueError("state_covariance must be a numpy array")
            if state_covariance.shape != (self.N, self.N):
                raise ValueError(
                    f"state_covariance must have shape {(self.N, self.N)}, got {state_covariance.shape}"
                )
            if not np.allclose(state_covariance, state_covariance.T):
                raise ValueError("state_covariance must be symmetric")
            # Map to the discrete per-step covariance the simulator samples.
            # For a continuous-time frontend ``state_covariance`` is a noise
            # *intensity* and must be propagated through the van-Loan integral;
            # for an already-discrete frontend it is the per-step covariance
            # directly (``discretize`` does the conversion when it builds one).
            if self.is_discrete_time:
                state_cov_d = state_covariance
            else:
                state_cov_d = _discrete_process_noise_cov(
                    self.A[0, :, :], state_covariance, self.dt
                )
            self._state_covariance = state_covariance
            self._state_cov_cholesky = _psd_factor(state_cov_d)

    @staticmethod
    def input_referred_covariance_matrix(
        analog_frontend: "AnalogFrontend", input_covariance: np.ndarray
    ):
        """Compute the input-referred covariance matrix

        Parameters
        ----------
        analog_filter: :py:class:`scipy.signal.StateSpace`
            the analog filter
        input_covariance: :py:class`numpy.ndarray`, shape=(L, L)
            the input covariance matrix


        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N, N)
            the input-referred covariance matrix
        """
        if not isinstance(analog_frontend, AnalogFrontend):
            raise ValueError("analog_filter must be an AnalogFrontend instance")
        if not isinstance(input_covariance, np.ndarray):
            raise ValueError("input_covariance must be a numpy array")
        if input_covariance.shape != (
            analog_frontend.L,
            analog_frontend.L,
        ):
            raise ValueError(
                f"input_covariance must have shape {(analog_frontend.L, analog_frontend.L)}, got {input_covariance.shape}"
            )
        if not np.allclose(input_covariance, input_covariance.T):
            raise ValueError("input_covariance must be symmetric")
        B = analog_frontend.B[0, :, : analog_frontend.L]
        return B @ input_covariance @ B.transpose()

    @property
    def output_covariance(self) -> Union[None, np.ndarray]:
        """The output covariance matrix

        Returns
        -------
        : numpy.ndarray
            the output covariance matrix
        """
        return self._output_covariance

    @output_covariance.setter
    def output_covariance(self, output_covariance: Optional[np.ndarray] = None):
        if output_covariance is None:
            self._output_covariance = None
        else:
            if not isinstance(output_covariance, np.ndarray):
                raise ValueError("output_covariance must be a numpy array")
            if output_covariance.shape != (self.M, self.M):
                raise ValueError(
                    f"output_covariance must have shape {(self.M, self.M)}, got {output_covariance.shape}"
                )
            if not np.allclose(output_covariance, output_covariance.T):
                raise ValueError("output_covariance must be symmetric")
            self._output_covariance = output_covariance
            self._output_cov_cholesky = _psd_factor(output_covariance)

    def simulateDSM(
        self,
        size: int,
        x: Optional[np.ndarray] = None,
        t0: float = 0.0,
    ):
        """Schreier's Matlab toolbox simulator port


        Parameters
        ----------
        size: `int`
            the number of samples to simulate
        x: `np.ndarray`, optional
            the initial state, defaults to zero.
        t0: `float`, optional
            the initial time, defaults to 0.0.

        Returns
        -------
        : dict[str, np.ndarray]
            a dictionary containing the simulation results where
            - 't': the time vector, shape (size,)
            - 'u': the analog signal evaluated at t, shape (size, L, J)
            - 'v': the digital control signals, shape (size, M, J)
            - 'x': the state vector, shape (size, N, J)
            - 'y': the quantization input vector, shape (size, M, J)
        """

        if not self.is_discrete_time:
            raise ValueError("Analog frontend must be discrete-tim to use simulateDSM")

        if self.dt != 1.0:
            # Normalize
            self.dt = 1.0

        # Allocate memory
        inputs = np.zeros((size, self.L + self.M, self.J), dtype=float)
        states = np.zeros((size, self.N, self.J), dtype=float)
        outputs = np.zeros((size, self.M, self.J), dtype=float)

        if x is None:
            x = np.zeros(self.N)
        states[0, :, 0] = x

        t = np.arange(size) * self.digital_control.dt + t0

        # pre compute input signal contributions
        inputs[:, : self.L] = self.analog_signal.evaluate(t)

        if self.M != 1:
            raise ValueError("Only single dimension systems are supported")
        if self.J != 1:
            raise ValueError("Parallel simulation is not supported")

        v, xn, _, y = simulateDSM(
            inputs[:, : self.L, 0].transpose(),
            self.ABCD,
            self.quantization_level[0],
            x0=states[0, :, 0],
        )
        inputs[:, -1, 0] = v.transpose()
        outputs[:, 0, 0] = y.transpose()
        states[1:, :, 0] = xn.transpose()[:-1, :]

        return {
            "t": t,
            "u": inputs[:, : self.L, :],
            "v": inputs[:, self.L :, :],
            "x": states,
            "y": outputs,
        }

    # Simulation methods
    def simulate(
        self,
        size: int,
        x: Optional[np.ndarray] = None,
        t0: float = 0.0,
        atol=1e-7,
        rtol=1e-6,
        domain: str = "discrete",
        precision: str = "high",
        method: Optional[str] = None,
        dtype=np.double,
        device: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """Simulate the analog frontend.

        The scheme is chosen along two self-explanatory axes:

        ``domain``
            * ``"discrete"`` -- simulate the discrete-time difference equation
              (the frontend is discretised first if needed). Fast; exact for a
              piecewise-constant (ZOH) input. This is the default and the most
              trusted path.
            * ``"continuous"`` -- integrate the true continuous-time dynamics.

        ``precision`` (only for ``domain="continuous"``)
            * ``"fast"`` -- analytic pre-computation; exact for a sinusoidal
              input, comparable in speed to the discrete scheme.
            * ``"high"`` -- numerical ODE solver (``atol``/``rtol``); general
              input, ~100x slower.

        Parameters
        ----------
        size: `int`
            the number of samples to simulate.
        x: `np.ndarray`, optional, shape=(N, J)
            the initial state, defaults to zero.
        t0: `float`, optional
            the initial time, defaults to 0.0.
        atol, rtol: `float`, optional
            ODE solver tolerances (``domain="continuous", precision="high"``).
        domain: `str`, optional
            ``"discrete"`` (default) or ``"continuous"``.
        precision: `str`, optional
            ``"fast"`` or ``"high"`` (default); continuous domain only.
        method: `str`, optional
            *Deprecated.* Explicit scheme name (``"dsim"``, ``"sin"``, ``"ode"``,
            ``"ode_full"``); overrides ``domain``/``precision`` when given.
        dtype: `data-type`, optional
            the floating-point data type, defaults to np.double.
        device: `str`, optional
            reserved for a future torch backend.

        Returns
        -------
        : dict[str, np.ndarray]
            a dictionary with keys ``t`` (size,), ``u`` (size, L, J),
            ``v`` (size, M, J), ``x`` (size, N, J), ``y`` (size, M, J).
        """
        if method is None:
            if domain == "discrete":
                method = "dsim"
            elif domain == "continuous":
                if precision == "fast":
                    method = "sin"
                elif precision == "high":
                    method = "ode"
                elif precision == "exact":
                    method = "ode_full"
                else:
                    raise ValueError(
                        f"Unknown precision {precision!r}; use 'fast' or 'high'"
                    )
            else:
                raise ValueError(
                    f"Unknown domain {domain!r}; use 'discrete' or 'continuous'"
                )

        if method == "dsim":
            return self.simulate_dt(size, x, t0, atol, rtol, dtype)
        elif method == "sin":
            return self.simulate_sin(size, x, t0, atol, rtol, dtype)
        elif method == "ode":
            return self.simulate_ode(size, x, t0, atol, rtol, dtype)
        elif method == "ode_full":
            return self.simulate_ode_full(size, x, t0, atol, rtol, dtype)
        else:
            raise ValueError(f"Unknown simulation method {method}")

    # ------------------------------------------------------------------
    # Shared simulation building blocks (one definition for every scheme)
    # ------------------------------------------------------------------
    def _simulate_alloc(self, size, x, t0, dtype):
        """Allocate buffers, inject noise, set the initial state and the t/u
        signals, and seed the first output/control. Shared by every scheme."""
        inputs = np.zeros((size, self.L + self.M, self.J), dtype=dtype)
        if self.state_covariance is not None:
            states = self._state_cov_cholesky @ self.rng.normal(
                size=(size, self.N, self.J)
            )
        else:
            states = np.zeros((size, self.N, self.J), dtype=dtype)
        if self.output_covariance is not None:
            outputs = self._output_cov_cholesky @ self.rng.normal(
                size=(size, self.M, self.J)
            )
        else:
            outputs = np.zeros((size, self.M, self.J), dtype=dtype)

        if x is None:
            x = np.zeros((self.N, self.J), dtype=dtype)
        states[0, :, :] = np.clip(
            x, self.state_min[:, np.newaxis], self.state_max[:, np.newaxis]
        )
        t = np.arange(size, dtype=dtype) * self.digital_control.dt + t0
        inputs[:, : self.L, :] = self.analog_signal.evaluate(t)
        outputs[0] += self.C[0] @ states[0] + self.D[0] @ inputs[0]
        inputs[0, self.L :, :] = self._quantize(outputs[0], 0)
        return inputs, states, outputs, t

    def _quantizer_params(self):
        """Pre-reshape the quantizer parameter arrays once (shape (M, 1)) and the
        per-(cyclic-)step update mask. The mask marks which control channels are
        observed (non-zero C/D row) and therefore update at a given step; the
        rest hold their previous value -- this is the discrete-time (``dsim``)
        convention, which is the canonical one."""
        dc = self.digital_control
        self._q_o_scale = dc._o_scale.reshape((self.M, 1))
        self._q_pre_gain = dc._pre_gain.reshape((self.M, 1))
        self._q_mid_thread = dc._mid_thread.reshape((self.M, 1))
        self._q_mid_rise = dc._mid_rise.reshape((self.M, 1))
        self._q_min = dc._min.reshape((self.M, 1))
        self._q_max = dc._max.reshape((self.M, 1))
        self._q_update = (
            np.abs(self.C).sum(axis=-1) + np.abs(self.D).sum(axis=-1)
        ) > 0.0

    def _quantize(self, output_i, i, prev_ctrl=None):
        """Inlined single-bit/multi-level quantizer + cyclic hold (canonical).

        ``output_i`` is the quantizer input at step ``i`` (shape (M, J)). Channels
        whose update mask is False hold ``prev_ctrl``. This is the one definition
        of the quantizer math, shared by every simulation scheme; it mirrors what
        ``simulate`` previously inlined in the discrete-time loop.
        """
        if not hasattr(self, "_q_update"):
            self._quantizer_params()
        quantized = self._q_o_scale * np.clip(
            2.0 * np.floor(self._q_pre_gain * output_i + self._q_mid_thread)
            + self._q_mid_rise,
            self._q_min,
            self._q_max,
        )
        if prev_ctrl is None:
            return quantized
        update = self._q_update[i % self._q_update.shape[0]][:, np.newaxis]
        return np.where(update, quantized, prev_ctrl)

    @staticmethod
    def _simulate_result(t, inputs, states, outputs, L):
        return {
            "t": t,
            "u": inputs[:, :L, :],
            "v": inputs[:, L:, :],
            "x": states,
            "y": outputs,
        }

    def simulate_dt(
        self,
        size: int,
        x: Optional[np.ndarray] = None,
        t0: float = 0.0,
        atol=None,
        rtol=None,
        dtype=np.double,
    ) -> dict[str, np.ndarray]:
        """Simulate the analog frontend

        Parameters
        ----------
        size: `int`
            the number of samples to simulate
        x: `np.ndarray`, optional, shape=(N, J)
            the initial state, defaults to zero.
        t0: `float`, optional
            the initial time, defaults to 0.0.
        atol: `float`, optional
            absolute tolerance for the integrator, defaults to 1e-12.
        rtol: `float`, optional
            relative tolerance for the integrator, defaults to 1e-7.
        state_covariance: `np.ndarray`, optional
            the state covariance matrix, defaults to None.
        output_covariance: `np.ndarray`, optional
            the output covariance matrix, defaults to None.
        dtype: `data-type`, optional
            the data type for the simulation, defaults to np.double.


        Returns
        -------
        : dict[str, np.ndarray]
            a dictionary containing the simulation results where
            - 't': the time vector, shape (size,)
            - 'u': the analog signal evaluated at t, shape (size, L, J)
            - 'v': the digital control signals, shape (size, M, J)
            - 'x': the state vector, shape (size, N, J)
            - 'y': the quantization input vector, shape (size, M, J)

        """

        if atol is None or rtol is None:
            logger.warning("atol and rtol are ignored in discrete-time simulation")
        if not self.is_discrete_time:
            logger.info(
                "Analog frontend is not discrete-time, discretizing before simulating..."
            )
            return self.discretize(dt=self.digital_control.dt).simulate_dt(
                size, x, t0, atol, rtol, dtype
            )

        # Discrete-time analog filter
        logging.info("Simulating discrete-time analog frontend")
        if not np.isclose(self.digital_control.dt % self.analog_filter.dt, 0):
            raise ValueError(
                f"Digital control sampling period {self.digital_control.dt:0.2e} must be a multiple of the analog filter sampling period {self.analog_filter.dt:0.2e}"
            )

        inputs, states, outputs, t = self._simulate_alloc(size, x, t0, dtype)
        slew_rate_dt = self.slew_rate * self.digital_control.dt

        # Compute simulation recursions
        for i in range(1, size):
            # state evolution
            states[i] += np.clip(
                self.A[i % self.A.shape[0]] @ states[i - 1]
                + self.B[i % self.B.shape[0]] @ inputs[i - 1],
                -slew_rate_dt[:, np.newaxis],
                slew_rate_dt[:, np.newaxis],
            )
            states[i] = np.clip(
                states[i],
                self.state_min[:, np.newaxis],
                self.state_max[:, np.newaxis],
            )
            # output computation
            outputs[i] += (
                self.C[i % self.C.shape[0]] @ states[i]
                + self.D[i % self.D.shape[0]] @ inputs[i]
            )

            # quantize + cyclic hold (shared canonical path)
            inputs[i, self.L :, :] = self._quantize(
                outputs[i], i, prev_ctrl=inputs[i - 1, self.L :, :]
            )
        result = {
            "t": t,
            "u": inputs[:, : self.L, :],
            "v": inputs[:, self.L :, :],
            "x": states,
            "y": outputs,
        }

        # Perform System identification and estimaton where
        # the first partitions of the signal are used for training
        # and the last partition is used for testing.
        if isinstance(self.analog_signal, PartitionedSignal):
            from numpy.lib.stride_tricks import sliding_window_view

            from .digital_backend import decimate

            slices = self.analog_signal.partition_indices()
            train_slices = slices[1:]

            K = 1 << 7
            DSR = 1

            # Stack all training partitions along J; strip K transient samples; decimate
            v_train = np.concatenate(
                [result["v"][:, :, slc] for slc in train_slices], axis=2
            )
            u_train = np.concatenate(
                [result["u"][:, :, slc] for slc in train_slices], axis=2
            )
            dec_v_train = decimate(v_train[K:], DSR, method="direct")
            dec_u_train = decimate(u_train[K:], DSR, method="direct")

            # Build least-squares system A w = b on decimated training data
            J_train = dec_v_train.shape[2]
            dec_size = dec_v_train.shape[0]
            batch_size = dec_size - K + 1
            x_window = sliding_window_view(
                dec_v_train, K, axis=0
            )  # (batch_size, M, J_train, K)
            N_train = batch_size * J_train
            A = np.empty((N_train, K * self.M + 1))
            A[:, :-1] = x_window.transpose(0, 2, 3, 1).reshape(N_train, -1)
            A[:, -1] = 1.0
            # h0 delta at K//2-1 → valid convolution shift = K - K//2
            shift = K - K // 2
            b = (
                dec_u_train[shift : shift + batch_size]
                .transpose(0, 2, 1)
                .reshape(N_train, -1)
            )

            sol = np.linalg.lstsq(A, b, rcond=None)
            h = sol[0][:-1].reshape(K, self.M, self.L)  # (K, M, L)
            h_offset = sol[0][-1]  # (L,)

            # Apply fitted filter to all J partitions (strip transient; decimate)
            J_all = result["v"].shape[2]
            dec_v_all = decimate(
                result["v"][K:], DSR, method="direct"
            )  # (dec_size, M, J_all)
            x_all = sliding_window_view(
                dec_v_all, K, axis=0
            )  # (batch_size, M, J_all, K)
            N_all = batch_size * J_all
            x_flat = x_all.transpose(0, 2, 3, 1).reshape(N_all, -1)
            u_hat = (
                (x_flat @ h.reshape(K * self.M, self.L) + h_offset)
                .reshape(batch_size, J_all, self.L)
                .transpose(0, 2, 1)
            )  # (batch_size, L, J_all)

            result["h"] = h  # (K, M, L)
            result["u_hat"] = u_hat  # (batch_size, L, J_all)

        return result

    def simulate_sin(
        self,
        size: int,
        x: Optional[np.ndarray] = None,
        t0: float = 0.0,
        atol=1e-12,
        rtol=1e-7,
        dtype=np.double,
    ) -> dict[str, np.ndarray]:
        """Simulate the analog frontend

        Parameters
        ----------
        size: `int`
            the number of samples to simulate
        x: `np.ndarray`, optional, shape=(N, J)
            the initial state, defaults to zero.
        t0: `float`, optional
            the initial time, defaults to 0.0.
        atol: `float`, optional
            absolute tolerance for the integrator, defaults to 1e-12.
        rtol: `float`, optional
            relative tolerance for the integrator, defaults to 1e-7.
        state_covariance: `np.ndarray`, optional
            the state covariance matrix, defaults to None.
        output_covariance: `np.ndarray`, optional
            the output covariance matrix, defaults to None.
        dtype: `np.dtype`, optional
            the floating point data type to use, defaults to np.double.

        Returns
        -------
        : dict[str, np.ndarray]
            a dictionary containing the simulation results where
            - 't': the time vector, shape (size,)
            - 'u': the analog signal evaluated at t, shape (size, L, J)
            - 'v': the digital control signals, shape (size, M, J)
            - 'x': the state vector, shape (size, N, J)
            - 'y': the quantization input vector, shape (size, M, J)

        """

        inputs, states, outputs, t = self._simulate_alloc(size, x, t0, dtype)

        # Sinusoidal special case
        if self.A.shape[0] != 1 or self.B.shape[0] != 1:
            raise NotImplementedError(
                "Only time invariant systems are supported for pre-computed input contribbutions, i.e. A.shape[0] == B.shape[0] == 1"
            )
        # exp(A * dt) computation
        A_d = _linalg.expm(self.A[0] * self.dt)
        C_d = self.C[:]
        D_d = self.D[:]
        # for i in range(D_d.shape[0]):
        #     D_d[i, self.L :] *= self.digital_control.evaluate(
        #     self.dt, np.ones((self.M, 1))
        # ).reshape((1, self.M))
        B_d = np.zeros((self.N, self.L + self.M, self.J), dtype=dtype)

        tmp_sig_vec = np.zeros((1, self.L + self.M, self.J), dtype=dtype)
        tmp_sig_vec[0, : self.L, :] = self.analog_signal.offset

        def derivative(t: float, x: np.ndarray) -> np.ndarray:
            tmp_sig_vec[0, self.L :, :] = self.digital_control.impulse_response(
                np.array([t])
            ).reshape((self.M, 1))
            return (
                np.tensordot(
                    self.A[0],
                    x.reshape((self.N, self.L + self.M, self.J)),
                    axes=[[1], [0]],
                )
                + self.B[0, :, :, np.newaxis] * tmp_sig_vec
            ).flatten()

        # Compute the control contributions
        res = _integrate.solve_ivp(
            derivative,
            (0.0, self.dt),
            np.zeros((self.N, self.L + self.M, self.J), dtype=dtype).flatten(),
            atol=atol,
            rtol=rtol,
            # method="Radau",
            # jac=block_diag(*[self.A[0] for _ in range(self.J)]),
        )
        B_d = res.y[:, -1].reshape((self.N, self.L + self.M, self.J))

        _, tf = self.transfer_function(
            np.array([2j * np.pi * self.analog_signal.frequency]), state_output=True
        )
        # shape (N, L)
        tf = tf[0, :, : self.L]

        # shape = (N, L)
        tf_mag = np.abs(tf)
        tf_phase = np.angle(tf)
        # shape = (size, N, L, J)
        tmp = (
            self.analog_signal.amplitude[np.newaxis, np.newaxis, :, :]
            * tf_mag[np.newaxis, :, :, np.newaxis]
        ) * np.sin(
            (
                self.analog_signal._angular_frequency[np.newaxis, np.newaxis, :, :]
                * t[:, np.newaxis, np.newaxis, np.newaxis]
            )
            + (
                tf_phase[np.newaxis, :, :, np.newaxis]
                + self.analog_signal.phase[np.newaxis, np.newaxis, :, :]
            )
        )

        # collapse inputs
        # shape(size, N, J)
        tmp = np.sum(tmp, axis=2)
        # shifted to index 1
        # shape = (size-1, N, J)
        pre_computed_inputs = tmp[1:, :, :] - np.tensordot(
            tmp[:-1, :, :], A_d, axes=[[1], [1]]
        ).transpose(0, 2, 1)
        # pre_computed_inputs = tmp[1:] - A_d @ tmp[:-1]
        # shape = (N, J)
        pre_computed_input_offset = np.sum(B_d[:, : self.L, :], axis=1)

        slew_rate_dt = self.slew_rate * self.digital_control.dt

        # Compute simulation recursions
        for i in range(1, size):
            # state evolution
            states[i] += np.clip(
                A_d @ states[i - 1]
                # control feedback
                + B_d[:, self.L :, 0] @ inputs[i - 1, self.L :, :]
                # pre-computed input signal contributions
                + pre_computed_input_offset
                + pre_computed_inputs[i - 1],
                -slew_rate_dt[:, np.newaxis],
                slew_rate_dt[:, np.newaxis],
            )
            states[i] = np.clip(
                states[i],
                self.state_min[:, np.newaxis],
                self.state_max[:, np.newaxis],
            )
            # output computation
            outputs[i] += (
                C_d[i % C_d.shape[0]] @ states[i] + D_d[i % D_d.shape[0]] @ inputs[i]
            )

            # quantize + cyclic hold (shared canonical path)
            inputs[i, self.L :, :] = self._quantize(
                outputs[i], i, prev_ctrl=inputs[i - 1, self.L :, :]
            )

        return self._simulate_result(t, inputs, states, outputs, self.L)

    # Simulation methods
    def simulate_ode(
        self,
        size: int,
        x: Optional[np.ndarray] = None,
        t0: float = 0.0,
        atol=1e-12,
        rtol=1e-7,
        dtype=np.double,
    ) -> dict[str, np.ndarray]:
        """Simulate the analog frontend

        Parameters
        ----------
        size: `int`
            the number of samples to simulate
        x: `np.ndarray`, optional, shape=(N, J)
            the initial state, defaults to zero.
        t0: `float`, optional
            the initial time, defaults to 0.0.
        atol: `float`, optional
            absolute tolerance for the integrator, defaults to 1e-12.
        rtol: `float`, optional
            relative tolerance for the integrator, defaults to 1e-7.
        state_covariance: `np.ndarray`, optional
            the state covariance matrix, defaults to None.
        output_covariance: `np.ndarray`, optional
            the output covariance matrix, defaults to None.
        dtype: `data-type`, optional
            the data type to use for the simulation, defaults to np.double.

        Returns
        -------
        : dict[str, np.ndarray]
            a dictionary containing the simulation results where
            - 't': the time vector, shape (size,)
            - 'u': the analog signal evaluated at t, shape (size, L, J)
            - 'v': the digital control signals, shape (size, M, J)
            - 'x': the state vector, shape (size, N, J)
            - 'y': the quantization input vector, shape (size, M, J)

        """

        # Full ode solver
        logging.info("Simulating continuous-time analog frontend")
        # Make sure the input signal is not piecewise constant,
        # otherwise the simulation can be better performed
        # by discretizing the analog frontend before simulation.
        if self.analog_signal.piecewise_constant:
            logging.warning(
                """"Piecewise constant input signal, discretizing
                            analog frontend before simulation may be beneficial."""
            )

        inputs, states, outputs, t = self._simulate_alloc(size, x, t0, dtype)

        K = max(self.A.shape[0], self.B.shape[0])
        Ad = np.zeros_like(self.A, dtype=dtype)
        Bd = np.zeros_like(self.B, dtype=dtype)
        x0 = np.zeros((self.N, self.J), dtype=dtype).flatten()
        derivatives = []
        for k in range(K):
            tmp_arg = np.vstack(
                (
                    np.hstack(
                        (self.A[k % self.A.shape[0]], self.B[k % self.B.shape[0]])
                    ),
                    np.zeros((self.L + self.M, self.N + self.L + self.M), dtype=dtype),
                )
            )
            tmp = _linalg.expm(tmp_arg * self.dt)
            Ad[k, :, :] = tmp[: self.N, : self.N]
            Bd[k, :, :] = tmp[: self.N, self.N :]

            def derivative(t: float, x: np.ndarray) -> np.ndarray:
                return (
                    # state evolution
                    self.A[k % self.A.shape[0]] @ x.reshape(self.N, self.J)
                    # input signal contributions
                    + self.B[k % self.B.shape[0], :, : self.L]
                    @ self.analog_signal.evaluate(np.array([t]))[0, :, :]
                ).flatten()

            derivatives.append(derivative)

        for i in range(1, size):
            res = _integrate.solve_ivp(
                derivatives[i % K],
                (t[i - 1], t[i]),
                x0,
                atol=atol,
                rtol=rtol,
                method="Radau",
                jac=block_diag(*[self.A[i % K] for _ in range(self.J)]),
                # method="DOP853",
            )
            states[i] += np.clip(
                Ad[i % Ad.shape[0]] @ states[i - 1]
                + Bd[i % Bd.shape[0], :, self.L :] @ inputs[i - 1, self.L :, :]
                + res.y[:, -1].reshape((self.N, self.J)),
                -self._slew_rate[:, np.newaxis],
                self._slew_rate[:, np.newaxis],
            )
            states[i] = np.clip(
                states[i],
                self.state_min[:, np.newaxis],
                self.state_max[:, np.newaxis],
            )
            # output computation
            outputs[i] += (
                self.C[i % self.C.shape[0]] @ states[i]
                + self.D[i % self.D.shape[0]] @ inputs[i]
            )
            # quantize + cyclic hold (shared canonical path)
            inputs[i, self.L :, :] = self._quantize(
                outputs[i], i, prev_ctrl=inputs[i - 1, self.L :, :]
            )

        return self._simulate_result(t, inputs, states, outputs, self.L)

    def simulate_ode_full(
        self,
        size: int,
        x: Optional[np.ndarray] = None,
        t0: float = 0.0,
        atol=1e-12,
        rtol=1e-7,
        dtype=np.longdouble,
    ) -> dict[str, np.ndarray]:
        """Simulate the analog frontend

        Parameters
        ----------
        size: `int`
            the number of samples to simulate
        x: `np.ndarray`, optional, shape=(N, J)
            the initial state, defaults to zero.
        t0: `float`, optional
            the initial time, defaults to 0.0.
        atol: `float`, optional
            absolute tolerance for the integrator, defaults to 1e-12.
        rtol: `float`, optional
            relative tolerance for the integrator, defaults to 1e-7.
        state_covariance: `np.ndarray`, optional
            the state covariance matrix, defaults to None.
        output_covariance: `np.ndarray`, optional
            the output covariance matrix, defaults to None.
        dtype: `data-type`, optional
            the data type to use for the simulation, defaults to np.longdouble.


        Returns
        -------
        : dict[str, np.ndarray]
            a dictionary containing the simulation results where
            - 't': the time vector, shape (size,)
            - 'u': the analog signal evaluated at t, shape (size, L, J)
            - 'v': the digital control signals, shape (size, M, J)
            - 'x': the state vector, shape (size, N, J)
            - 'y': the quantization input vector, shape (size, M, J)

        """

        logging.info("Simulating continuous-time analog frontend")
        # Make sure the input signal is not piecewise constant,
        # otherwise the simulation can be better performed
        # by discretizing the analog frontend before simulation.
        if self.analog_signal.piecewise_constant:
            logging.warning(
                """"Piecewise constant input signal, discretizing
                            analog frontend before simulation may be beneficial."""
            )

        inputs, states, outputs, t = self._simulate_alloc(size, x, t0, dtype)

        K = max(self.A.shape[0], self.B.shape[0])
        derivatives = []
        for k in range(K):

            def derivative(t: float, x: np.ndarray, *args) -> np.ndarray:
                # return (
                #     # state evolution
                #     self.A[k % self.A.shape[0]] @ x.reshape(self.N, self.J)
                #     # input signal contributions
                #     + self.B[k % self.B.shape[0], :, : self.L]
                #     @ self.analog_signal.evaluate(np.array([t]))[0, :, :]
                #     # control feedback
                #     + self.B[k % self.B.shape[0], :, self.L :]
                #     # args[0] is the current time and args[1] the quantizer input
                #     @ self.digital_control.evaluate(
                #         t - args[0], args[1][:, np.newaxis, :]
                #     )
                # ).flatten()
                return np.clip(
                    # state evolution
                    self.A[k % self.A.shape[0]] @ x.reshape(self.N, self.J)
                    # input signal contributions
                    + self.B[k % self.B.shape[0], :, : self.L]
                    @ self.analog_signal.evaluate(np.array([t]))[0, :, :]
                    # control feedback
                    + self.B[k % self.B.shape[0], :, self.L :]
                    # args[0] is the current time and args[1] the quantizer input
                    @ self.digital_control.evaluate(
                        t - args[0], args[1][:, np.newaxis, :]
                    ).reshape((self.M, self.J)),
                    -self.slew_rate[:, np.newaxis],
                    self.slew_rate[:, np.newaxis],
                ).flatten()

            derivatives.append(derivative)

        for i in range(1, size):
            res = _integrate.solve_ivp(
                derivatives[i % K],
                (t[i - 1], t[i]),
                states[i - 1].flatten(),
                args=(t[i - 1], outputs[i - 1]),
                atol=atol,
                rtol=rtol,
                jac=block_diag(*[self.A[i % K] for _ in range(self.J)]),
                method="Radau",
                # method="DOP853",
            )
            # slew_rate_dt = self.slew_rate * self.digital_control.dt
            # states[i] += np.clip(
            #     res.y[:, -1].reshape((self.N, self.J)),
            #     -slew_rate_dt[:, np.newaxis],
            #     slew_rate_dt[:, np.newaxis],
            # )
            states[i] += res.y[:, -1].reshape((self.N, self.J))
            states[i] = np.clip(
                states[i],
                self.state_min[:, np.newaxis],
                self.state_max[:, np.newaxis],
            )
            # output computation
            outputs[i] += (
                self.C[i % self.C.shape[0]] @ states[i]
                + self.D[i % self.D.shape[0]] @ inputs[i]
            )
            # quantize + cyclic hold (shared canonical path)
            inputs[i, self.L :, :] = self._quantize(
                outputs[i], i, prev_ctrl=inputs[i - 1, self.L :, :]
            )

        return self._simulate_result(t, inputs, states, outputs, self.L)

    def discretize(
        self, dt: float, atol: float = 1e-15, rtol: float = 1e-10
    ) -> "AnalogFrontend":
        """Discretize the analog frontend

        Discretize the analog frontend.


        Parameters
        ----------
        dt : `float`
            the sampling period
        atol: `float`, optional
            absolute tolerance for the integrator, defaults to 1e-15.
        rtol: `float`, optional
            relative tolerance for the integrator, defaults to 1e-10.

        Returns
        -------
        : :py:class:`cbadc.analog_frontend.AnalogFrontend`
            returns a discretized analog frontend instance
        """
        if self.is_discrete_time:
            logger.warning("Analog frontend is already discrete-time.")
            return self

        C_d = self.C[:]
        D_d = self.D[:]
        # If return to zero DAC there is no direct path at the end of the time period.

        # assuming piecewise constant input signal
        if not self.analog_signal.piecewise_constant:
            logger.warning(
                "Non piecewise constant input signal. The discretization may not be accurate."
            )

        K = max(self.A.shape[0], self.B.shape[0])
        if (
            # self.analog_signal.piecewise_constant and
            self.digital_control.dac_waveform == "nrz"
        ):
            # largest repetition of A and B matrices
            A_d = np.zeros((K, self.N, self.N), dtype=self.A.dtype)
            B_d = np.zeros((K, self.N, self.L + self.M), dtype=self.B.dtype)
            for i in range(K):
                tmp_arg = np.vstack(
                    (
                        np.hstack(
                            (self.A[i % self.A.shape[0]], self.B[i % self.B.shape[0]])
                        ),
                        np.zeros(
                            (self.L + self.M, self.N + self.L + self.M),
                            dtype=self.A.dtype,
                        ),
                    )
                )
                tmp = _linalg.expm(tmp_arg * dt)
                A_d[i, :, :] = tmp[: self.N, : self.N]
                B_d[i, :, :] = tmp[: self.N, self.N :]
        else:
            delay_steps = np.max(self.digital_control.delay_steps())
            additional_states: int = delay_steps * self.M

            A_d = np.zeros(
                (K, self.N + additional_states, self.N + additional_states),
                dtype=self.A.dtype,
            )
            B_d = np.zeros(
                (K, self.N + additional_states, self.L + self.M), dtype=self.B.dtype
            )
            C_d = np.zeros(
                (self.C.shape[0], self.M, self.N + additional_states),
                dtype=self.C.dtype,
            )
            C_d[:, :, : self.N] = self.C[:, :, :]
            for i in range(K):
                A_d[i, : self.N, : self.N] = _linalg.expm(
                    self.A[i % self.A.shape[0], :, :] * dt
                )

                tmp_sig_vec = np.zeros(
                    (1, self.L + self.M + additional_states), dtype=self.A.dtype
                )
                B_temp = np.zeros(
                    (self.N, self.L + self.M + additional_states), dtype=self.B.dtype
                )
                B_temp[:, : self.L + self.M] = self.B[
                    i % self.B.shape[0], :, : self.L + self.M
                ]
                for k in range(1, delay_steps + 1):
                    B_temp[
                        :,
                        self.L + k * self.M : self.L + (k + 1) * self.M,
                    ] = self.B[i % self.B.shape[0], :, self.L :]

                def derivative(t: float, x: np.ndarray) -> np.ndarray:
                    t_array = np.array([t])
                    tmp_sig_vec[0, : self.L] = self.analog_signal.impulse_response(
                        t_array
                    )[:, :, 0]
                    tmp_sig_vec[0, self.L :] = self.digital_control.impulse_response(
                        t_array + self.digital_control.dt * np.arange(delay_steps + 1)
                    ).T.flatten()

                    return (
                        self.A[i % self.A.shape[0]] @ x.reshape((self.N, -1))
                        + B_temp * tmp_sig_vec
                    ).flatten()

                # Compute input signal contributions
                res = _integrate.solve_ivp(
                    derivative,
                    (0.0, dt),
                    np.zeros(
                        self.N * (self.L + self.M + additional_states),
                        dtype=self.A.dtype,
                    ),
                    atol=atol,
                    rtol=rtol,
                    method="DOP853",
                )
                # shape(N, L + M + additional_states)
                x_vals = res.y[:, -1].reshape((self.N, -1))
                # Bd = [[ Bu, Bc ], [0, I]]
                B_d[i, : self.N, : self.L + self.M] = x_vals[:, : self.L + self.M]
                if additional_states > 0:
                    # A_d matrix structure:
                    # [[A, B2, B3, ...],
                    #  [0, 0, ...],
                    #  [0, I, 0, ...],
                    #  [0, 0, I, ...]]
                    A_d[i] = np.vstack(
                        (
                            np.hstack(
                                (
                                    A_d[i, : self.N, : self.N],
                                    x_vals[:, self.L + self.M :].reshape(
                                        (self.N, additional_states)
                                    ),
                                    # np.zeros((self.N, additional_states), dtype=float),
                                )
                            ),
                            np.zeros((additional_states, self.N + additional_states)),
                        )
                    )
                    # create delay elements
                    for k in range(1, delay_steps):
                        A_d[
                            i,
                            self.N + k * self.M : self.N + (k + 1) * self.M,
                            self.N + (k - 1) * self.M : self.N + k * self.M,
                        ] = np.eye(self.M, dtype=float)
                    B_d[i, self.N : self.N + self.M, self.L :] = np.eye(
                        self.M, dtype=float
                    )
                    # possible remove below should this be duplicates?
                    # A_d[: self.N, self.N :] = res.y[
                    #     self.N * (self.L + self.M) :, -1
                    # ].reshape((self.N, additional_states))
                    # C_d = [C, 0, 0, ...]
                    # C_d[i] = np.hstack(
                    #     (C_d[i], np.zeros((self.M, additional_states), dtype=float))
                    # )
        if K == 1 and C_d.shape[0] == 1 and D_d.shape[0] == 1:
            analog_filter = StateSpace(A_d[0], B_d[0], C_d[0], D_d[0], dt=dt)
        else:
            analog_filter = CyclicStateSpace(A_d, B_d, C_d, D_d, dt=dt)
        digital_control = _deepcopy(self.digital_control)
        analog_signal = _deepcopy(self.analog_signal)

        # Propagate state constraints, extending for any additional delay states
        new_N = analog_filter.A.shape[-1]
        extra = new_N - self.N
        state_max = (
            np.concatenate([self.state_max, np.full(extra, np.inf)])
            if extra > 0
            else self.state_max.copy()
        )
        state_min = (
            np.concatenate([self.state_min, np.full(extra, -np.inf)])
            if extra > 0
            else self.state_min.copy()
        )
        slew_rate = (
            np.concatenate([self.slew_rate, np.full(extra, np.inf)])
            if extra > 0
            else self.slew_rate.copy()
        )
        # Propagate the state-noise intensity as the *discrete* per-step
        # covariance, so the injected noise magnitude is identical whether the
        # user sets it before or after discretising.  The van-Loan integral uses
        # the original continuous-time A; any added delay states carry no process
        # noise, so the covariance is zero-padded into the enlarged state space.
        if self.state_covariance is None:
            state_cov = None
        else:
            state_cov_d = _discrete_process_noise_cov(
                self.A[0, :, :], self.state_covariance, dt
            )
            if extra > 0:
                padded = np.zeros((new_N, new_N), dtype=float)
                padded[: self.N, : self.N] = state_cov_d
                state_cov = padded
            else:
                state_cov = state_cov_d

        return AnalogFrontend(
            analog_filter,
            digital_control,
            analog_signal,
            state_covariance=state_cov,
            output_covariance=_deepcopy(self.output_covariance),
            slew_rate=slew_rate,
            state_max=state_max,
            state_min=state_min,
        )

    # Transfer function methods
    def transfer_function(
        self,
        jw: np.ndarray,
        open_loop=True,
        input_index: Optional[int] = None,
        output_index: Optional[int] = None,
        state_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the transfer function of the analog frontend

        Parameters
        ----------
        jw: np.ndarray, shape=(size,), dtype=complex
            the angular frequency vector
        open_loop: bool
            whether to compute the open-loop transfer function
        input_index: int, optional
            the input index, defaults to None
        output_index: int, optional
            the output index, defaults to None
        state_output: bool, optional
            whether to return the state vector, defaults to False

        Returns
        -------
        :py:class:`numpy.ndarray`
            the complex angular frequency vector
        :py:class:`numpy.ndarray`, shape=(size, M, L)
            the transfer function
        """
        if open_loop:
            temp_filter = StateSpace(
                self.A[0],
                self.B[0],
                self.C[0],
                self.D[0],
            )
        else:
            # Feedback transfer function, i.e.,
            Bl = self.B[0, :, : self.L]
            Bm = self.B[0, :, self.L :]
            Dl = self.D[0, :, : self.L]
            Dm = self.D[0, :, self.L :]
            I_DM = np.linalg.inv(np.eye(self.M) - Dm)
            A_new = self.A[0] + Bm @ I_DM @ self.C[0]
            B_new = Bl + I_DM @ Dl
            C_new = I_DM @ self.C[0]
            D_new = I_DM @ Dl

            temp_filter = StateSpace(A_new, B_new, C_new, D_new)

        A: np.ndarray = temp_filter.A
        B: np.ndarray = temp_filter.B
        C: np.ndarray = temp_filter.C
        D: np.ndarray = temp_filter.D

        if input_index is not None:
            B: np.ndarray = B[:, input_index].reshape((self.N, -1))
            D: np.ndarray = D[:, input_index].reshape((self.M, -1))

        if output_index is not None:
            C: np.ndarray = C[output_index].reshape((-1, self.N))
            D: np.ndarray = D[output_index]

        if self.is_discrete_time:
            # convert jw to z
            z = np.exp(jw * self.digital_control.dt)
            jw = z

        # Solve the transfer function
        # H(s) = C (sI - A)^-1 B + D
        #
        # (sI - A) x = B
        # C x + D

        # shape (size, N, N)
        s_minus_A = (
            jw.reshape((-1, 1, 1)) * np.eye(self.N, dtype=complex)[np.newaxis, :, :]
            - A[np.newaxis, :, :]
        )
        # shape (size, N, L+M)
        x = np.linalg.solve(s_minus_A, B)
        if state_output:
            return jw, x
        # shape (size, M, L+M )
        h = C @ x + D
        return jw, h

    def quadrate(self, wp: float) -> "AnalogFrontend":
        """Quadrature the analog frontend

        Parameters
        ----------
        wp: `float`
            the quadrature frequency

        Returns
        -------
        : :py:class:`cbadc.analog_frontend.AnalogFrontend`
            returns a quadrature analog frontend instance
        """
        # Aq = [[A, -wp I], [wp I, A]]
        K = max(self.A.shape[0], self.B.shape[0])
        H = max(self.C.shape[0], self.D.shape[0])
        Aq = np.zeros((K, 2 * self.N, 2 * self.N), dtype=float)
        Bq = np.zeros((K, 2 * self.N, 2 * (self.L + self.M)), dtype=float)
        Cq = np.zeros((H, 2 * self.M, 2 * self.N), dtype=float)
        Dq = np.zeros((H, 2 * self.M, 2 * (self.L + self.M)), dtype=float)
        for k in range(K):
            Aq[k] = _linalg.block_diag(self.A, self.A)
            Aq[k, : self.N, self.N :] = -wp * np.eye(self.N)
            Aq[k, self.N :, : self.N] = wp * np.eye(self.N)
            # Bq = [[B, 0], [0, B]]
            Bq[k] = _linalg.block_diag(self.B, self.B)
        for h in range(H):
            # Cq = [[C, 0], [0, C]]
            Cq[h] = _linalg.block_diag(self.C, self.C)
            # Dq = [[D, 0], [0, D]]
            Dq[h] = _linalg.block_diag(self.D, self.D)

        if K == 1 and H == 1:
            analog_filter = StateSpace(Aq[0], Bq[0], Cq[0], Dq[0])
        else:
            analog_filter = CyclicStateSpace(Aq, Bq, Cq, Dq)
        alpha = np.tile(self.digital_control.alpha, 2)
        beta = np.tile(self.digital_control.beta, 2)

        digital_control = DigitalControl(2 * self.M, self.dt, alpha, beta)
        analog_signal = ConcatenatedSignals(self.analog_signal, self.analog_signal)
        return AnalogFrontend(analog_filter, digital_control, analog_signal)

    def active_RC(self, GBWP: float, DC_gain: float) -> "AnalogFrontend":
        """Compute the active RC analog frontend

        Parameters
        ----------
        GBWP: `float`
            the gain-bandwidth product
        DC_gain: `float`
            the DC gain

        Returns
        -------
        : :py:class:`cbadc.analog_frontend.AnalogFrontend`
            returns an active RC analog frontend instance where
            the internal integrator states are appended as x[N:].
        """
        # DC W_BW/(s-W_BW)
        omega_BW = 2 * np.pi * GBWP / DC_gain
        I_N = np.eye(self.N, dtype=float)
        A = np.vstack(
            (
                np.hstack((self.A, I_N)),
                np.hstack((DC_gain * omega_BW * I_N, -omega_BW * I_N)),
            )
        )
        B = np.vstack((self.B, np.zeros((self.N, self.L + self.M), dtype=float)))
        C = np.hstack((self.C, np.zeros((self.M, self.N), dtype=float)))
        D = self.D

        raise NotImplementedError("This requires more thinking.")
        return AnalogFrontend(StateSpace(A, B, C, D), self.digital_control)

    def global_control(self) -> "AnalogFrontend":
        """Compute and generate global control"""
        # TODO implement global control
        raise NotImplementedError

    def wiener_filter(
        self, eta2: Optional[float] = None, OSR: Optional[int] = None
    ) -> "WienerFilter":
        """Compute the Wiener filter

        resulting in the signal transfer function
        STF(jw) = G^H(jw) (G(jw)G^H(jw) + eta2 I)^-1 G(jw)
        where G(jw) is the transfer function of the
        analog frontend

        Parameters
        ----------
        eta2: `float`
            the bandwidth of the Wiener filter

        Returns
        -------
        : :py:class:`cbadc.wiener_filter.WienerFilter`
            returns a Wiener filter instance
        """
        from .digital_backend import WienerFilter

        if eta2 is not None and OSR is not None:
            logger.warning("Both eta2 and OSR are provided, using eta2")
        elif eta2 is not None:
            eta2 = float(eta2)
        elif eta2 is None and OSR is not None:
            # Compute the signal transfer function
            jomega_Bw = 1j * np.pi / (OSR * self.dt)
            _, tf = self.transfer_function(
                np.array([jomega_Bw]), input_index=0, output_index=-1
            )
            eta2: float = np.abs(tf[0, 0, 0]) ** 2
        else:
            raise ValueError("eta2 or OSR must be provided")
        return WienerFilter(self, eta2)

    def black_box_estimator(
        self,
        DSR: int,
        K: int = 1 << 6,
        max_amplitude: float = 1.0,
        sim_size: int = 1 << 16,
        J: int = 4,
        seed: int = 90128310230123,
    ):
        from .digital_backend import DataAidedEstimator

        return DataAidedEstimator(self, DSR, K, max_amplitude, sim_size, J, seed)

    def calibrate(
        self,
        DSR: int,
        K: int = 1 << 8,
        J: int = 4,
        sim_size: int = 1 << 16,
        max_amplitude: float = 1.0,
        reference=None,
        seed: int = 90128310230123,
        fit: str = "lstsq",
    ):
        """Calibrate a data-aided reconstruction filter for this frontend.

        Drives the frontend with a known, persistently-exciting ``reference``
        (a full-scale random sequence by default, ``J`` sequences in parallel),
        simulates, decimates by ``DSR`` and fits a ``K``-tap FIR by least
        squares. This is the recommended (data-aided) readout; the analytical
        :meth:`wiener_filter` remains available for the model-based path.

        Parameters
        ----------
        DSR : int
            decimation / down-sampling ratio (typically the OSR).
        K : int, optional
            number of FIR taps, defaults to 256.
        J : int, optional
            number of parallel reference sequences in one simulation, default 4.
        sim_size : int, optional
            calibration simulation length, defaults to ``1 << 16``.
        max_amplitude : float, optional
            amplitude of the generated reference, defaults to 1.0.
        reference : :py:class:`cbadc.analog_signal.AnalogSignal`, optional
            a custom calibration reference; if ``None`` a full-scale uniform
            reference is generated.
        seed : int, optional
            RNG seed for the generated reference.

        Returns
        -------
        : :py:class:`cbadc.digital_backend.DataAidedEstimator`
            the calibrated estimator; call ``estimator.reconstruct(v)`` to
            estimate the input from control signals.
        """
        from .digital_backend import DataAidedEstimator

        return DataAidedEstimator(
            self,
            DSR,
            K,
            max_amplitude,
            sim_size,
            J,
            seed,
            reference=reference,
            fit=fit,
        )

    def simulateSNR(
        self,
        OSR: int,
        amp_dB: Optional[np.ndarray] = None,
        f0: float = 0.0,
        f: Optional[float] = None,
        k: int = 13,
        K: int = 1 << 8,
        method: str = "dsim",
        debug: bool = False,
    ):
        """Simulate the SNR of the analog frontend


        Parameters
        ----------
        amplitude: `np.ndarray`, optional
            the amplitude of the input signal, defaults to None.
        OSR: `int`
            the oversampling ratio
        f0: `float`, optional
            the input tone frequency, defaults to 0.0.
        f: `float`, optional
            the input tone frequency, defaults to None.
        k: `int`, optional
            the FFT size, defaults to 13.
        K: `int`, optional
            number of taps of reconstruction filter, defaults to 64.
        method: `str`, optional
            the simulation method, defaults to "dsim". Valid options are:

            * 'dsim': simulate the discrete-time analog frontend

            * 'sin': simulate the continuous-time analog frontend for sinusoidal input

            * 'ode': simulate the continuous-time analog frontend using an ODE solver with discrete time steps.

            * 'ode_full': simulate the continuous-time analog frontend using an ODE solver with full time steps.
        debug: `bool`, optional
            whether to plot the resulting spectrums for debugging, defaults to False.
        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(J,)
            the signal-to-noise ratio
        : :py:class:`numpy.ndarray`, shape=(J,)
            the corresponding amplitude of the input signal
        : `float`
            the average power of the input signal
        """
        if amp_dB is None:
            amp_dB = np.concatenate(
                (
                    np.arange(-120, -20 + 1, 10),
                    np.array((-15,)),
                    np.arange(-10, 3),
                )
            ).reshape((1, -1))
        elif isinstance(amp_dB, (int, float)):
            amp_dB = np.array([amp_dB], dtype=float)
        elif not isinstance(amp_dB, np.ndarray):
            raise ValueError("amp must be a numpy array")

        if not isinstance(OSR, int):
            raise ValueError("OSR must be an integer")

        if amp_dB.ndim != 2:
            raise ValueError(
                "amp must be a 2D numpy array with shape (L, J) where L is the number "
                "of dimensions and J is the number of parallel inputs"
            )

        # dB to linear scale
        amp_lin = 10 ** (amp_dB / 20)

        fft_bins = 1 << k
        f = np.round(f0 + 0.5 * fft_bins / (2 * OSR))

        # if f is None:
        #     f = f0 + 0.5 / (OSR * 2)  # Halfway across the band
        # if np.abs(f - f0) > 0.5 / OSR:
        #     logger.warning("The input tone is out-of-band.")
        # fft_bins = 1 << k
        # if fft_bins < (OSR << 4):
        #     logger.warning(
        #         "The FFT size is too small.",
        #         "Increasing k to accommodate a large oversampling ratio.",
        #     )
        #     k = int(np.ceil(np.log2(OSR << 4)))
        #     fft_bins = 1 << k
        # f_int = int(np.round(f * fft_bins))
        # if np.abs(f_int) < 2:
        #     logger.warning(
        #         "The input tone is too close to DC.",
        #         "Increasing k to accommodate a low input frequency.",
        #     )
        #     k = np.ceil(np.log2(1.0 / np.abs(f)))
        #     fft_bins = 1 << k
        #     f_int = 2
        warm_up = 1 << 7
        window = np.hanning(fft_bins)
        # if f0 == 0.0:
        #     in_band_bins = fft_bins // 2 + np.arange(
        #         3, np.round(fft_bins / (2 * OSR)) + 1, dtype=int
        #     )
        #     f_int -= 2

        # before overwriting the analog signal, store the old one
        _old_analog_signal = _deepcopy(self.analog_signal)

        self.analog_signal = Sinusoidal(
            amp_lin, f / fft_bins * self.fs * np.ones_like(amp_dB)
        )
        # print(f"Simulating SNR at frequency: {f / fft_bins * self.fs} Hz")
        sim = self.simulate(warm_up + fft_bins * OSR, method=method)

        avg_pow = np.sum(np.mean(self.avg_power(sim["x"]), axis=1))

        # OSR = 1 / (2 * dt * BW)
        # df = self.wiener_filter(OSR=OSR)
        # # shape = (fft_bins+warm_up, J)
        # u_hat = df.evaluate(sim["v"])[:, :, :]

        df = self.black_box_estimator(
            OSR, K=K, max_amplitude=1.0, sim_size=1 << 16, J=1
        )

        u_hat = df.convolve(sim["v"], DSR=OSR)

        hwfft = np.fft.rfft(u_hat[-fft_bins:, 0, :] * window[:, np.newaxis], axis=0)
        # print(f"FFT bins: {hwfft.shape[0]}")
        # shape = (J,)
        # snr = self.calculateSNR_from_fft(hwfft[in_band_bins - 1])
        snr = self.calculateSNR_from_fft(hwfft[3:-3], extra_bins=10)
        # reset the analog signal

        if debug:
            import matplotlib.pyplot as plt

            for j in range(amp_dB.shape[1] - 3):
                plt.figure("spectrum")
                plt.semilogx(
                    20 * np.log10(np.abs(hwfft[:, j]) / np.sqrt(np.sum(window**2) / 2)),
                    label=f"Amplitude: {amp_dB[0, j]} dB",
                )
            # plt.title(f"FFT of Reconstructed Signal (Amplitude: {amp_dB[0,j]} dB)")
            plt.xlabel("FFT Bins")
            plt.ylabel("Magnitude (dB)")
            plt.legend()
            plt.grid()
            plt.show()

        self.analog_signal = _old_analog_signal

        return snr, amp_dB.flatten(), avg_pow

    def calculateSNR_from_fft(self, fft: np.ndarray, extra_bins: int = 2):
        """Calculate the SNR from an FFT

        Parameters
        ----------
        fft: `np.ndarray`, shape=(size, ...)
            the FFT
        extra_bins: `int`, optional
            the number of extra bins surronding the input frequency bin
            to include in the signal power computation,
            defaults to 1.

        Returns
        -------
        : `np.ndarray`, shape=(...)
            the signal-to-noise ratio
        """
        # indicate signal bins
        # shape = (J,)
        f_argmax = np.argmax(np.abs(fft), axis=0)
        f_argmax = int(np.median(f_argmax))

        signal_index = f_argmax + np.arange(-extra_bins // 2, extra_bins // 2 + 1)
        noise_index = np.setdiff1d(np.arange(fft.shape[0]), signal_index)

        signal_power = np.linalg.norm(
            fft[signal_index],
            ord=2,
            axis=0,
        )
        noise_power = np.linalg.norm(fft[noise_index], ord=2, axis=0)
        no_noise = np.isclose(noise_power, np.zeros_like(noise_power))
        snr = np.zeros_like(signal_power, dtype=float)
        snr[no_noise] = np.inf
        snr[~no_noise] = 20 * np.log10(signal_power[~no_noise] / noise_power[~no_noise])
        return snr

    # Synthesis Methods
    @staticmethod
    def chain_of_integrators(**kwargs) -> tuple["AnalogFrontend", float]:
        """Parameterize a chain-of-integrator analog frontend

        Returns a parameterized analog system and
        digital control corresponding to a given
        target specification.

        Parameters
        ----------
        ENOB: `float`
            targeted effective number of bits.
        N: `int`
            system order.
        BW: `float`
            target bandwidth
        xi: `float`, `optional`
            a proportionality constant, defaults to 0.0016.
        local_feedback: `bool`, `optional`
            include local feedback, defaults to False.
        excess_delay: `float`, `optional`
            delay control actions by an excess delay, defaults to 0.
        finite_gain: `bool`, `optional`
            include finite gain, defaults to False.


        Returns
        -------
        : :py:class:`cbadc.analog_frontend.AnalogFrontend`
            returns an analog frontend instance
        : `float`
            the oversampling ratio
        """
        finite_gain = kwargs.get("finite_gain", False)

        if all(param in kwargs for param in ("ENOB", "N", "BW")):
            SNR = enob_to_snr(kwargs["ENOB"])
            snr = snr_from_dB(SNR)
            N = kwargs["N"]
            omega_3dB = 2.0 * np.pi * kwargs["BW"]
            xi = kwargs.get("xi", 2.3e-3)
            gamma = (xi / _g_i_chain_of_integrators(N) * snr) ** (1.0 / (2.0 * N))
            beta = -gamma * omega_3dB
            if "local_feedback" in kwargs and kwargs["local_feedback"] is True:
                rho = -omega_3dB / gamma
            else:
                rho = kwargs.get("rho", 0.0)
            kappa = beta
            dt = 1.0 / np.abs(2.0 * beta)
            all_ones = np.ones(N)
            analog_filter = ChainOfIntegrators(
                beta * all_ones, rho * all_ones, kappa * np.eye(N)
            )
            if finite_gain:
                analog_filter.A += -omega_3dB / (gamma ** (2 * N)) * np.eye(N)
            t0 = dt * kwargs.get("excess_delay", 0.0)

            alpha = np.zeros(N)
            beta = np.ones(N) + (t0 / dt)
            OSR = 1.0 / (2.0 * dt * kwargs["BW"])
            digital_control = DigitalControl(N, dt, alpha, beta)
            return AnalogFrontend(analog_filter, digital_control), OSR

        elif all(param in kwargs for param in ("SNR", "N", "BW")):
            return AnalogFrontend.chain_of_integrators(
                ENOB=snr_to_enob(kwargs["SNR"]), **kwargs
            )
        elif all(param in kwargs for param in ("OSR", "N", "BW")):
            N = kwargs["N"]
            BW = kwargs["BW"]
            OSR = kwargs["OSR"]
            delta = kwargs.get("delta", 1.0)
            dt = 1.0 / (2 * BW * OSR)
            beta = delta / (2 * dt)
            kappa_vec = np.array([beta * delta**n for n in range(N)])
            rho = kwargs.get("rho", 0.0)
            analog_filter = ChainOfIntegrators(
                beta * np.ones(N),
                rho * np.ones(N),
                np.diag(kappa_vec),
            )
            t0 = dt * kwargs.get("excess_delay", 0.0)
            alpha = np.zeros(N)
            beta = np.ones(N) + (t0 / dt)
            digital_control = DigitalControl(N, dt, alpha, beta)
            return AnalogFrontend(analog_filter, digital_control), OSR
        elif all(param in kwargs for param in ("OSR", "N", "T")):
            N = kwargs["N"]
            dt = kwargs["T"]
            OSR = kwargs["OSR"]
            BW = 1.0 / (2 * dt * OSR)
            rho = kwargs.get("rho", 0.0)
            return (
                AnalogFrontend.chain_of_integrators(N=N, OSR=OSR, BW=BW, rho=rho),
                OSR,
            )
        raise NotImplementedError

    @staticmethod
    def leapfrog(**kwargs) -> tuple["AnalogFrontend", float]:
        """Parameterize a leap-frog ADC

        Returns a parameterized analog system and
        digital control corresponding to a given
        target specification.

        Parameters
        ----------
        ENOB: `float`
            targeted effective number of bits.
        N: `int`
            system order.
        BW: `float`
            target bandwidth
        xi: `float`, `optional`
            a proportionality constant, defaults to 4e-3.
        local_feedback: `bool`, `optional`
            include local feedback, defaults to False.
        excess_delay: `float`, `optional`
            delay control actions by an excess delay, defaults to 0.
        bw_3dB: `bool`, `optional`
            scale ``omega_p`` so the open-loop signal transfer function has its
            3 dB edge exactly at ``BW`` (extends the stock design, whose edge
            sits at ~0.75-0.9 BW). Only used in the (ENOB, N, BW) branch.
            Defaults to False.


        Returns
        -------
        : :py:class:`cbadc.analog_frontend.AnalogFrontend`
            returns an analog frontend instance
        : `float`
            the oversampling ratio
        """

        if all(param in kwargs for param in ("ENOB", "N", "BW")):
            SNR = enob_to_snr(float(kwargs["ENOB"]))
            snr = snr_from_dB(SNR)
            N = int(kwargs["N"])
            omega_BW = 2.0 * np.pi * float(kwargs["BW"])
            xi = kwargs.get("xi", 4e-3)
            delta = kwargs.get("delta", 1.0)
            gamma_over_delta = (xi / _g_i_leapfrog(N) * snr) ** (1.0 / (2.0 * N))
            gamma = gamma_over_delta * delta
            omega_p = omega_BW / 2.0
            # omega_p /= np.cos(N * np.pi / (N + 1.0))
            beta = -omega_p * (2.0 * gamma)
            alpha = omega_p / (2.0 * gamma)
            if kwargs.get("bw_3dB", False):
                # extend omega_p so the loop-filter 3 dB edge sits at BW
                # (stock omega_p=omega_BW/2 places it below BW)
                r = _leapfrog_3dB_factor(alpha, beta, N, float(kwargs["BW"]))
                beta *= r
                alpha *= r
            rho = 0
            dt = 1.0 / np.abs(2.0 * omega_BW * gamma / delta)
            OSR = 1.0 / (2.0 * dt * kwargs["BW"])
            kappa = beta
        elif all(param in kwargs for param in ("OSR", "N", "BW")):
            N = int(kwargs["N"])
            BW = float(kwargs["BW"])
            OSR = float(kwargs["OSR"])
            dt = 1.0 / (2 * OSR * BW)
            omega_BW = 2.0 * np.pi * BW
            delta = kwargs.get("delta", 1.0)
            gamma = delta * OSR / (2 * np.pi)
            beta = -gamma * omega_BW
            alpha = omega_BW / (4 * gamma)
            kappa = beta
            rho = 0
        elif all(param in kwargs for param in ("SNR", "N", "BW")):
            return AnalogFrontend.leapfrog(ENOB=snr_to_enob(kwargs["SNR"]), **kwargs)
        elif all(param in kwargs for param in ("OSR", "N", "T")):
            dt = kwargs.pop("T")
            BW = 1.0 / (2 * kwargs["OSR"] * dt)
            return AnalogFrontend.leapfrog(BW=BW, **kwargs)
        else:
            raise NotImplementedError

        analog_filter = LeapFrog(
            beta * np.ones(N),
            alpha * np.ones(N - 1),
            rho * np.ones(N),
            np.diag([kappa * delta**n for n in range(N)]),
        )
        digital_control = DigitalControl(N, dt)
        return (
            AnalogFrontend(
                analog_filter,
                digital_control,
            ),
            OSR,
        )

    @staticmethod
    def ctsdm(
        ABCD: np.ndarray,
        tdac: np.ndarray,
        dt: float = 1.0,
        quantization_levels: int = 2,
        L: int = 1,
        M: int = 1,
    ) -> "AnalogFrontend":
        """Create a continuous-time sigma-delta modulator

        Parameters
        ----------
        ABCD: `np.ndarray`
            the ABCD matrix of the analog filter
        tdac: `np.ndarray`
            the impulse response of the DAC
        dt: `float`, optional
            the sampling period, defaults to 1.0
        quantization_levels: `int`, optional
            the number of quantization levels, defaults to 2
        L: `int`, optional
            the number of inputs, defaults to 1
        M: `int`, optional
            the number of outputs, defaults to 1

        Returns
        -------
        : :py:class:`cbadc.analog_frontend.AnalogFrontend`
            returns an analog frontend instance
        """
        A, B, C, D = partitionABCD(ABCD, m=L + M, r=M)
        analog_filter = StateSpace(A, B, C, D)
        L = 1
        M = B.shape[1] - L
        digital_control = DigitalControl(
            M, dt=1.0, quantization_levels=quantization_levels
        )
        analog_frontend = AnalogFrontend(analog_filter, digital_control)
        analog_frontend.dt = dt
        return analog_frontend

    @staticmethod
    def dtsdm(
        ABCD: np.ndarray,
        quantization_levels: Optional[np.ndarray] = None,
        L: int = 1,
        M: int = 1,
    ) -> "AnalogFrontend":
        """Create a discrete-time sigma-delta modulator

        Parameters
        ----------
        ABCD: `np.ndarray`
            the ABCD matrix of the analog filter
        quantization_levels: `np.ndarray`, optional
            the number of quantization levels, defaults to 2
        L: `int`, optional
            the number of inputs, defaults to 1
        M: `int`, optional
            the number of outputs, defaults to 1

        Returns
        -------
        : :py:class:`cbadc.analog_frontend.AnalogFrontend`
            returns an analog frontend instance
        """
        if quantization_levels is None:
            quantization_levels = 2 * np.ones((M, 1), dtype=float)
        A, B, C, D = partitionABCD(ABCD, m=L + M, r=M)
        dt = 1.0
        analog_filter = StateSpace(A, B, C, D, dt=dt)
        digital_control = DigitalControl(
            M, dt=dt, quantization_levels=quantization_levels
        )
        return AnalogFrontend(analog_filter, digital_control)

    def avg_power(self, states: np.ndarray):
        """Compute the power consumption

        Parameters
        ----------
        states: `numpy.ndarray`, shape=(size, N, J)
            the state vector

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N,J)
            the power consumption
        """
        return np.inf * np.ones(states.shape[1:], dtype=float)


class GmC(AnalogFrontend):
    """Transconductance-Capacitance Analog Frontend


    Parameters
    ----------
    analog_frontend: : :py:class:`cbadc.analog_frontend.AnalogFrontend`
        the analog frontend
    Cint: : :py:class:`numpy.ndarray`, shape=(N,)
        the larger integration capacitor
    Ro: : :py:class:`numpy.ndarray`, shape=(N,)
        the output resistance, defaults to np.zeros(N)
    Cp: : :py:class:`numpy.ndarray`, shape=(N,)
        the parasitic capacitance, defaults to np.zeros(N)
    v_n: : :py:class:`numpy.ndarray`, shape=(N,)
        per-state (integrator-output-referred) noise density in V/sqrt(Hz),
        one entry per integrator node; sets
        ``state_covariance = diag(v_n**2)``. Defaults to np.zeros(N). For noise
        referred to the converter *input* instead, use
        :func:`cbadc.noise.per_state_intensity` /
        :meth:`AnalogFrontend.input_referred_covariance_matrix` and assign
        ``state_covariance`` directly.
    v_out_min: : :py:class:`numpy.ndarray`, shape=(N,)
        the minimum output voltage, defalts to -np.inf * np.ones(N)
    v_out_max: : :py:class:`numpy.ndarray`, shape=(N,)
        the maximum output voltage, defaults to np.inf * np.ones(N)
    slew_rate: : :py:class:`numpy.ndarray`, shape=(N,)
        the slew rate, defaults to np.inf * np.ones(N)

    """

    def __init__(
        self,
        analog_frontend: AnalogFrontend,
        Cint: np.ndarray,
        Ro: Optional[np.ndarray] = None,
        Cp: Optional[np.ndarray] = None,
        v_n: Optional[np.ndarray] = None,
        v_out_min: Optional[np.ndarray] = None,
        v_out_max: Optional[np.ndarray] = None,
        slew_rate: Optional[np.ndarray] = None,
    ):
        super().__init__(
            _deepcopy(analog_frontend.analog_filter),
            _deepcopy(analog_frontend.digital_control),
            _deepcopy(analog_frontend.analog_signal),
            state_max=v_out_max,
            state_min=v_out_min,
            slew_rate=slew_rate,
        )
        if not isinstance(analog_frontend, AnalogFrontend):
            raise ValueError("analog_frontend must be an instance of AnalogFrontend")
        if analog_frontend.is_discrete_time:
            raise ValueError("Analog frontend must be continuous-time")

        if not isinstance(Cint, np.ndarray):
            raise ValueError("C must be a numpy array")
        if Cint.size != analog_frontend.N:
            raise ValueError(
                "C must have the same number of rows as the analog frontend"
            )
        self._C_int = Cint.flatten()

        if Ro is not None:
            self._Ro = Ro.flatten()
        else:
            # No output resistance
            self._Ro = np.zeros_like(Cint)

        if Cp is not None:
            self._Cp = Cp.flatten()
        else:
            # No parasitic capacitance
            self._Cp = np.zeros_like(self.Cint, dtype=float)

        if v_n is not None:
            self.v_n = v_n.flatten()

        self._compute_ABCD()

    @property
    def dc_gain(self) -> np.ndarray:
        """The DC gain

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N, L + M + N)
            the DC gain vector.
        """
        return self.gm * self.Ro[:, np.newaxis]

    @property
    def Ro(self):
        """The output resistance

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N,)
            the Ro resistance vector.
        """
        return self._Ro

    @Ro.setter
    def Ro(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("R must be a numpy array")
        if value.size != self.N:
            raise ValueError(f"R must have size {self.N}, got {value.size}")
        self._Ro = value.flatten()
        self._compute_ABCD()

    @property
    def Cint(self):
        """The larger integration capacitor

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N,)
            the capacitance matrix.
        """
        return self._C_int

    @Cint.setter
    def Cint(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("Cint must be a numpy array")
        if value.size != self.N:
            raise ValueError(f"Cint must have size {self.N}, got {value.size}")
        self._C_int = value.flatten()
        self._compute_ABCD()

    @property
    def gm(self):
        """Transconductance matrix

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N, L + M + N)
            the transconductance matrix.
        """
        gm = np.zeros((self.N, self.L + self.M + self.N), dtype=float)
        gm[: self.N, : self.N] = self._C_int[:, np.newaxis] * (
            self.A[0] - np.diag(np.diag(self.A[0]))
        )
        gm[: self.N, self.N :] = self._C_int[:, np.newaxis] * self.B[0, 0]
        return gm

    @property
    def Cp(self):
        """The parasitic capacitance

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N,)
            the capacitance matrix.
        """
        return self._Cp

    @Cp.setter
    def Cp(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("Cp must be a numpy array")
        if value.size != self.N:
            raise ValueError(f"Cp must have size {self.N}, got {value.size}")
        self._Cp = value.flatten()
        self._compute_ABCD()

    @property
    def v_n(self):
        """Per-state (integrator-output-referred) noise density in V/sqrt(Hz).

        Setting ``v_n`` assigns ``state_covariance = diag(v_n**2)``; it is the
        noise referred to each integrator's own node, *not* the converter input.

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N,)
            the per-state noise voltage density.
        """
        return self._v_n

    @v_n.setter
    def v_n(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise ValueError("v_n must be a numpy array")
        if value.size != self.N:
            raise ValueError(f"v_n must have size {self.N}, got {value.size}")
        self._v_n = value.flatten()
        self.state_covariance = np.diag(self._v_n**2).astype(float)

    def _compute_ABCD(self):
        one_over_RC = 1.0 / (self._Ro * (self._C_int + self._Cp))
        self.A -= np.diag(one_over_RC)

    def avg_power(self, states: np.ndarray):
        """Compute the power consumption

        Parameters
        ----------
        states: `numpy.ndarray`, shape=(size, N, J)
            the state vector

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N,J)
            the power consumption
        """
        return np.mean(states**2 / self.Ro.reshape((1, -1, 1)), axis=0)


class ActiveRC(AnalogFrontend):
    """Active RC Analog Frontend

    Parameters
    ----------
    analog_frontend: : :py:class:`cbadc.analog_frontend.AnalogFrontend`
        the analog frontend
    Cint: : :py:class:`numpy.ndarray`, shape=(N,)
        the integration capacitance.
    gm: : :py:class:`numpy.ndarray`, shape=(N,)
        the transconductance.
    Ro: : :py:class:`numpy.ndarray`, shape=(N,)
        the internal state resistance.
    Co: : :py:class:`numpy.ndarray`, shape=(N,)
        the internal state capacitance.

    """

    def __init__(
        self,
        analog_frontend: AnalogFrontend,
        Cint: np.ndarray,
        gm: np.ndarray,
        Ro: np.ndarray,
        Co: np.ndarray,
    ):
        if not isinstance(analog_frontend, AnalogFrontend):
            raise ValueError("analog_frontend must be an instance of AnalogFrontend")
        if analog_frontend.is_discrete_time:
            raise ValueError("Analog frontend must be continuous-time")

        N_2 = analog_frontend.N
        N = 2 * analog_frontend.N
        A = np.zeros((N, N))
        B = np.zeros((N, analog_frontend.L + analog_frontend.M))
        C = np.zeros((analog_frontend.M, N))
        D = analog_frontend.D[0]

        if not isinstance(Ro, np.ndarray):
            raise ValueError("Ro must be a numpy array")
        if Ro.size != analog_frontend.N:
            raise ValueError(
                "Ro must have the same number of rows as the analog frontend"
            )
        if not isinstance(Co, np.ndarray):
            raise ValueError("Co must be a numpy array")
        if Co.size != analog_frontend.N:
            raise ValueError(
                "Co must have the same number of rows as the analog frontend"
            )
        if not isinstance(Cint, np.ndarray):
            raise ValueError("Cint must be a numpy array")
        if Cint.size != analog_frontend.N:
            raise ValueError(
                "Cint must have the same number of rows as the analog frontend"
            )

        self._Ro = Ro.flatten()
        self._Co = Co.flatten()
        self._Cint = Cint.flatten()
        self._gm = gm.flatten()

        #
        A[N_2:, N_2:] = -np.diag(1.0 / (self._Ro * self._Co))
        A[N_2:, :N_2] = -np.diag(self._gm / self._Co)

        A[:N_2, N_2:] = -analog_frontend.A[0]
        A[:N_2, :N_2] -= np.diag(np.sum(np.abs(analog_frontend.A[0]), axis=1))
        # + dV_int /dt
        A[:N_2, :] += A[N_2:, :]

        B[:N_2, :] = analog_frontend.B[0]
        C[:, N_2:] = -analog_frontend.C[0]
        # print(A, B, C, D)
        analog_filter = StateSpace(A, B, C, D)

        super().__init__(
            analog_filter,
            _deepcopy(analog_frontend.digital_control),
            _deepcopy(analog_frontend.analog_signal),
            _deepcopy(analog_frontend.state_covariance),
            _deepcopy(analog_frontend.output_covariance),
            # _deepcopy(analog_frontend.slew_rate),
        )

        self._state_labels = [f"x{i}" for i in range(analog_frontend.N)] + [
            f"x_int{i}" for i in range(analog_frontend.N)
        ]

    @property
    def Ro(self):
        """The internal op-amp resistance

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N/2,)
            the resistance matrix.
        """
        return self._Ro

    @property
    def Co(self):
        """The internal op-amp capacitance

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N/2,)
            the capacitance matrix.
        """
        return self._Co

    @property
    def Cint(self):
        """The internal integrator capacitance

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N/2,)
            the capacitance matrix.
        """
        return self._Cint

    @property
    def GBWP(self):
        """The gain-bandwidth product

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N/2,)
            the gain-bandwidth product
        """
        N_2 = self.N // 2
        return np.diag(self.A[N_2:, :N_2]) / (2 * np.pi)

    @property
    def DC_gain(self):
        """The DC gain

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N/2,)
            the DC gain
        """
        N_2 = self.N // 2
        return -np.diag(self.A[N_2:, N_2:]) / (2 * np.pi)

    @property
    def R_int(self):
        """The integration resistance

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N/2,N/2)
            the integration resistance
        """
        return 1.0 / (self._Cint * (self.A - np.diag(np.diag(self.A))))

    def avg_power(self, states: np.ndarray):
        """Compute the power consumption

        Parameters
        ----------
        states: `numpy.ndarray`, shape=(size, N)
            the state vector

        Returns
        -------
        : `float`
            the power consumption
        """
        return np.inf * np.ones_like(states, dtype=float)
