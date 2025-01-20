"""The analog fronted module."""

from typing import Optional, Union
from .digital_control import DigitalControl
from .fom import snr_from_dB, enob_to_snr, snr_to_enob
from .analog_signal import AnalogSignal, Sinusoidal, ConcatenatedSignals
from .analog_filter import ChainOfIntegrators, LeapFrog
from .analog_filter.analog_system import AnalogSystem, as2af
from .delsig import simulateDSM, partitionABCD
from scipy.signal import StateSpace, freqresp
from scipy.linalg import block_diag

import numpy as np
import sympy as sp
import logging
import scipy.linalg as _linalg
import scipy.integrate as _integrate
from copy import deepcopy as _deepcopy

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


class AnalogFrontend:
    """An analog frontend

    As a cornerstone of the control-bounded conversion theory,
    an analog frontend is a analog-to-digital conversion system
    which operates by controlling an analog filter using digital
    control.

    In particular, an analog filter is specified by a
    [[A B], [C D]] state-space representation, where
    - A is the state matrix of shape ((N, N))
    - B is the input matrix of shape ((N, L + M)), where L is the number of analog signals and M is the number of control signals.
    Furthermore, B is partitioned into B = [B_L B_M],
    where B_0 is the corresponding analog signal input matrix of shape ((N, L))
    and B_1 is the digital control matrix of shape ((N, M)).
    - C is the output matrix (control input matrix) of shape ((M, N)), and
    - D is the feedthrough matrix of shape ((M, L + M)) where
    D is partitioned into D = [D_L D_M], where
    D_L is the feedthrough matrix for the analog signals and
    D_M is the feedthrough matrix for the control signals.


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

        # Check dimensions
        L = self.analog_signal.L
        M = self.digital_control.M
        N = self.analog_filter.A.shape[0]
        self.A = self.analog_filter.A
        self.B = self.analog_filter.B
        self.C = self.analog_filter.C
        self.D = self.analog_filter.D
        if self.B.shape != (N, L + M):
            raise ValueError(
                f"analog_filter.B must have shape {(N, L + M)}, got {self.B.shape}"
            )
        if self.C.shape != (M, N):
            raise ValueError(
                f"analog_filter.C must have shape {(M, N)}, got {self.C.shape}"
            )
        if self.D.shape != (M, L + M):
            raise ValueError(
                f"analog_filter.D must have shape {(M, L + M)}, got {self.D.shape}"
            )
        self.state_covariance = state_covariance
        self.output_covariance = output_covariance
        self.rng = np.random.default_rng(seed)

        if slew_rate is None:
            # V/s
            self.slew_rate = np.inf * np.ones((N))
        elif isinstance(slew_rate, np.ndarray):
            if slew_rate.size != N:
                raise ValueError(f"slew_rate must have size {N}, got {slew_rate.size}")
            self.slew_rate = slew_rate.flatten()
        else:
            raise ValueError(f"slew_rate must be None or a {N}-sized numpy array")

        if state_max is None:
            self.state_max = np.inf * np.ones((N))
        elif isinstance(state_max, np.ndarray):
            if state_max.size != N:
                raise ValueError(f"state_max must have size {N}, got {state_max.size}")
            self.state_max = state_max.flatten()
        else:
            raise ValueError(f"state_max must be None or a {N}-sized numpy array")

        if state_min is None:
            self.state_min = -np.inf * np.ones((N))
        elif isinstance(state_min, np.ndarray):
            if state_min.size != N:
                raise ValueError(f"state_min must have size {N}, got {state_min.size}")
            self.state_min = state_min.flatten()
        else:
            raise ValueError(f"state_min must be None or a {N}-sized numpy array")

    def __str__(self) -> str:
        return (
            f"AnalogFrontend(\n"
            "  analog_filter=StateSpace(\n"
            f"      A={self.A},\n"
            f"      B={self.B},"
            f"      C={self.C},\n"
            f"      D={self.D}),\n"
            "   )"
            f"  digital_control={self.digital_control},\n"
            f"  analog_signal={self.analog_signal},\n"
            f"  state_covariance={self.state_covariance},\n"
            f"  output_covariance={self.output_covariance},\n"
            f"  N={self.N},\n"
            f"  L={self.L},\n"
            f"  M={self.M},\n"
            f"  dt={self.dt}\n"
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
        if A.shape != (self.N, self.N):
            raise ValueError(f"A must have shape {(self.N, self.N)}, got {A.shape}")
        self._A: np.ndarray = np.asarray(A)
        self.analog_filter.A = np.asarray(A)

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
        if B.shape != (self.N, self.L + self.M):
            raise ValueError(
                f"B must have shape {(self.N, self.L + self.M)}, got {B.shape}"
            )
        self._B: np.ndarray = np.asarray(B)
        self.analog_filter.B = np.asarray(B)

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
        if C.shape != (self.M, self.N):
            raise ValueError(f"C must have shape {(self.M, self.N)}, got {C.shape}")
        self._C = np.asarray(C)
        self.analog_filter.C = np.asarray(C)

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
        if D.shape != (self.M, self.L + self.M):
            raise ValueError(
                f"D must have shape {(self.M, self.L + self.M)}, got {D.shape}"
            )
        self._D = np.asarray(D)
        self.analog_filter.D = np.asarray(D)

    @property
    def ABCD(self):
        """The ABCD matrix

        Returns
        -------
        : numpy.ndarray, shape=(M + N, L + M + N)
            the analog filter's ABCD matrix
        """
        return np.vstack((np.hstack((self.A, self.B)), np.hstack((self.C, self.D))))

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
            signal = AnalogSignal()
        if not isinstance(signal, AnalogSignal):
            raise ValueError("analog_signal must be a list of AnalogSignal")
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
        return self.analog_filter.A.shape[0]

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
            # discretize the state covariance matrix
            if self.is_discrete_time:
                state_cov_d = state_covariance
            else:
                tmp = np.vstack(
                    (
                        np.hstack((-self.A, state_covariance)),
                        np.hstack((np.zeros((self.N, self.N), dtype=float), self.A.T)),
                    )
                )
                tmp = _linalg.expm(tmp * self.dt)
                state_cov_d = tmp[self.N :, self.N :].T @ tmp[: self.N, self.N :]
            self._state_covariance = state_covariance
            self._state_cov_cholesky = np.linalg.cholesky(state_cov_d)

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
        B = analog_frontend.B[:, : analog_frontend.L]
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
            self._output_cov_cholesky = np.linalg.cholesky(output_covariance)

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
            - 's': the digital control signals, shape (size, M, J)
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
            "s": inputs[:, self.L :, :],
            "x": states,
            "y": outputs,
        }

    # Simulation methods
    def simulate(
        self,
        size: int,
        x: Optional[np.ndarray] = None,
        t0: float = 0.0,
        atol=1e-12,
        rtol=1e-7,
        sub_samples: int = 1,
        method: Optional[str] = None,
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
        sub_samples: `int`, optional
            the number of sub-samples per digital control period, defaults to 1.
            Note that in case a discrete-time filter with different dt compared
            to the dt of the digital control is used, the sub_samples parameter
            is automatically adjusted to match the largest of sub_samples and
            the ratio of the two sampling periods
        state_covariance: `np.ndarray`, optional
            the state covariance matrix, defaults to None.
        output_covariance: `np.ndarray`, optional
            the output covariance matrix, defaults to None.
        method: `str`, optional
            the simulation method, defaults to None. If None, the method is
            automatically selected based on the analog frontend's properties.
            Valid options are:

            * 'dsim': simulate the discrete-time analog frontend

            * 'sin': simulate the continuous-time analog frontend for sinusoidal input

            * 'ode': simulate the continuous-time analog frontend using an ODE solver


        Returns
        -------
        : dict[str, np.ndarray]
            a dictionary containing the simulation results where
            - 't': the time vector, shape (size,)
            - 'u': the analog signal evaluated at t, shape (size, L, J)
            - 's': the digital control signals, shape (size, M, J)
            - 'x': the state vector, shape (size, N, J)
            - 'y': the quantization input vector, shape (size, M, J)

        """

        # Allocate memory
        inputs = np.zeros((size, self.L + self.M, self.J), dtype=float)
        # add state noise
        if self.state_covariance is not None:
            states = self._state_cov_cholesky @ self.rng.normal(
                size=(size, self.N, self.J)
            )
        else:
            states = np.zeros((size, self.N, self.J), dtype=float)
        # add output noise
        if self.output_covariance is not None:
            outputs = self._output_cov_cholesky @ self.rng.normal(
                size=(size, self.M, self.J)
            )
        else:
            outputs = np.zeros((size, self.M, self.J), dtype=float)

        # initialize the states
        if x is None:
            x = np.zeros((self.N, self.J))
        states[0, :, :] = np.clip(
            x, self.state_min[:, np.newaxis], self.state_max[:, np.newaxis]
        )

        if sub_samples < 1:
            raise ValueError("sub_samples must be greater than 0")
        t = np.arange(size) * self.digital_control.dt / sub_samples + t0

        if method is None:
            if self.is_discrete_time:
                method = "dsim"
            elif isinstance(self.analog_signal, Sinusoidal):
                method = "sin"
            else:
                method = "ode"

        # Discrete-time analog filter
        if method == "dsim":
            logging.info("Simulating discrete-time analog frontend")
            if not np.isclose(self.digital_control.dt % self.analog_filter.dt, 0):
                raise ValueError(
                    "Digital control sampling period must be a multiple of the analog filter sampling period"
                )

            sub_samples = np.maximum(
                sub_samples, int(self.digital_control.dt / self.analog_filter.dt)
            )
            t = np.arange(size) * self.digital_control.dt / sub_samples + t0

            # pre compute input signal contributions

            inputs[:, : self.L, :] = self.analog_signal.evaluate(t)

            # populate first output and control
            outputs[0] += self.C @ states[0] + self.D @ inputs[0]
            inputs[0, self.L :, :] = self.digital_control.quantize(outputs[0])

            slew_rate_dt = self.slew_rate * self.digital_control.dt

            # Compute simulation recursions
            for i in range(size - 1):
                # state evolution
                states[i + 1] += np.clip(
                    self.A @ states[i] + self.B @ inputs[i],
                    -slew_rate_dt[:, np.newaxis],
                    slew_rate_dt[:, np.newaxis],
                )
                states[i + 1] = np.clip(
                    states[i + 1],
                    self.state_min[:, np.newaxis],
                    self.state_max[:, np.newaxis],
                )
                # output computation
                outputs[i + 1] += self.C @ states[i + 1] + self.D @ inputs[i + 1]
                # control update
                if i % sub_samples == 0:
                    inputs[i + 1, self.L :, :] = self.digital_control.quantize(
                        outputs[i + 1]
                    ).reshape(self.M, self.J)
                else:
                    inputs[i + 1, self.L :, :] = inputs[i, self.L :, :].reshape(
                        self.M, self.J
                    )
        # Sinusoidal special case
        elif method == "sin":
            logging.info(
                "Simulating continuous-time analog frontend for sinusoidal input"
            )
            if self.analog_signal.piecewise_constant:
                logging.info("Piecewise constant input signal")
                logging.info("Discretizing analog frontend before simulation.")
                ds = self.discretize(self.dt, atol=atol, rtol=rtol)
                # simulate the discretized system
                return ds.simulate(size, x, t0, atol, rtol)

            # exp(A * dt) computation
            A_d = _linalg.expm(self.A * self.dt)
            C_d = self.C[:]
            D_d = self.D[:]
            D_d[:, self.L :] *= self.digital_control.evaluate(
                self.dt, np.ones((self.M, 1))
            ).reshape((1, self.M))
            B_d = np.zeros((self.N, self.L + self.M, self.J), dtype=float)

            tmp_sig_vec = np.zeros((1, self.L + self.M, self.J), dtype=float)
            tmp_sig_vec[0, : self.L, :] = self.analog_signal.offset

            def derivative(t: float, x: np.ndarray) -> np.ndarray:
                tmp_sig_vec[0, self.L :, :] = self.digital_control.impulse_response(
                    np.array([t])
                ).reshape((self.M, 1))
                return (
                    np.tensordot(
                        self.A,
                        x.reshape((self.N, self.L + self.M, self.J)),
                        axes=[[1], [0]],
                    )
                    + self.B[:, :, np.newaxis] * tmp_sig_vec
                ).flatten()

            # Compute the control contributions
            res = _integrate.solve_ivp(
                derivative,
                (0.0, self.dt),
                np.zeros((self.N, self.L + self.M, self.J), dtype=float).flatten(),
                atol=atol,
                rtol=rtol,
                method="DOP853",
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

            # populate first output and control
            inputs[:, : self.L, :] = self.analog_signal.evaluate(t)
            outputs[0] += self.C @ states[0] + self.D @ inputs[0]
            inputs[0, self.L :, :] = self.digital_control.quantize(outputs[0])

            slew_rate_dt = self.slew_rate * self.digital_control.dt

            # Compute simulation recursions
            for i in range(size - 1):
                # state evolution
                states[i + 1] += np.clip(
                    A_d @ states[i]
                    # control feedback
                    + B_d[:, self.L :, 0] @ inputs[i, self.L :, :]
                    # pre-computed input signal contributions
                    + pre_computed_input_offset + pre_computed_inputs[i],
                    -slew_rate_dt[:, np.newaxis],
                    slew_rate_dt[:, np.newaxis],
                )
                states[i + 1] = np.clip(
                    states[i + 1],
                    self.state_min[:, np.newaxis],
                    self.state_max[:, np.newaxis],
                )
                # output computation
                outputs[i + 1] += C_d @ states[i + 1] + D_d @ inputs[i + 1]
                # control update
                if i % sub_samples == 0:
                    inputs[i + 1, self.L :, :] = self.digital_control.quantize(
                        outputs[i + 1]
                    )
                else:
                    inputs[i + 1, self.L :, :] = inputs[i, self.L :, :]
        # Full ode solver
        elif method == "ode":
            logging.info("Simulating continuous-time analog frontend")
            t = np.arange(size) * self.digital_control.dt + t0
            # Make sure the input signal is not piecewise constant,
            # otherwise the simulation can be better performed
            # by discretizing the analog frontend before simulation.
            if self.analog_signal.piecewise_constant:
                logging.warning(
                    """"Piecewise constant input signal, discretizing
                                analog frontend before simulation may be beneficial."""
                )

            inputs[:, : self.L, :] = self.analog_signal.evaluate(t)
            # populate first output and control
            outputs[0] += self.C @ states[0] + self.D @ inputs[0]
            inputs[0, self.L :, :] = self.digital_control.quantize(outputs[0])

            def derivative(t: float, x: np.ndarray, *args) -> np.ndarray:

                return np.clip(
                    # state evolution
                    self.A @ x.reshape(self.N, self.J)
                    # input signal contributions
                    + self.B[:, : self.L]
                    @ self.analog_signal.evaluate(np.array([t]))[0, :, :]
                    # control feedback
                    + self.B[:, self.L :]
                    # args[0] is the current time and args[1] the quantizer input
                    @ self.digital_control.evaluate(
                        t - args[0], args[1][:, np.newaxis, :]
                    ),
                    -self._slew_rate[:, np.newaxis],
                    self._slew_rate[:, np.newaxis],
                ).flatten()

            for i in range(size - 1):

                res = _integrate.solve_ivp(
                    derivative,
                    (t[i], t[i + 1]),
                    states[i].flatten(),
                    args=(t[i], outputs[i]),
                    atol=atol,
                    rtol=rtol,
                    # method="DOP853",
                )
                states[i + 1] += res.y[:, -1].reshape((self.N, self.J))
                states[i + 1] = np.clip(
                    states[i + 1],
                    self.state_min[:, np.newaxis],
                    self.state_max[:, np.newaxis],
                )
                # output computation
                outputs[i + 1] += self.C @ states[i + 1] + self.D @ inputs[i + 1]
                # control update
                if i % sub_samples == 0:
                    inputs[i + 1, self.L :, :] = self.digital_control.quantize(
                        outputs[i + 1]
                    ).reshape((self.M, self.J))
                else:
                    inputs[i + 1, self.L :, :] = inputs[i, self.L :, :]
        else:
            raise ValueError(f"Unknown simulation method {method}")

        return {
            "t": t,
            "u": inputs[:, : self.L, :],
            "s": inputs[:, self.L :, :],
            "x": states,
            "y": outputs,
        }

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

        C_d = self.C[:]
        D_d = self.D[:]
        # If return to zero DAC there is no direct path at the end of the time period.

        # assuming piecewise constant input signal
        if not self.analog_signal.piecewise_constant:
            logger.warning(
                "Non piecewise constant input signal. The discretization may not be accurate."
            )

        if (
            self.analog_signal.piecewise_constant
            and self.digital_control.dac_waveform == "nrz"
        ):
            tmp_arg = np.vstack(
                (
                    np.hstack((self.A, self.B)),
                    np.zeros((self.L + self.M, self.N + self.L + self.M), dtype=float),
                )
            )
            tmp = _linalg.expm(tmp_arg * dt)
            A_d = tmp[: self.N, : self.N]
            B_d = tmp[: self.N, self.N :]
        else:
            A_d = _linalg.expm(self.A * dt)

            delay_steps = np.max(self.digital_control.delay_steps())
            additional_states: int = delay_steps * self.M

            tmp_sig_vec = np.zeros(
                (1, self.L + self.M + additional_states), dtype=float
            )
            B_temp = np.zeros(
                (self.N, self.L + self.M + additional_states), dtype=float
            )
            B_temp[:, : self.L + self.M] = self.B[:, : self.L + self.M]
            for k in range(1, delay_steps):
                B_temp[
                    :,
                    self.L + k * self.M : self.L + (k + 1) * self.M,
                ] = self.B[:, self.L :]

            def derivative(t: float, x: np.ndarray) -> np.ndarray:
                t_array = np.array([t])
                tmp_sig_vec[0, : self.L] = self.analog_signal.impulse_response(t_array)[
                    :, :, 0
                ]
                tmp_sig_vec[0, self.L :] = self.digital_control.impulse_response(
                    t_array + self.digital_control.dt * np.arange(delay_steps + 1)
                ).T.flatten()

                return (
                    self.A @ x.reshape((self.N, -1)) + B_temp * tmp_sig_vec
                ).flatten()

            # Compute input signal contributions
            res = _integrate.solve_ivp(
                derivative,
                (0.0, dt),
                np.zeros(self.N * (self.L + self.M + additional_states), dtype=float),
                atol=atol,
                rtol=rtol,
                method="DOP853",
            )
            # shape(N, L + M + additional_states)
            x_vals = res.y[:, -1].reshape((self.N, -1))
            # Bd = [[ Bu, Bc ], [0, I]]
            B_d = np.zeros((self.N + additional_states, self.L + self.M))
            B_d[: self.N, : self.L + self.M] = x_vals[:, : self.L + self.M]
            if additional_states > 0:
                # A_d matrix structure:
                # [[A, B2, B3, ...],
                #  [0, 0, ...],
                #  [0, I, 0, ...],
                #  [0, 0, I, ...]]
                A_d = np.vstack(
                    (
                        np.hstack(
                            (
                                A_d,
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
                        self.N + k * self.M : self.N + (k + 1) * self.M,
                        self.N + (k - 1) * self.M : self.N + k * self.M,
                    ] = np.eye(self.M, dtype=float)
                B_d[self.N : self.N + self.M, self.L :] = np.eye(self.M, dtype=float)
                A_d[: self.N, self.N :] = res.y[
                    self.N * (self.L + self.M) :, -1
                ].reshape((self.N, additional_states))
                # C_d = [C, 0, 0, ...]
                C_d = np.hstack(
                    (C_d, np.zeros((self.M, additional_states), dtype=float))
                )

        analog_filter = StateSpace(A_d, B_d, C_d, D_d, dt=dt)
        digital_control = _deepcopy(self.digital_control)
        analog_signal = _deepcopy(self.analog_signal)
        return AnalogFrontend(analog_filter, digital_control, analog_signal)

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
                self.A,
                self.B,
                self.C,
                self.D,
            )
        else:
            # Feedback transfer function, i.e.,
            Bl = self.B[:, : self.L]
            Bm = self.B[:, self.L :]
            Dl = self.D[:, : self.L]
            Dm = self.D[:, self.L :]
            I_DM = np.linalg.inv(np.eye(self.M) - Dm)
            A_new = self.A + Bm @ I_DM @ self.C
            B_new = Bl + I_DM @ Dl
            C_new = I_DM @ self.C
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
        Aq = _linalg.block_diag(self.A, self.A)
        Aq[: self.N, self.N :] = -wp * np.eye(self.N)
        Aq[self.N :, : self.N] = wp * np.eye(self.N)
        # Bq = [[B, 0], [0, B]]
        Bq = _linalg.block_diag(self.B, self.B)
        # Cq = [[C, 0], [0, C]]
        Cq = _linalg.block_diag(self.C, self.C)
        # Dq = [[D, 0], [0, D]]
        Dq = _linalg.block_diag(self.D, self.D)

        analog_filter = StateSpace(Aq, Bq, Cq, Dq)
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

    def simulateSNR(
        self,
        OSR: int,
        amp_dB: Optional[np.ndarray] = None,
        f0: float = 0.0,
        f: Optional[float] = None,
        k: int = 13,
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

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(J,)
            the signal-to-noise ratio
        : :py:class:`numpy.ndarray`, shape=(J,)
            the corresponding amplitude of the input signal
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

        if f is None:
            f = f0 + 0.5 / (OSR * 2)  # Halfway across the band
        if np.abs(f - f0) > 0.5 / OSR:
            logger.warning("The input tone is out-of-band.")
        fft_bins = 1 << k
        if fft_bins < (OSR << 4):
            logger.warning(
                "The FFT size is too small.",
                "Increasing k to accommodate a large oversampling ratio.",
            )
            k = int(np.ceil(np.log2(OSR << 4)))
            fft_bins = 1 << k
        f_int = int(np.round(f * fft_bins))
        if np.abs(f_int) < 2:
            logger.warning(
                "The input tone is too close to DC.",
                "Increasing k to accommodate a low input frequency.",
            )
            k = np.ceil(np.log2(1.0 / np.abs(f)))
            fft_bins = 1 << k
            f_int = 2
        warm_up = 1 << 7
        window = np.hanning(fft_bins)
        if f0 == 0.0:
            in_band_bins = fft_bins // 2 + np.arange(
                3, np.round(fft_bins / (2 * OSR)) + 1, dtype=int
            )
            f_int -= 2

        # before overwriting the analog signal, store the old one
        _old_analog_signal = self.analog_signal

        self.analog_signal = Sinusoidal(
            amp_lin, f_int / fft_bins * self.fs * np.ones_like(amp_dB)
        )
        sim = self.simulate(fft_bins + warm_up)

        avg_pow = np.sum(np.mean(self.avg_power(sim["x"]), axis=1))

        # OSR = 1 / (2 * dt * BW)
        wf = self.wiener_filter(OSR=OSR)
        # shape = (fft_bins+warm_up, J)
        u_hat = wf.evaluate(sim["s"])[:, 0, :]
        # shape = (fft_bins, J)
        hwfft = np.fft.fftshift(
            np.fft.fft(u_hat[warm_up:] * window[:, np.newaxis], axis=0), axes=0
        )
        # shape = (J,)
        snr = self.calculateSNR_from_fft(hwfft[in_band_bins - 1])

        # reset the analog signal
        self.analog_signal = _old_analog_signal

        return snr, amp_dB.flatten(), avg_pow

    def calculateSNR_from_fft(self, fft: np.ndarray, extra_bins: int = 2):
        """Calculate the SNR from an FFT

        Parameters
        ----------
        fft: `np.ndarray`, shape=(size, ...)
            the FFT
        f: `float`
            the frequency of the input tone
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
            dt = 1.0 / (2 * BW * OSR)
            beta = 1 / (2 * dt)
            kappa = beta
            rho = kwargs.get("rho", 0.0)
            analog_filter = ChainOfIntegrators(
                beta * np.ones(N),
                rho * np.ones(N),
                kappa * np.eye(N),
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
        the input referred noise density in V rms, defaults to np.zeros(N)
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
        gm[: self.N, : self.N] = self._C_int * (self.A - np.diag(np.diag(self.A)))
        gm[: self.N, self.N :] = self._C_int * self.B
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
        """The input referred noise density in V

        Returns
        -------
        : :py:class:`numpy.ndarray`, shape=(N,)
            the noise voltage matrix.
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
        D = analog_frontend.D

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

        A[:N_2, N_2:] = -analog_frontend.A
        A[:N_2, :N_2] -= np.diag(np.sum(np.abs(analog_frontend.A), axis=1))
        # + dV_int /dt
        A[:N_2, :] += A[N_2:, :]

        B[:N_2, :] = analog_frontend.B
        C[:, N_2:] = -analog_frontend.C

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
        return np.mean(states**2 / self.Ro.reshape((1, -1)), axis=0)
