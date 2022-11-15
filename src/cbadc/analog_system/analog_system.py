"""Analog systems

This module provides a general :py:class:`cbadc.analog_system.AnalogSystem`
class with the necessary functionality to do transient simulations, compute
transfer functions, and exposing the relevant system parameters as
attributes. Additionally, several derived convenience classes are defined
to quickly initialize analog systems of particular structures.
"""
import numpy as np
import scipy.signal
import logging
from typing import Union
import sympy as sp

logger = logging.getLogger(__name__)


class AnalogSystem:
    """Represents an analog system.

    The AnalogSystem class represents an analog system goverened by the
    differential equations,

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    where we refer to :math:`\mathbf{A} \in \mathbb{R}^{N \\times N}` as the
    system matrix, :math:`\mathbf{B} \in \mathbb{R}^{N \\times L}` as the
    input matrix, and :math:`\mathbf{\Gamma} \in \mathbb{R}^{N \\times M}` is
    the control input matrix. Furthermore,
    :math:`\mathbf{x}(t)\in\mathbb{R}^{N}` is the state vector of the system,
    :math:`\mathbf{u}(t)\in\mathbb{R}^{L}` is the vector valued,
    continuous-time, analog input signal, and
    :math:`\mathbf{s}(t)\in\mathbb{R}^{M}` is the vector valued control signal.

    The analog system also has two (possibly vector valued) outputs namely:

    * The control observation
      :math:`\\tilde{\mathbf{s}}(t)=\\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t) + \\tilde{\mathbf{D}} \mathbf{u}(t)` and
    * The signal observation :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{D} \mathbf{u}(t)`

    where
    :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}\in\mathbb{R}^{\\tilde{M} \\times N}`
    is the control observation matrix,
    :math:`\\tilde{\mathbf{D}}\in\mathbb{R}^{\\tilde{M} \\times L}`
    is the direct control observation matrix, and
    :math:`\mathbf{C}^\mathsf{T}\in\mathbb{R}^{\\tilde{N} \\times N}` is the
    signal observation matrix.

    Parameters
    ----------
    A : `array_like`, shape=(N, N)
        system matrix.
    B : `array_like`, shape=(N, L)
        input matrix.
    CT : `array_like`, shape=(N_tilde, N)
        signal observation matrix.
    Gamma : `array_like`, shape=(N, M)
        control input matrix.
    Gamma_tildeT : `array_like`, shape=(M_tilde, N)
        control observation matrix.
    D : `array_like`, shape=(N_tilde, L), optional
        the direct matrix, defaults to None
    D_tilde : `array_like`, shape=(M_tilde, L), optional
        the direct control observation matrix, defaults to None
    A_tilde : `array_like`, shape=(M_tilde, N), optional
        the self control observation matrix, defaults to None

    Attributes
    ----------
    N : `int`
        state space order :math:`N`.
    N_tilde : `int`
        number of signal observations :math:`\\tilde{N}`.
    M : `int`
        number of digital control signals :math:`M`.
    M_tilde : `int`
        number of control signal observations :math:`\\tilde{M}`.
    L : `int`
        number of input signals :math:`L`.
    A : `array_like`, shape=(N, N)
        system matrix :math:`\mathbf{A}`.
    B : `array_like`, shape=(N, L)
        input matrix :math:`\mathbf{B}`.
    CT : `array_like`, shape=(N_tilde, N)
        signal observation matrix :math:`\mathbf{C}^\mathsf{T}`.
    Gamma : `array_like`, shape=(N, M)
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : `array_like`, shape=(M_tilde, N)
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}`.
    t: :py:class:`sympy.Symbol`
        the symbolic time variable.
    x: [:py:class:`sympy.Function`]
        a list containing the state variable functions.

    See also
    --------
    :py:class:`cbadc.simulator.StateSpaceSimulator`

    Example
    -------
    >>> import numpy as np
    >>> from cbadc.analog_system import AnalogSystem
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[1], [2]])
    >>> CT = np.array([[1, 2], [0, 1]]).transpose()
    >>> Gamma = np.array([[-1, 0], [0, -5]])
    >>> Gamma_tildeT = CT.transpose()
    >>> system = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)

    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        CT: np.ndarray,
        Gamma: Union[np.ndarray, None],
        Gamma_tildeT: Union[np.ndarray, None],
        D: Union[np.ndarray, None] = None,
        D_tilde: Union[np.ndarray, None] = None,
        A_tilde: Union[np.ndarray, None] = None,
    ):
        """Create an analog system.

        Parameters
        ----------
        A : `array_like`, shape=(N, N)
            system matrix.
        B : `array_like`, shape=(N, L)
            input matrix.
        CT : `array_like`, shape=(N_tilde, N)
            signal observation matrix.
        Gamma : `array_like`, shape=(N, M)
            control input matrix.
        Gamma_tildeT : `array_like`, shape=(M_tilde, N)
            control observation matrix.
        D : `array_like`, shape=(N_tilde, L), optional
            the direct matrix, defaults to None
        D_tilde : `array_like`, shape=(M_tilde, L), optional
            the direct control observation matrix, defaults to None
        """

        self.A = np.array(A, dtype=np.double)
        self._A_s = sp.Matrix(A)
        self.B = np.array(B, dtype=np.double)
        self._B_s = sp.Matrix(B)
        self.CT = np.array(CT, dtype=np.double)
        self._CT_s = sp.Matrix(CT)
        if Gamma is not None:
            self.Gamma = np.array(Gamma, dtype=np.double)
            self._Gamma_s = sp.Matrix(Gamma)
            if self.Gamma.shape[0] != self.A.shape[0]:
                raise InvalidAnalogSystemError(
                    self, "N does not agree with control input matrix Gamma."
                )
            self.M: int = self.Gamma.shape[1]
        else:
            self.Gamma = None
            self._Gamma_s = None
            self.M: int = 0

        if Gamma_tildeT is not None:
            self.Gamma_tildeT = np.array(Gamma_tildeT, dtype=np.double)
            self._Gamma_tildeT_s = sp.Matrix(Gamma_tildeT)
            if self.Gamma_tildeT.shape[1] != self.A.shape[0]:
                raise InvalidAnalogSystemError(
                    self,
                    """N does not agree with control observation matrix
                    Gamma_tilde.""",
                )
            self.M_tilde: int = self.Gamma_tildeT.shape[0]
        else:
            self.Gamma_tildeT = None
            self._Gamma_tildeT_s = None
            self.M_tilde: int = 0

        self.N: int = self.A.shape[0]

        if self.A.shape[0] != self.A.shape[1]:
            raise InvalidAnalogSystemError(self, "system matrix not square")

        # ensure matrices
        if len(self.B.shape) == 1:
            self.B = self.B.reshape((self.N, 1))
        if len(self.CT.shape) == 1:
            self.CT = self.CT.reshape((1, self.N))

        self.L: int = self.B.shape[1]
        self.N_tilde: int = self.CT.shape[0]

        self.temp_derivative = np.zeros(self.N, dtype=np.double)
        self.temp_y = np.zeros(self.N_tilde, dtype=np.double)
        self.temp_s_tilde = np.zeros(self.M_tilde, dtype=np.double)

        if self.B.shape[0] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with input matrix B."
            )

        if self.CT.shape[1] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with signal observation matrix C."
            )

        if D is not None:
            self.D = np.array(D, dtype=np.double)
        else:
            self.D = np.zeros((self.N_tilde, self.L))
        self._D_s = sp.Matrix(self.D)

        if self.D is not None and (
            self.D.shape[0] != self.N_tilde or self.D.shape[1] != self.L
        ):
            raise InvalidAnalogSystemError(
                self, "D matrix has wrong dimensions. Should be N_tilde x L"
            )
        if D_tilde is not None:
            self.D_tilde = np.array(D_tilde, dtype=np.double)
        else:
            self.D_tilde = np.zeros((self.M_tilde, self.L))
        self._D_tilde_s = sp.Matrix(self.D_tilde)

        if self.D is not None and (
            self.D.shape[0] != self.N_tilde or self.D.shape[1] != self.L
        ):
            raise InvalidAnalogSystemError(
                self, "D matrix has wrong dimensions. Should be N_tilde x L"
            )

        if self.D_tilde is not None and (
            self.D_tilde.shape[0] != self.M_tilde or self.D_tilde.shape[1] != self.L
        ):
            raise InvalidAnalogSystemError(
                self, "D_tilde matrix has wrong dimensions. Should be M_tilde x L"
            )

        if A_tilde is not None:
            self.A_tilde = np.array(A_tilde, dtype=np.double)
        else:
            self.A_tilde = np.zeros((self.M_tilde, self.M))
        self._A_tilde_s = sp.Matrix(self.A_tilde)

        if self.A_tilde is not None and (
            self.A_tilde.shape[0] != self.M_tilde or self.A_tilde.shape[1] != self.M
        ):
            raise InvalidAnalogSystemError(
                self, "D_tilde matrix has wrong dimensions. Should be M_tilde x L"
            )

        self.t = sp.Symbol('t', real=True)
        # self.x = [sp.Function(f'x_{i+1}')(self.t) for i in range(self.N)]
        self.omega = sp.Symbol('omega')
        self._atf_lambda = None
        self._ctf_lambda = None

    def _symbolic_x(self, n: int):
        return sp.Function(f'x_{n}')(self.t)

    def derivative(
        self, x: np.ndarray, t: float, u: np.ndarray, s: np.ndarray
    ) -> np.ndarray:
        """Compute the derivative of the analog system.

        Specifically, produces the state derivative

        :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`

        as a function of the state vector :math:`\mathbf{x}(t)`, the given time
        :math:`t`, the input signal value :math:`\mathbf{u}(t)`, and the
        control contribution value :math:`\mathbf{s}(t)`.

        Parameters
        ----------
        x : `array_like`, shape=(N,)
            the state vector evaluated at time t.
        t : `float`
            the time t.
        u : `array_like`, shape=(L,)
            the input signal vector evaluated at time t.
        s : `array_like`, shape=(M,)
            the control contribution evaluated at time t.

        Returns
        -------
        `array_like`, shape=(N,)
            the derivative :math:`\dot{\mathbf{x}}(t)`.
        """
        return np.dot(self.A, x) + np.dot(self.B, u) + np.dot(self.Gamma, s)

    def homogenius_solution(self):
        """Compute the symbolic homogenious solution

        This is done by analytically computing the
        matrix exponential

        :math:`\exp(\mathbf{A} t)`

        Returns
        ----------
        : :py:class:`sympy.Matrix`
            the resulting matrix expression.
        """
        return sp.solvers.ode.systems.matrix_exp(self._A_s, self.t)

    def symbolic_differential_equations(
        self, input: sp.Function, dim: int, input_signal=True
    ):
        """Organise system matrixes into
        ordinary differential equations

        Parameters
        ----------
        input: :py:class:`sympy.Matrix`
            the input function
        dim: `int`
            the dimension of the input
        input_signal: `bool`
            determine if it is a input signal
            or digital control that is to be computed,
            defaults to True (input signal not control).

        Returns
        -------
        : [:py:class:`sympy:Eq]
            the resulting symbolic system equations
        : [:py:class:`sympy:Function`]
            the functions for which the equations relate.
        """
        equations = []
        # functions = []
        for n in range(self.N):
            expr = sp.Float(0)
            for nn in range(self.N):
                expr += self._A_s[n, nn] * self._symbolic_x(nn + 1)
            if input_signal:
                expr += self._B_s[n, dim] * input
            elif self._Gamma_s is not None:
                expr += self._Gamma_s[n, dim] * input
            equations.append(sp.Eq(self._symbolic_x(n + 1).diff(self.t), expr))
        return equations, self._symbolic_x(n + 1)

    def signal_observation(self, x: np.ndarray) -> np.ndarray:
        """Computes the signal observation for a given state vector
        :math:`\mathbf{x}(t)` evaluated at time :math:`t`.

        Specifically, returns

        :math:`\mathbf{y}(t)=\mathbf{C}^\mathsf{T} \mathbf{x}(t)`

        Parameters
        ----------
        x : `array_like`, shape=(N,)
            the state vector.

        Returns
        -------
        `array_like`, shape=(N_tilde,)
            the signal observation.

        """
        return np.dot(self.CT, x)

    def control_observation(
        self, x: np.ndarray, u: np.ndarray = None, s: np.ndarray = None
    ) -> np.ndarray:
        """Computes the control observation for a given state vector :math:`\mathbf{x}(t)`
        evaluated at time :math:`t`.

        Specifically, returns

        :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t) + \\tilde{\mathbf{D}} \mathbf{u}(t)`

        Parameters
        ----------
        x : `array_like`, shape=(N,)
            the state vector.
        u : `array_like`, shape=(L,)
            the input vector
        s : `array_like`, shape=(M,)
            the control signal
        Returns
        -------
        `array_like`, shape=(M_tilde,)
            the control observation.

        """
        if u is None:
            return np.dot(self.Gamma_tildeT, x)
        if s is None:
            np.dot(self.Gamma_tildeT, x) + np.dot(self.D_tilde, u)
        return (
            np.dot(self.Gamma_tildeT, x)
            + np.dot(self.D_tilde, u)
            + np.dot(self.A_tilde, s)
        )

    def _lazy_initialize_ATF(self):
        logger.info("computing analytical transfer function matrix")
        # self._atf_s_matrix = sp.simplify(
        #     self._A_s_P *
        #     (sp.I * self.omega * self._A_s_P_inv * self._A_s_P -
        #      self._A_s_D).inv() * self._A_s_P_inv * self._B_s
        # )
        self._general_atf_s_matrix = sp.simplify(
            (sp.I * self.omega * sp.eye(self.N) - self._A_s).inv()
        )
        self._atf_s_matrix = sp.simplify(self._general_atf_s_matrix * self._B_s)
        self._general_atf_lambda = sp.lambdify((self.omega), self._general_atf_s_matrix)
        self._atf_lambda = sp.lambdify((self.omega), self._atf_s_matrix)

    def _atf_symbolic(self, _omega: float) -> np.ndarray:
        if self._atf_lambda is None:
            self._lazy_initialize_ATF()
        return np.array(self._atf_lambda(_omega)).astype(np.complex128)

    def _atf(self, _omega: float) -> np.ndarray:
        tf = np.dot(
            np.linalg.pinv(complex(0, _omega) * np.eye(self.N) - self.A, rcond=1e-300),
            self.B,
        )
        return tf

    def _general_atf_symbolic(self, _omega: float) -> np.ndarray:
        if self._atf_lambda is None:
            self._lazy_initialize_ATF()
        return np.array(self._general_atf_lambda(_omega)).astype(np.complex128)

    def _general_atf(self, _omega: float) -> np.ndarray:
        tf = np.linalg.pinv(complex(0, _omega) * np.eye(self.N) - self.A, rcond=1e-300)
        return tf

    def _lazy_initialize_CTF(self):
        logger.info("Computing analytical control transfer function.")
        # Diagonalize A
        self._A_s_P, self._A_s_D = self._A_s.diagonalize(normalize=True)
        self._A_s_P_inv = self._A_s_P.inv()
        self._ctf_s_matrix = (
            self._A_s_P
            * (sp.I * self.omega * sp.eye(self.N) - self._A_s_D).inv()
            * self._A_s_P.inv()
            * self._Gamma_s
        )
        self._ctf_lambda = sp.lambdify((self.omega), self._ctf_s_matrix)

    def _ctf(self, _omega: float) -> np.ndarray:
        tf = np.dot(
            np.linalg.pinv(complex(0, _omega) * np.eye(self.N) - self.A, rcond=1e-300),
            self.Gamma,
        )
        return tf
        # if self._atf_lambda is None:
        #     self._lazy_initialize_CTF()
        # return np.array(self._ctf_lambda(_omega)).astype(np.float128)
        # return np.dot(
        #     np.linalg.pinv(complex(0, _omega) * np.eye(self.N) - self.A),
        #     self.Gamma,
        # )

    def control_signal_transfer_function_matrix(self, omega: np.ndarray) -> np.ndarray:
        """Evaluates the transfer functions between control signals and the system
        output.

        Specifically, evaluates

        :math:`\\bar{\mathbf{G}}(\omega) = \mathbf{C}^\mathsf{T} \\left(\mathbf{A} - i \omega \mathbf{I}_N\\right)^{-1} \mathbf{\\Gamma} \in \mathbb{R}^{\\tilde{N} \\times M}`

        for each angular frequency in omega where :math:`\mathbf{I}_N`
        represents a square identity matrix of the same dimensions as
        :math:`\mathbf{A}` and :math:`i=\sqrt{-1}`.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for
            evaluation.

        Returns
        -------
        `array_like`, shape=(N_tilde, M, K)
            the signal transfer function evaluated at K different angular
            frequencies.
        """
        size: int = omega.size
        result = np.zeros((self.N, self.M, size), dtype=complex)
        for index in range(size):
            result[:, :, index] = self._ctf(omega[index])
        # resp = np.einsum("ij,jkl", self.CT, result)
        return np.asarray(result)

    def transfer_function_matrix(
        self,
        omega: np.ndarray,
        symbolic: bool = True,
        general=False,
    ) -> np.ndarray:
        """Evaluate the analog signal transfer function at the angular
        frequencies of the omega array.

        Specifically, evaluates

        :math:`\mathbf{G}(\omega) = \mathbf{C}^\mathsf{T} \\left(\mathbf{A} - i \omega \mathbf{I}_N\\right)^{-1} \mathbf{B} + \mathbf{D}`

        for each angular frequency in omega where :math:`\mathbf{I}_N`
        represents a square identity matrix of the same dimensions as
        :math:`\mathbf{A}` and :math:`i=\sqrt{-1}`.

        Parameters
        ----------
        omega: `array_like`, shape=(K,)
            an array_like object containing the angular frequencies for
            evaluation.
        symbolic: `bool`, `optional`
            solve using symbolic methods, defaults to True.
        general: `bool`, `optional`
            to return general transfer function or not, defaults to False.

        Returns
        -------
        `array_like`, shape=(N_tilde, L, K)
            the signal transfer function evaluated at K different angular
            frequencies.
        """
        size: int = omega.size
        if not general:
            resp = np.zeros((self.N_tilde, self.L, size))
            result = np.zeros((self.N, self.L, size), dtype=complex)
        else:
            resp = np.zeros((self.N_tilde, self.N, size))
            result = np.zeros((self.N, self.N, size), dtype=complex)

        for index in range(size):
            if not general:
                resp[:, :, index] = self.D
            if symbolic and not general:
                result[:, :, index] = self._atf_symbolic(omega[index])
            elif symbolic and general:
                result[:, :, index] = self._general_atf_symbolic(omega[index])
            elif not symbolic and general:
                result[:, :, index] = self._general_atf(omega[index])
            else:
                result[:, :, index] = self._atf(omega[index])
            resp[:, :, index] = self.D
        # resp = np.einsum('ij,jkl', self.CT, result)
        resp = resp + np.tensordot(self.CT, result, axes=((1), (0)))
        return np.asarray(resp)

    def zpk(self, input=0):
        """return zero-pole-gain representation of system

        Parameters
        ----------
        input, `int`, `optional`
            determine for which input (in case of L > 1) to compute zpk, defaults to 0.

        Returns
        -------
        `array_like`, shape=(?, ?, 1)
            z,p,k the zeros, poles and gain of the system
        """
        return scipy.signal.ss2zpk(self.A, self.B, self.CT, self.D, input=0)

    def eta2(self, BW):
        """Compute the eta2 parameter of the system.

        Parameters
        ----------
        BW: `float`
            bandwidth of the system

        Returns
        -------
        `float`
            eta2 parameter of the system at bandwidth BW
        """
        return (
            np.linalg.norm(self.transfer_function_matrix(np.array([2 * np.pi * BW])))
            ** 2
        )

    def __str__(self):
        np.set_printoptions(formatter={'float': '{: 0.2e}'.format})
        return f"The analog system is parameterized as:\nA =\n{np.array(self.A)},\nB =\n{np.array(self.B)},\nCT = \n{np.array(self.CT)},\nGamma =\n{np.array(self.Gamma)},\nGamma_tildeT =\n{np.array(self.Gamma_tildeT)}, and D={self.D}"


class InvalidAnalogSystemError(Exception):
    """Error when detecting faulty analog system specification

    Parameters
    ----------
    system : :py:class:`cbadc.analog_system.AnalogSystem`
        An analog system object
    message: str
        error message
    """

    def __init__(self, system, message):
        self.analog_system = system
        self.message = message
