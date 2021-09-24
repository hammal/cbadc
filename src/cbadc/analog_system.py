"""Analog systems

This module provides a general :py:class:`cbadc.analog_system.AnalogSystem`
class with the necessary functionality to do transient simulations, compute
transfer functions, and exposing the relevant system parameters as
attributes. Additionally, several derived convenience classes are defined
to quickly initialize analog systems of particular structures.
"""
import numpy as np
import numpy.typing as npt
import scipy.signal
from typing import Tuple, List, Union


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
      :math:`\\tilde{\mathbf{s}}(t)=\\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)` and
    * The signal observation :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{C} \mathbf{u}(t)`

    where
    :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}\in\mathbb{R}^{\\tilde{M} \\times N}`
    is the control observation matrix and
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
    >>> print(AnalogSystem(A, B, CT, Gamma, Gamma_tildeT))
    The analog system is parameterized as:
    A =
    [[1. 2.]
     [3. 4.]],
    B =
    [[1.]
     [2.]],
    CT =
    [[1. 0.]
     [2. 1.]],
    Gamma =
    [[-1.  0.]
     [ 0. -5.]],
    and Gamma_tildeT =
    [[1. 2.]
     [0. 1.]]

    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(
        self,
        A: npt.ArrayLike,
        B: npt.ArrayLike,
        CT: npt.ArrayLike,
        Gamma: npt.ArrayLike,
        Gamma_tildeT: npt.ArrayLike,
        D: Union[npt.ArrayLike, None] = None,
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
        """

        self.A = np.array(A, dtype=np.double)
        self.B = np.array(B, dtype=np.double)
        self.CT = np.array(CT, dtype=np.double)
        if Gamma is not None:
            self.Gamma = np.array(Gamma, dtype=np.double)
            if self.Gamma.shape[0] != self.A.shape[0]:
                raise InvalidAnalogSystemError(
                    self, "N does not agree with control input matrix Gamma."
                )
            self.M = self.Gamma.shape[1]
        else:
            self.Gamma = None
            self.M = 0

        if Gamma_tildeT is not None:
            self.Gamma_tildeT = np.array(Gamma_tildeT, dtype=np.double)
            if self.Gamma_tildeT.shape[1] != self.A.shape[0]:
                raise InvalidAnalogSystemError(
                    self,
                    """N does not agree with control observation matrix
                    Gamma_tilde.""",
                )
            self.M_tilde = self.Gamma_tildeT.shape[0]
        else:
            self.Gamma_tildeT = None
            self.M_tilde = 0

        self.N = self.A.shape[0]

        if self.A.shape[0] != self.A.shape[1]:
            raise InvalidAnalogSystemError(self, "system matrix not square")

        # ensure matrices
        if len(self.B.shape) == 1:
            self.B = self.B.reshape((self.N, 1))
        if len(self.CT.shape) == 1:
            self.CT = self.CT.reshape((1, self.N))

        self.L = self.B.shape[1]
        self.N_tilde = self.CT.shape[0]

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

        if self.D is not None and (
            self.D.shape[0] != self.N_tilde or self.D.shape[1] != self.L
        ):
            raise InvalidAnalogSystemError(
                self, "D matrix has wrong dimensions. Should be N_tilde x L"
            )

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

    def control_observation(self, x: np.ndarray) -> np.ndarray:
        """Computes the control observation for a given state vector :math:`\mathbf{x}(t)`
        evaluated at time :math:`t`.

        Specifically, returns

        :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`

        Parameters
        ----------
        x : `array_like`, shape=(N,)
            the state vector.

        Returns
        -------
        `array_like`, shape=(M_tilde,)
            the control observation.

        """
        return np.dot(self.Gamma_tildeT, x)

    def _atf(self, _omega: float) -> np.ndarray:
        tf = np.dot(
            np.linalg.pinv(complex(0, _omega) * np.eye(self.N) - self.A, rcond=1e-300),
            self.B,
        )
        return tf

    def _ctf(self, _omega: float) -> np.ndarray:
        return np.dot(
            np.linalg.pinv(complex(0, _omega) * np.eye(self.N) - self.A),
            self.Gamma,
        )

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
        resp = np.einsum("ij,jkl", self.CT, result)
        return np.asarray(resp)

    def transfer_function_matrix(self, omega: np.ndarray) -> np.ndarray:
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

        Returns
        -------
        `array_like`, shape=(N_tilde, L, K)
            the signal transfer function evaluated at K different angular
            frequencies.
        """
        size: int = omega.size
        result = np.zeros((self.N, self.L, size), dtype=complex)
        resp = np.zeros((self.N_tilde, self.L, size))
        for index in range(size):
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

    def __str__(self):
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


class ChainOfIntegrators(AnalogSystem):
    """Represents an chain-of-integrators analog system.

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and
    creates a convenient way of creating chain-of-integrator A/D analog
    systems. For more information about chain-of-integrator ADCs see
    `chain-of-Integrator ADC <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf?sequence=1&isAllowed=y&page=96/>`_.


    Chain-of-integrators analog systems are system goverened by the
    differential equations,

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`

    where

    :math:`\mathbf{A} = \\begin{pmatrix} \\rho_1 & \\\ \\beta_2 & \\rho_2  \\\ & \ddots & \ddots \\\ & & \\beta_N & \\rho_N \\end{pmatrix}`

    :math:`\mathbf{B} = \\begin{pmatrix} \\beta_1 & 0 & \cdots & 0 \\end{pmatrix}^\mathsf{T}`

    :math:`\mathbf{C}^\mathsf{T} = \mathbf{I}_N`

    :math:`\mathbf{\Gamma} = \\begin{pmatrix} \\kappa_1  \\\  & \ddots \\\ & & \\kappa_N  \\end{pmatrix}`

    :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T} = \mathbf{I}_N`

    Parameters
    ----------
    beta : `array_like`, shape=(N,)
        vector with per integrator signal amplification :math:`\\begin{pmatrix}\\beta_1 & \cdots & \\beta_N \\end{pmatrix}`.
    rho : `array_like`, shape=(N,)
        local feedback factor vector :math:`\\begin{pmatrix}\\rho_1 & \cdots & \\rho_N \\end{pmatrix}`.
    kappa : `array_like`, shape=(N,) or shape=(M, N),
        control gain vector :math:`\\begin{pmatrix}\\kappa_1 & \cdots & \\kappa_N \\end{pmatrix}`.
    kappa_tilde : `array_like`, shape=(N, M_tilde), `optional`

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

    See also
    --------
    :py:class:`cbadc.analog_system.AnalogSystem`

    Example
    -------
    >>> import numpy as np
    >>> from cbadc.analog_system import ChainOfIntegrators
    >>> beta = np.array([100, 100, 100])
    >>> rho = np.array([-1, -1, -1])
    >>> kappa = np.array([[100, 100, 100]]).transpose()
    >>> print(ChainOfIntegrators(beta, rho, kappa))
    The analog system is parameterized as:
    A =
    [[ -1.   0.   0.]
     [100.  -1.   0.]
     [  0. 100.  -1.]],
    B =
    [[100.]
     [  0.]
     [  0.]],
    CT =
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]],
    Gamma =
    [[100.   0.   0.]
     [  0. 100.   0.]
     [  0.   0. 100.]],
    and Gamma_tildeT =
    [[-1.  0.  0.]
     [ 0. -1.  0.]
     [ 0.  0. -1.]]


    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, beta: np.ndarray, rho: np.ndarray, kappa: np.ndarray):
        """Create an chain-of-integrators analog system."""
        if beta.shape[0] != beta.size:
            InvalidAnalogSystemError(self, "beta must be a one dimensional vector")
        if rho.shape[0] != rho.size:
            InvalidAnalogSystemError(self, "rho must be a one dimensional vector")
        if kappa.shape[0] != rho.size:
            InvalidAnalogSystemError(
                self,
                "kappa must be a one dimensional vector of size N or matrix with N rows",
            )
        if beta.size != rho.size and rho.size != kappa[:, 0].size:
            InvalidAnalogSystemError(
                self, "beta, rho, kappa vector must be of same size"
            )

        # State space order
        N = beta.size

        # Analog system parameters
        A = np.diag(rho) + np.diag(beta[1:], k=-1)
        B = np.zeros((N, 1))
        B[0] = beta[0]
        CT = np.eye(N)

        # Check if Kappa is specified as a vector
        if kappa.shape[1] == 1:
            Gamma = np.diag(kappa.flatten())
        else:
            Gamma = np.array(kappa, dtype=np.double)

        Gamma_tildeT = -Gamma.transpose()
        for row_index in range(Gamma_tildeT.shape[0]):
            Gamma_tildeT[row_index, :] = Gamma_tildeT[row_index, :] / np.linalg.norm(
                Gamma_tildeT[row_index, :]
            )

        # initialize parent class
        AnalogSystem.__init__(self, A, B, CT, Gamma, Gamma_tildeT)


class LeapFrog(AnalogSystem):
    """Represents an leap-frog analog system.

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and creates a convenient
    way of creating leap-frog A/D analog systems. For more information about leap-frog ADCs see
    `Leap Frog ADC <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf?sequence=1&isAllowed=y&page=126/>`_.


    A leap-frog analog system is goverened by the differential equations,

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`

    where

    :math:`\mathbf{A} = \\begin{pmatrix} \\rho_1 & \\rho_2 \\\ \\beta_2 & 0 & \\rho_3 \\\ & \ddots & \ddots & \ddots \\\ & & \\beta_N & 0 & \\rho_{N+1} \\end{pmatrix}`

    :math:`\mathbf{B} = \\begin{pmatrix} \\beta_1 & 0 & \cdots & 0 \\end{pmatrix}^\mathsf{T}`

    :math:`\mathbf{C}^\mathsf{T} = \mathbf{I}_N`

    :math:`\mathbf{\Gamma} = \\begin{pmatrix} \\kappa_1  \\\  & \ddots \\\ & & \\kappa_N  \\end{pmatrix}`

    :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T} = \mathbf{I}_N`

    Parameters
    ----------
    beta : `array_like`, shape=(N,)
        vector with per integrator signal amplification :math:`\\begin{pmatrix}\\beta_1 & \cdots & \\beta_N \\end{pmatrix}`.
    rho : `array_like`, shape=(N,)
        local feedback factor vector :math:`\\begin{pmatrix}\\rho_1 & \cdots & \\rho_N \\end{pmatrix}`.
    kappa : `array_like`, shape=(N,)
        control gain vector :math:`\\begin{pmatrix}\\kappa_1 & \cdots & \\kappa_N \\end{pmatrix}`.

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

    See also
    --------
    :py:class:`cbadc.analog_system.ChainOfIntegrators`

    Example
    -------
    >>> import numpy as np
    >>> from cbadc.analog_system import LeapFrog
    >>> beta = np.array([101, 102, 103])
    >>> rho = np.array([-1, -2, -3])
    >>> kappa = np.arange(100,109).reshape((3, 3))
    >>> print(LeapFrog(beta, rho, kappa))
    The analog system is parameterized as:
    A =
    [[ -1.  -2.   0.]
     [102.   0.  -3.]
     [  0. 103.   0.]],
    B =
    [[101.]
     [  0.]
     [  0.]],
    CT =
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]],
    Gamma =
    [[100. 101. 102.]
     [103. 104. 105.]
     [106. 107. 108.]],
    and Gamma_tildeT =
    [[-0.5603758  -0.57718708 -0.59399835]
     [-0.56054048 -0.5771902  -0.59383992]
     [-0.560702   -0.57719323 -0.59368447]]


    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, beta: np.ndarray, rho: np.ndarray, kappa: np.ndarray):
        """Create an leap-frog analog system."""
        if beta.shape[0] != beta.size:
            InvalidAnalogSystemError(self, "beta must be a one dimensional vector")
        if rho.shape[0] != rho.size:
            InvalidAnalogSystemError(self, "rho must be a one dimensional vector")
        if kappa.shape[0] != kappa.size:
            InvalidAnalogSystemError(self, "kappa must be a one dimensional vector")
        if beta.size != rho.size and rho.size != kappa.size:
            InvalidAnalogSystemError(
                self, "beta, rho, kappa vector must be of same size"
            )

        # State space order
        N = beta.size

        # Analog system parameters
        A = np.diag(rho[1:], k=1) + np.diag(beta[1:], k=-1)
        A[0, 0] = rho[0]
        B = np.zeros((N, 1))
        B[0] = beta[0]
        CT = np.eye(N)

        # Check if Kappa is specified as a vector
        if kappa.shape[1] == 1:
            Gamma = np.diag(kappa.flatten())
        else:
            Gamma = np.array(kappa, dtype=np.double)

        Gamma_tildeT = -Gamma.transpose()
        for row_index in range(Gamma_tildeT.shape[0]):
            Gamma_tildeT[row_index, :] = Gamma_tildeT[row_index, :] / np.linalg.norm(
                Gamma_tildeT[row_index, :]
            )

        # initialize parent class
        AnalogSystem.__init__(self, A, B, CT, Gamma, Gamma_tildeT)


class ButterWorth(AnalogSystem):
    """A Butterworth filter analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and creates a convenient
    way of creating Butterworth filter analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} u(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{D} u(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, :math:`\mathbf{C}^\mathsf{T}`, and :math:`\mathbf{D}`
    are determined using the :py:func:`scipy.signal.iirfilter`.

    Furthermore, as this system is intended as a pure filter and therefore have no

    :math:`\mathbf{\Gamma}` and :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}` specified.

    Parameters
    ----------
    N: `int`
        filter order
    Wn: `float`
        array containing the critical frequency of the filter.
        The frequency is specified in rad/s.


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
    D: `array_like`, shape=(N_tilde, L)
        direct matrix
    Gamma : None
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : None
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}`.
    Wn : float
        critical frequency of filter.

    See also
    --------
    :py:class:`cbadc.analog_system.ChebyshevI`
    :py:class:`cbadc.analog_system.ChebyshevII`
    :py:class:`cbadc.analog_system.Cauer`



    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, N: int, Wn: float):
        """Create a Butterworth filter"""
        # State space order
        self.Wn = Wn

        # Create filter as chain of biquadratic filters
        z, p, k = scipy.signal.iirfilter(
            N, Wn, analog=True, btype="lowpass", ftype="butter", output="zpk"
        )

        A, B, CT, D = zpk2abcd(z, p, k)
        AnalogSystem.__init__(self, A, B, CT, None, None, D)


class ChebyshevI(AnalogSystem):
    """A Chebyshev type I filter's analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and is a convenient
    way of creating Chebyshev type I filter's analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} u(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{D} u(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, :math:`\mathbf{C}^\mathsf{T}`, and :math:`\mathbf{D}`
    are determined using the :py:func:`scipy.signal.iirfilter`.

    Furthermore, as this system is intended as a pure filter and therefore have no

    :math:`\mathbf{\Gamma}` and :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}` specified.

    Parameters
    ----------
    N: `int`
        filter order
    Wn: (float, float)
        array containing the critical frequencies (low, high) of the filter.
        The frequencies are specified in rad/s.
    rp: float
        maximum ripple in passband. Specified in dB.


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
    D: `array_like`, shape=(N_tilde, L)
        direct matrix
    Gamma : None
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : None
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}`.
    Wn : float
        critical frequency of filter.

    See also
    --------
    :py:class:`cbadc.analog_system.ButterWorth`
    :py:class:`cbadc.analog_system.ChebyshevII`
    :py:class:`cbadc.analog_system.Cauer`



    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, N: int, Wn: float, rp: float):
        """Create a Chebyshev type I filter"""
        # State space order
        self.Wn = Wn
        self.rp = rp

        z, p, k = scipy.signal.iirfilter(
            N, Wn, rp, analog=True, btype="lowpass", ftype="cheby1", output="zpk"
        )
        A, B, CT, D = zpk2abcd(z, p, k)
        AnalogSystem.__init__(self, A, B, CT, None, None, D)


class ChebyshevII(AnalogSystem):
    """A Chebyshev type II filter's analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and is a convenient
    way of creating Chebyshev type II filter's analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} u(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{D} u(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, :math:`\mathbf{C}^\mathsf{T}`, and :math:`\mathbf{D}`
    are determined using the :py:func:`scipy.signal.iirfilter`.

    Furthermore, as this system is intended as a pure filter and therefore have no

    :math:`\mathbf{\Gamma}` and :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}` specified.

    Parameters
    ----------
    N: `int`
        filter order
    Wn: (float, float)
        array containing the critical frequencies (low, high) of the filter.
        The frequencies are specified in rad/s.
    rs: float
        minimum attenutation in stopband. Specified in dB.


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
    D: `array_like`, shape=(N_tilde, L)
        direct matrix
    Gamma : None
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : None
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}`.
    Wn : float
        critical frequency of filter.
    rs: float
        minimum attenuation in stop band (dB).

    See also
    --------
    :py:class:`cbadc.analog_system.ButterWorth`
    :py:class:`cbadc.analog_system.ChebyshevI`
    :py:class:`cbadc.analog_system.Cauer`



    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, N: int, Wn: float, rs: float):
        """Create a Chebyshev type II filter"""
        # State space order
        self.Wn = Wn
        self.rs = rs
        z, p, k = scipy.signal.iirfilter(
            N, Wn, rs=rs, analog=True, btype="lowpass", ftype="cheby2", output="zpk"
        )
        A, B, CT, D = zpk2abcd(z, p, k)
        AnalogSystem.__init__(self, A, B, CT, None, None, D)


class Cauer(AnalogSystem):
    """A Cauer (elliptic) filter's analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and is a convenient
    way of creating Cauer filter's analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} u(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{D} u(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, :math:`\mathbf{C}^\mathsf{T}`, and :math:`\mathbf{D}`
    are determined using the :py:func:`scipy.signal.iirfilter`.

    Furthermore, as this system is intended as a pure filter and therefore have no

    :math:`\mathbf{\Gamma}` and :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}` specified.

    Parameters
    ----------
    N: `int`
        filter order
    Wn: (float, float)
        array containing the critical frequencies (low, high) of the filter.
        The frequencies are specified in rad/s.
    rp: float
        maximum ripple in passband. Specified in dB.
    rs: float
        minimum attenutation in stopband. Specified in dB.


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
    D: `array_like`, shape=(N_tilde, L)
        direct matrix
    Gamma : None
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : None
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}`.
    Wn : float
        critical frequency of filter.
    rp: float
        maximal ripple in passband, specified in  (dB).
    rs: float
        minimum attenuation in stop band (dB).

    See also
    --------
    :py:class:`cbadc.analog_system.ButterWorth`
    :py:class:`cbadc.analog_system.ChebyshevI`
    :py:class:`cbadc.analog_system.ChebyshevII`



    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, N: int, Wn: float, rp: float, rs: float):
        """Create a Cauer filter"""
        # State space order
        self.Wn = Wn
        self.rp = rp
        self.rs = rs

        z, p, k = scipy.signal.iirfilter(
            N, Wn, rp, rs, analog=True, btype="lowpass", ftype="ellip", output="zpk"
        )
        A, B, CT, D = zpk2abcd(z, p, k)
        AnalogSystem.__init__(self, A, B, CT, None, None, D)


class IIRDesign(AnalogSystem):
    """An analog signal designed using standard IIRDesign tools

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and is a convenient
    way of creating IIR filters in an analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} u(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t) + \mathbf{D} u(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, :math:`\mathbf{C}^\mathsf{T}`, and :math:`\mathbf{D}`
    are determined using the :py:func:`scipy.signal.iirdesign`.

    Furthermore, as this system is intended as a pure filter and therefore have no

    :math:`\mathbf{\Gamma}` and :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}` specified.

    Parameters
    ----------
    wp, ws: `float or array_like`, shape=(2,)
        Passband and stopband edge frequencies. Possible values are scalars (for lowpass and highpass filters) or ranges (for bandpass and bandstop filters). For digital filters, these are in the same units as fs. By default, fs is 2 half-cycles/sample, so these are normalized from 0 to 1, where 1 is the Nyquist frequency. For example:

        * Lowpass: wp = 0.2, ws = 0.3
        * Highpass: wp = 0.3, ws = 0.2
        * Bandpass: wp = [0.2, 0.5], ws = [0.1, 0.6]
        * Bandstop: wp = [0.1, 0.6], ws = [0.2, 0.5]

        wp and ws are angular frequencies (e.g., rad/s). Note, that for bandpass and bandstop filters passband must lie strictly inside stopband or vice versa.

    gpass: `float`
        The maximum loss in the passband (dB).

    gstop: `float`
        The minimum attenuation in the stopband (dB).

    ftype: `string`, `optional`
        IIR filter type, defaults to ellip. Complete list of choices:

        * Butterworth : ‘butter’
        * Chebyshev I : ‘cheby1’
        * Chebyshev II : ‘cheby2’
        * Cauer/elliptic: ‘ellip’
        * Bessel/Thomson: ‘bessel’


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
    D: `array_like`, shape=(N_tilde, L)
        direct matrix
    Gamma : None
        control input matrix :math:`\mathbf{\Gamma}`.
    Gamma_tildeT : None
        control observation matrix :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T}`.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.ticker
    >>> from cbadc.analog_system import IIRDesign
    >>> wp = 2 * np.pi * 1e3
    >>> ws = 2 * np.pi * 2e3
    >>> gpass = 0.1
    >>> gstop = 80
    >>> filter = IIRDesign(wp, ws, gpass, gstop)
    >>> f = np.logspace(1, 5)
    >>> w = 2 * np.pi * f
    >>> tf = filter.transfer_function_matrix(w)
    >>> fig, ax1 = plt.subplots()
    >>> ax1.set_title('Analog filter frequency response')
    >>> ax1.set_ylabel('Amplitude [dB]', color='b')
    >>> ax1.set_xlabel('Frequency [Hz]')
    >>> ax1.semilogx(f, 20 * np.log10(np.abs(tf[0, 0, :])))
    >>> ax1.grid()
    >>> ax2 = ax1.twinx()
    >>> angles = np.unwrap(np.angle(tf[0, 0, :]))
    >>> ax2.plot(f, angles, 'g')
    >>> ax2.set_ylabel('Angle (radians)', color='g')
    >>> ax2.grid()
    >>> ax2.axis('tight')
    >>> nticks = 8
    >>> ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
    >>> ax2.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))

    See also
    --------
    :py:class:`cbadc.analog_system.ButterWorth`
    :py:class:`cbadc.analog_system.ChebyshevI`
    :py:class:`cbadc.analog_system.ChebyshevII`
    :py:class:`cbadc.analog_system.Cauer`

    Raises
    ------
    :py:class:`InvalidAnalogSystemError`
        For faulty analog system parametrization.
    """

    def __init__(self, wp, ws, gpass, gstop, ftype="ellip"):
        """Create a IIR filter"""
        z, p, k = scipy.signal.iirdesign(
            wp, ws, gpass, gstop, analog=True, ftype=ftype, output="zpk"
        )
        A, B, CT, D = zpk2abcd(z, p, k)
        AnalogSystem.__init__(self, A, B, CT, None, None, D)


def sos2abcd(sos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform a series of biquad (second order systems (sos)) filters into
    their A,B,C,D state space model equivalent.

    Specifcally, for a filter with the transfer function

    :math:`\Pi_{\ell = 1}^{L} \\frac{b_{\ell,0}s^2 + b_{\ell,1} s + b_{\ell,2}}{s^2 + a_{\ell,1}s + a_{\ell,2}}`

    represented in a sos matrix

    :math:`\\text{sos} = \\begin{pmatrix} & & \\vdots  \\\ b_{\ell,0}, & b_{\ell,1}, & b_{\ell,2}, & 1, & a_{\ell,1}, & a_{\ell,2} \\\ & & \\vdots \\end{pmatrix}`

    is represented in a controllable canonical state space representation form

    :math:`\\begin{pmatrix} \dot{x}_{\ell, 1}(t) \\\ \dot{x}_{\ell,2}(t) \\end{pmatrix} = \\begin{pmatrix}1 , & 0 \\\ -a_{\ell,2}, & -a_{\ell,1} \\end{pmatrix} \\begin{pmatrix} x_{\ell,1}(t) \\\ x_{\ell,2}(t)\\end{pmatrix} + \\begin{pmatrix} 0 \\\ 1 \\end{pmatrix} u_\ell(t)`

    :math:`y_\ell(t) = \\begin{pmatrix}b_{\ell,2} - b_{\ell,0} * a_{\ell,2}, & b_{\ell,1} - b_{\ell,0}*a_{\ell,1} \\end{pmatrix} \\begin{pmatrix} x_{\ell,1}(t) \\\ x_{\ell,2}(t) \\end{pmatrix}  + \\begin{pmatrix}b_{\ell,0}\\end{pmatrix} u_\ell(t)`

    which are then chained together using :py:func:`cbadc.analog_system.chain`.

    Parameters
    ----------
    sos: np.ndarray, shape(N, 6)
        biquad equations

    Returns
    A: numpy.ndarray, shape=(2 * L, 2 * L)
        a joint state transition matrix.
    B: numpy.ndarray, shape=(2 * L, 1)
        a joint input matrix.
    C: numpy.ndarray, shape=(1, 2 * L)
        a joint signal observation matrix.
    D: numpy.ndarray, shape=(1, 1)
        The direct matrix.
    """
    biquadratic_analog_systems = []
    for row in range(sos.shape[0]):
        b_0 = sos[row, 0]
        b_1 = sos[row, 1]
        b_2 = sos[row, 2]
        a_0 = sos[row, 3]
        a_1 = sos[row, 4]
        a_2 = sos[row, 5]

        if a_0 != 1.0:
            b_0 /= a_0
            b_1 /= a_0
            b_2 /= a_0
            a_1 /= a_0
            a_2 /= a_0
            a_0 = 1.0

        A = np.array([[0.0, 1.0], [-a_2, -a_1]])
        B = np.array([[0.0], [1.0]])
        CT = np.array([[(b_2 - b_0 * a_2), (b_1 - b_0 * a_1)]])
        D = np.array([[b_0]])
        biquadratic_analog_systems.append(AnalogSystem(A, B, CT, None, None, D))
    chained_system = chain(biquadratic_analog_systems)
    return chained_system.A, chained_system.B, chained_system.CT, chained_system.D


def tf2abcd(
    b: np.ndarray, a: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform a transferfunctions into a controllable canonical state space form.

    Specifcally, for a filter with the transfer function

    :math:`\\frac{\Sigma_{\ell = 0}^{L} b_{\ell}s^\ell}{s^{L} + \Sigma_{\ell=0}^{L-1} a_{k} s^\ell}`

    is represented in a controllable canonical state space representation form

    :math:`\dot{\mathbf{x}}(t) = \\begin{pmatrix}0, & 1, & 0 \\\ & \\ddots & \ddots \\\ 0 & \\dots &  0, & 1 \\\ -a_{0}, & \\dots, &  & -a_{L-1} \\end{pmatrix} \mathbf{x}(t) + \\begin{pmatrix} 0 \\\ \\vdots \\\ 0 \\\ 1 \\end{pmatrix} u(t)`

    :math:`y_\ell(t) = \\begin{pmatrix}b_{0} - b_L a_0, & \\dots, & b_{L-1} - b_L a_{L-1} \\end{pmatrix} \mathbf{x}(t) + \\begin{pmatrix} b_L\\end{pmatrix} u(t)`

    Parameters
    ----------
    b: np.ndarray, shape=(L+1,)
        transferfunction nominator :math:`\\begin{pmatrix}b_{L}, & \\dots, & b_{0}\\end{pmatrix}`.
    a: np.ndarray, shape=(L,)
        transferfunction denominator :math:`\\begin{pmatrix}a_{L-1}, & \\dots, & a_{0}\\end{pmatrix}`.

    Returns
    A: numpy.ndarray, shape=(L, L)
        a joint state transition matrix.
    B: numpy.ndarray, shape=(L, 1)
        a joint input matrix.
    CT: numpy.ndarray, shape=(1,L)
        a joint signal observation matrix.
    D: numpy.ndarray, shape=(1,1)
        direct transition matrix.
    """
    if b.size != (a.size + 1) or len(b) > b.size or len(a) > a.size:
        raise BaseException(
            f"a and b are not correctly configures with b={b} and a={a}"
        )
    L = a.size
    A = np.zeros((a.size, a.size))
    B = np.zeros((a.size, 1))
    CT = np.zeros((1, a.size))
    D = np.zeros((1, 1))
    A[:-1, 1:] = np.eye(L - 1)
    A[-1, :] = -a[::-1]
    B[-1, 0] = 1.0
    CT[0, :] = (b[1::] - b[0] * a[::])[::-1]
    D[0, 0] = b[0]
    return A, B, CT, D


def chain(analog_systems: List[AnalogSystem]) -> AnalogSystem:
    """Construct an analog system by chaining several analog systems.

    The chaining is achieved by chainging :math:`\hat{N}` systems,
    parameterized by
    :math:`\mathbf{A}_1, \mathbf{B}_1, \mathbf{C}^\mathsf{T}_1, \mathbf{D}_1, \mathbf{\Gamma}_1, \\tilde{\mathbf{\Gamma}}_1, \\dots, \mathbf{A}_{\hat{N}}, \mathbf{B}_{\hat{N}}, \mathbf{C}^\mathsf{T}_{\hat{N}}, \mathbf{D}_{\hat{N}}, \mathbf{\Gamma}_{\hat{N}}, \\tilde{\mathbf{\Gamma}}_{\hat{N}}`,
    as

    :math:`\mathbf{A} = \\begin{pmatrix} \mathbf{A}_1 \\\ \mathbf{B}_2 \mathbf{C}_1^\mathsf{T} & \mathbf{A}_2 \\\  \mathbf{B}_3 \mathbf{D}_2 \mathbf{C}_1^\mathsf{T} & \mathbf{B}_3 \mathbf{C}_2^\mathsf{T} & \mathbf{A}_3 \\\ \mathbf{B}_4 \mathbf{D}_3 \mathbf{D}_2 \mathbf{C}_1^\mathsf{T} & \mathbf{B}_4\mathbf{D}_3\mathbf{C}_2^\mathsf{T} & \mathbf{B}_4 \mathbf{C}_3^\mathsf{T} & \mathbf{A}_4  \\\ \\vdots & \\vdots & \\vdots & \\vdots & \\ddots  \\end{pmatrix}`

    :math:`\mathbf{B} = \\begin{pmatrix} \mathbf{B}_1 \\\ \mathbf{B}_2 \mathbf{D}_1 \\\ \mathbf{B}_3 \mathbf{D}_2 \mathbf{D}_1 \\\ \mathbf{B}_4 \mathbf{D}_3 \mathbf{D}_2 \mathbf{D}_1 \\\ \\vdots \\end{pmatrix}`

    :math:`\mathbf{C}^\mathsf{T} = \\begin{pmatrix} \mathbf{D}_{\hat{N}}\\cdots \mathbf{D}_2 \mathbf{C}_1^\mathsf{T} & \mathbf{D}_{\hat{N}}\\cdots \mathbf{D}_3 \mathbf{C}_2^\mathsf{T} & \mathbf{D}_{\hat{N}} \\cdots \mathbf{D}_4 \mathbf{C}_3^{\mathsf{T}} & \\dots & \mathbf{D}_{\hat{N}} \mathbf{C}_{\hat{N}-1}^\mathsf{T} & \mathbf{C}_{\hat{N}}^\mathsf{T} \\end{pmatrix}`

    :math:`\mathbf{\Gamma} = \\begin{pmatrix} \ddots \\\ & \mathbf{\Gamma}_\ell  \\\ &  & \mathbf{\Gamma}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix}`

    :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T} = \\begin{pmatrix} \ddots \\\ & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_\ell  \\\ &  & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix}`

    :math:`\mathbf{D} = \mathbf{D}_{\hat{N}} \mathbf{D}_{\hat{N}-1} \\dots \mathbf{D}_{1}`

    where :math:`N = \\sum_\ell N_\ell`, :math:`L = L_1`, and :math:`\\tilde{N} = N_{-1}`.
    Above the index :math:`-1` refers to the last index of the list.

    The systems in the list are chained in the order they are listed.

    Parameters
    ----------
    analog_systems: List[:py:class:`cbadc.analog_system.AnalogSystem`]
        a list of analog systems to be chained.

    Returns
    -------
    :py:class:`cbadc.analog_system.AnalogSystem`
        a new analog system
    """
    N: int = np.sum(np.array([analog_system.N for analog_system in analog_systems]))
    M: int = np.sum(np.array([analog_system.M for analog_system in analog_systems]))
    M_tilde: int = np.sum(
        np.array([analog_system.M_tilde for analog_system in analog_systems])
    )
    L: int = analog_systems[0].L

    A = np.zeros((N, N))
    B = np.zeros((N, L))
    CT = np.zeros((analog_systems[0].CT.shape[0], N))

    CT[
        : analog_systems[0].CT.shape[0], : analog_systems[0].CT.shape[1]
    ] = analog_systems[0].CT

    Gamma = np.zeros((N, M))
    Gamma_tilde = np.zeros((M_tilde, N))
    D = np.eye(analog_systems[0].N_tilde, analog_systems[0].L)

    n: int = 0
    m: int = 0
    m_tilde: int = 0
    for analog_system in analog_systems:
        n_end = n + analog_system.N
        m_end = m + analog_system.M
        m_tilde_end = m_tilde + analog_system.M_tilde

        A[n:n_end, :] = np.dot(analog_system.B, CT)
        A[n:n_end, n:n_end] = analog_system.A

        B[n:n_end, :] = np.dot(analog_system.B, D)

        D = np.dot(analog_system.D, D)

        CT = np.dot(analog_system.D, CT)
        CT[:, n:n_end] = analog_system.CT

        Gamma[n:n_end, m:m_end] = analog_system.Gamma

        Gamma_tilde[m_tilde:m_tilde_end, n:n_end] = analog_system.Gamma_tildeT

        n += analog_system.N
        m += analog_system.M
        m_tilde += analog_system.M_tilde
    return AnalogSystem(A, B, CT, Gamma, Gamma_tilde, D)


def stack(analog_systems: List[AnalogSystem]) -> AnalogSystem:
    """Construct an analog system by stacking several analog systems in parallel.

    The parallel stack is achieved by constructing

    :math:`\mathbf{A} = \\begin{pmatrix}\ddots \\\ & \mathbf{A}_\ell \\\ & &  \mathbf{A}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{N \\times N}`

    :math:`\mathbf{B} = \\begin{pmatrix} \ddots \\\ & \mathbf{B}_\ell \\\ & &  \mathbf{B}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{N \\times L}`

    :math:`\mathbf{C}^\mathsf{T} = \\begin{pmatrix} \ddots \\\ & \mathbf{C}_\ell \\\ & &  \mathbf{C}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{\\tilde{N} \\times N}`

    :math:`\mathbf{\Gamma} = \\begin{pmatrix} \ddots \\\ & \mathbf{\Gamma}_\ell  \\\ &  & \mathbf{\Gamma}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{N \\times M}`

    :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T} = \\begin{pmatrix} \ddots \\\ & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_\ell  \\\ &  & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{\\tilde{M} \\times N}`

    :math:`\mathbf{D} = \\begin{pmatrix}\\ddots \\\ & \mathbf{D}_\ell \\\ && \\ddots\\end{pmatrix}`

    where for :math:`n` systems
    :math:`L = \\sum_{\ell = 1}^n L_\ell`, :math:`M = \\sum_{\ell = 1}^n M_\ell`,
    :math:`\\tilde{M} = \\sum_{\ell = 1}^n \\tilde{M}_\ell`, :math:`N = \\sum_{\ell = 1}^n N_\ell`,
    and :math:`\\tilde{N} = \\sum_{\ell = 1}^n \\tilde{N}_\ell`.

    Parameters
    ----------
    analog_systems: List[:py:class:`cbadc.analog_system.AnalogSystem`]
        a list of analog systems to be chained.

    Returns
    -------
    :py:class:`cbadc.analog_system.AnalogSystem`
        a new analog system
    """
    N: int = np.sum(np.array([analog_system.N for analog_system in analog_systems]))
    M: int = np.sum(np.array([analog_system.M for analog_system in analog_systems]))
    M_tilde: int = np.sum(
        np.array([analog_system.M_tilde for analog_system in analog_systems])
    )
    L: int = np.sum(np.array([analog_system.L for analog_system in analog_systems]))
    N_tilde: int = np.sum(
        np.array([analog_system.N_tilde for analog_system in analog_systems])
    )

    A = np.zeros((N, N))
    B = np.zeros((N, L))
    CT = np.zeros((N_tilde, N))
    Gamma = np.zeros((N, M))
    Gamma_tilde = np.zeros((M_tilde, N))
    D = np.zeros((N_tilde, L))

    n: int = 0
    m: int = 0
    m_tilde: int = 0
    l: int = 0
    n_tilde = 0
    for analog_system in analog_systems:
        n_end = n + analog_system.N
        m_end = m + analog_system.M
        m_tilde_end = m_tilde + analog_system.M_tilde
        l_end = l + analog_system.L
        n_tilde_end = n_tilde + analog_system.N_tilde

        A[n:n_end, n:n_end] = analog_system.A
        B[n:n_end, l:l_end] = analog_system.B
        CT[n_tilde:n_tilde_end, n:n_end] = analog_system.CT
        Gamma[n:n_end, m:m_end] = analog_system.Gamma
        Gamma_tilde[m_tilde:m_tilde_end, n:n_end] = analog_system.Gamma_tildeT

        D[n_tilde:n_tilde_end, l:l_end] = analog_system.D

        l += analog_system.L
        n += analog_system.N
        n_tilde += analog_system.N_tilde
        m += analog_system.M
        m_tilde += analog_system.M_tilde
    return AnalogSystem(A, B, CT, Gamma, Gamma_tilde, D)


def zpk2abcd(z, p, k):
    """Convert zeros and poles into A, B, C, D matrix

    Futhermore, the transfer function is divided into
    sequences of products of Biquad filters.

    Specifically, for a transfer function

    :math:`k \\cdot \\frac{(s - z_N)\\dots(s - z_1)}{(s-p_N)\\dots(s-p_1)}`

    we partition the transfer function into biquadratic blocks as

    :math:`\\Pi_{\ell=1}^{N / 2} \\frac{Z(s)_\ell}{(s - p_{1,\ell})(s-p_{2,\ell})}`

    where
    :math:`Z(s)_\ell = \\begin{cases} 1 & \\text{if no zeros are specified} \\\  (s - z_{1,\ell})(s - z_{2,\ell}) & \\text{for real valued zeros}  \\\ (s-z_{1,\ell})(s-\\bar{z}_{1,\ell}) & \\text{for complex conjugate zero-pairs.} \\end{cases}`


    The poles are and zeros are sorted in the following order:

    1. Complex conjugate pairs
    2. Decreasing absolute magnitude

    """
    if len(z) > len(p) or len(p) < 1:
        raise BaseException(
            "Incorrect specification. can't have more zeros than poles."
        )

    # Sort poles and zeros
    p = _sort_by_complex_descending(p)
    if len(z) > 0:
        z = _sort_by_complex_descending(z)

    k_per_state = np.float64(np.power(np.float64(np.abs(k)), 1.0 / np.float64(len(p))))

    index = 0
    systems = []
    while index < len(p):
        D = np.array([[0.0]])
        if index + 1 < len(p):
            # Two poles
            A = np.zeros((2, 2))
            B = k_per_state ** 2 * np.array([[1.0], [0.0]])
            CT = np.zeros((1, 2))
            D = np.array([[0.0]])

            pole_1, pole_2 = p[index], p[index + 1]
            # If complex conjugate pole pairs
            if np.allclose(pole_1, np.conjugate(pole_2)):
                a1, b1 = np.real(pole_1), np.imag(pole_1)
                a2, b2 = np.real(pole_2), np.imag(pole_2)
                A = np.array([[a1, b1], [b2, a2]])
            else:
                if np.imag(pole_1) != 0 or np.imag(pole_2) != 0:
                    raise BaseException("Can't have non-conjugate complex poles")
                A = np.array([[np.real(pole_1), 0], [1.0, np.real(pole_2)]])

            if index < len(z):
                zero_1 = z[index]
                if index + 1 < len(z):
                    # Two zeros left
                    zero_2 = z[index + 1]
                    y = np.array(
                        [
                            [zero_1 + zero_2 - A[0, 0] - A[1, 1]],
                            [zero_1 * zero_2 - A[0, 0] * A[1, 1] + A[0, 1] * A[1, 0]],
                        ]
                    )
                    if not np.allclose(np.imag(y), np.zeros(2)):
                        raise BaseException("Can't have non-conjugate complex zeros")
                    M = np.array([[-1.0, 0], [-A[1, 1], A[1, 0]]])
                    sol = np.linalg.solve(M, np.real(y))
                    D = k_per_state ** 2 * np.array([[1.0]])
                    CT = np.array([[sol[0, 0], sol[1, 0]]])
                else:
                    # Single zero
                    if np.imag(zero_1) != 0:
                        raise BaseException("Can't have non-conjugate complex zero")
                    c1 = 1.0
                    c2 = (A[1, 1] - np.real(zero_1)) / A[1, 0]
                    CT = np.array([[c1, c2]])
            else:
                # No zero
                #
                # gain
                CT = 1.0 / A[1, 0] * np.array([[0.0, 1.0]])

            index += 2
        else:
            # Only one pole and possibly zero left
            pole = p[index]
            if np.imag(pole) != 0:
                raise BaseException("Can't have non-conjugate complex poles")
            A = np.array([[np.real(pole)]])
            B = np.array([[k_per_state]])
            CT = np.array([[1.0]])
            D = np.array([[0.0]])
            if index < len(z):
                zero = z[index]
                if np.imag(zero) != 0:
                    raise BaseException("Cant have non-conjugate complex zeros")
                D[0, 0] = k_per_state
                CT[0, 0] = pole - np.real(zero)
            index += 1
        systems.append(AnalogSystem(A, B, CT, None, None, D))
    chained_system = chain(systems)
    return chained_system.A, chained_system.B, chained_system.CT, chained_system.D


def _sort_by_complex_descending(list: np.ndarray) -> np.ndarray:
    sorted_indexes = np.argsort(np.abs(np.imag(list)))[::-1]
    list = list[sorted_indexes]
    complex_indexes = np.imag(list) != 0
    number_of_complex_poles = np.sum(complex_indexes)
    if not _complex_conjugate_pairs(list[complex_indexes]):
        raise BaseException("Not complex conjugate pairs")
    sorted = np.zeros_like(list)
    sorted[:number_of_complex_poles] = list[complex_indexes]
    list[number_of_complex_poles:] = list[complex_indexes != True]
    return list


def _complex_conjugate_pairs(list: np.ndarray) -> bool:
    index = 0
    while index + 1 < len(list):
        if not np.allclose(list[index], list[index + 1].conjugate()):
            return False
        index += 2
    return True
