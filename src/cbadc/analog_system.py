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
from typing import Tuple, List


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
    * The signal observation :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t)`

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

    def __init__(self, A: npt.ArrayLike, B: npt.ArrayLike, CT: npt.ArrayLike,
                 Gamma: npt.ArrayLike, Gamma_tildeT: npt.ArrayLike):
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

    def derivative(self, x: np.ndarray, t: float, u: np.ndarray,
                   s: np.ndarray) -> np.ndarray:
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
        # return np.dot(
        #     np.linalg.pinv(complex(0, _omega) *
        #                    np.eye(self.N) - self.A, rcond=1e-600),
        #     self.B,
        # )
        return np.dot(
            np.linalg.pinv(complex(0, _omega) *
                           np.eye(self.N) - self.A, rcond=1e-300),
            self.B,
        )

    def transfer_function_matrix(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate the analog signal transfer function at the angular
        frequencies of the omega array.

        Specifically, evaluates

        :math:`\mathbf{G}(\omega) = \mathbf{C}^\mathsf{T} \\left(\mathbf{A} - i \omega \mathbf{I}_N\\right)^{-1} \mathbf{B}`

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
        for index in range(size):
            result[:, :, index] = self._atf(omega[index])
        # resp = np.einsum('ij,jkl', self.CT, result)
        resp = np.tensordot(self.CT, result, axes=((1), (0)))
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
        return scipy.signal.ss2zpk(self.A, self.B, self.CT, np.zeros((self.N_tilde, self.L)), input)

    def __str__(self):
        return f"The analog system is parameterized as:\nA =\n{np.array(self.A)},\nB =\n{np.array(self.B)},\nCT = \n{np.array(self.CT)},\nGamma =\n{np.array(self.Gamma)},\nand Gamma_tildeT =\n{np.array(self.Gamma_tildeT)}"


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
    `chain-of-Integrator ADC <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf?sequence=1&isAllowed=y#page=96/>`_.


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
        """Create an chain-of-integrators analog system.
        """
        if (beta.shape[0] != beta.size):
            InvalidAnalogSystemError(
                self, "beta must be a one dimensional vector")
        if (rho.shape[0] != rho.size):
            InvalidAnalogSystemError(
                self, "rho must be a one dimensional vector")
        if (kappa.shape[0] != rho.size):
            InvalidAnalogSystemError(
                self, "kappa must be a one dimensional vector of size N or matrix with N rows")
        if(beta.size != rho.size and rho.size != kappa[:, 0].size):
            InvalidAnalogSystemError(
                self, "beta, rho, kappa vector must be of same size")

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
            Gamma_tildeT[row_index, :] = \
                Gamma_tildeT[row_index, :] / \
                np.linalg.norm(Gamma_tildeT[row_index, :])

        # initialize parent class
        AnalogSystem.__init__(self, A, B, CT, Gamma, Gamma_tildeT)


class LeapFrog(AnalogSystem):
    """Represents an leap-frog analog system.

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and creates a convenient
    way of creating leap-frog A/D analog systems. For more information about leap-frog ADCs see
    `Leap Frog ADC <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf?sequence=1&isAllowed=y#page=126/>`_.


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
        """Create an leap-frog analog system.
        """
        if (beta.shape[0] != beta.size):
            InvalidAnalogSystemError(
                self, "beta must be a one dimensional vector")
        if (rho.shape[0] != rho.size):
            InvalidAnalogSystemError(
                self, "rho must be a one dimensional vector")
        if (kappa.shape[0] != kappa.size):
            InvalidAnalogSystemError(
                self, "kappa must be a one dimensional vector")
        if(beta.size != rho.size and rho.size != kappa.size):
            InvalidAnalogSystemError(
                self, "beta, rho, kappa vector must be of same size")

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
            Gamma_tildeT[row_index, :] = \
                Gamma_tildeT[row_index, :] / \
                np.linalg.norm(Gamma_tildeT[row_index, :])

        # initialize parent class
        AnalogSystem.__init__(self, A, B, CT, Gamma, Gamma_tildeT)


class ButterWorth(AnalogSystem):
    """A Butterworth filter analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and creates a convenient
    way of creating Butterworth filter analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, and :math:`\mathbf{C}^\mathsf{T}` are determined using the
    :py:func:`scipy.signal.iirfilter`.

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
        """Create a Butterworth filter
        """
        # State space order
        self.Wn = Wn

        sos = scipy.signal.iirfilter(
            N, Wn, analog=True, btype='lowpass', ftype='butter', output='sos')
        print(sos)
        A, B, CT, _ = scipy.signal.zpk2ss(z, p, k)

        # initialize parent class
        AnalogSystem.__init__(self, A, B, CT, None, None)


class ChebyshevI(AnalogSystem):
    """A Chebyshev type I filter's analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and is a convenient
    way of creating Chebyshev type I filter's analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, and :math:`\mathbf{C}^\mathsf{T}` are determined using the
    :py:func:`scipy.signal.iirfilter`.

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
        """Create a Chebyshev type I filter
        """
        # State space order
        self.Wn = Wn
        self.rp = rp

        z, p, k = scipy.signal.iirfilter(
            N, Wn, rp, analog=True, btype='lowpass', ftype='cheby1', output='zpk')
        A, B, CT, _ = scipy.signal.zpk2ss(z, p, k)

        # initialize parent class
        AnalogSystem.__init__(self, A, B, CT, None, None)


class ChebyshevII(AnalogSystem):
    """A Chebyshev type II filter's analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and is a convenient
    way of creating Chebyshev type II filter's analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, and :math:`\mathbf{C}^\mathsf{T}` are determined using the
    :py:func:`scipy.signal.iirfilter`.

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
        """Create a Chebyshev type II filter
        """
        # State space order
        self.Wn = Wn
        self.rs = rs
        z, p, k = scipy.signal.iirfilter(
            N, Wn, rs=rs, analog=True, btype='lowpass', ftype='cheby2', output='zpk')
        A, B, CT, _ = scipy.signal.zpk2ss(z, p, k)

        # initialize parent class
        AnalogSystem.__init__(self, A, B, CT, None, None)


class Cauer(AnalogSystem):
    """A Cauer (elliptic) filter's analog system

    This class inherits from :py:class:`cbadc.analog_system.AnalogSystem` and is a convenient
    way of creating Cauer filter's analog system representation.

    Specifically, we specify the filter by the differential equations

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t)`

    :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)`


    where

    internally :math:`\mathbf{A}` :math:`\mathbf{B}`, and :math:`\mathbf{C}^\mathsf{T}` are determined using the
    :py:func:`scipy.signal.iirfilter`.

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
        """Create a Cauer filter
        """
        # State space order
        self.Wn = Wn
        self.rp = rp
        self.rs = rs

        z, p, k = scipy.signal.iirfilter(
            N, Wn, rp, rs, analog=True, btype='lowpass', ftype='ellip', output='zpk')
        A, B, CT, _ = scipy.signal.zpk2ss(z, p, k)

        # initialize parent class
        AnalogSystem.__init__(self, A, B, CT, None, None)


def abcd2abc(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a A,B,C,D system into a A,B,C system.

    Specifically, for a state space model as

    :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t)`

    :math:`\mathbf{y}(t) = \mathbf{C} \mathbf{x}(t) + \mathbf{D} \mathbf{u}(t)`

    is reformulated as    

    :math:`\dot{\\tilde{\mathbf{x}}}(t) = \\tilde{\mathbf{A}} \\tilde{\mathbf{x}}(t) + \\tilde{\mathbf{B}} \mathbf{u}(t)`

    :math:`\mathbf{y}(t) = \\tilde{\mathbf{C}} \\tilde{\mathbf{x}}(t)`

    where

    :math:`\\tilde{\mathbf{A}} = \\begin{pmatrix} \mathbf{0}_{L \\times L} & \mathbf{0}_{L \\times N} \\\ \mathbf{B} & \mathbf{A} \\end{pmatrix} \in \mathbb{R}^{(L + N)\\times(L + N)}`

    :math:`\\tilde{\mathbf{B}} = \\begin{pmatrix} \mathbf{I}_L \\\ \mathbf{0}_{N \\times L} \\end{pmatrix} \in \mathbb{R}^{(N + L)\\times L}`

    :math:`\\tilde{\mathbf{C}} = \\begin{pmatrix} \mathbf{D} & \mathbf{C} \\end{pmatrix} \in \mathbb{R}^{(L + \\tilde{N}) \\times (L + N)}`

    and :math:`\mathbf{A}\in \mathbb{R}^{N \\times N}`, :math:`\mathbf{B}\in \mathbb{R}^{N \\times L}`, 
    :math:`\mathbf{C} \in \mathbb{R}^{\\tilde{N} \\times N}`, and :math:`\mathbf{D} \in \mathbb{R}^{\\tilde{N} \\times L}`.
    """
    A_tilde = np.zeros((A.shape[0] + B.shape[1], A.shape[1] + B.shape[1]))
    B_tilde = np.zeros((A.shape[0] + B.shape[1], B.shape[1]))
    C_tilde = np.zeros((C.shape[0], A.shape[0] + B.shape[1]))

    A_tilde[B.shape[1]:, B.shape[1]:] = A
    A_tilde[B.shape[1]:, :B.shape[1]] = B

    B_tilde[:B.shape[1], :B.shape[1]] = np.eye(B.shape[1])

    C_tilde[:, D.shape[1]:] = C
    C_tilde[:, :D.shape[1]] = D

    return A_tilde, B_tilde, C_tilde


def chain(analog_systems: List[AnalogSystem]) -> AnalogSystem:
    """Construct an analog system by chaining several analog systems.

    The chaining is achieved by constructing

    :math:`\mathbf{A} = \\begin{pmatrix}\ddots & \ddots \\\ & \mathbf{B}_\ell \mathbf{C}^\mathsf{T}_{\ell - 1} & \mathbf{A}_\ell \\\ & & \mathbf{B}_{\ell + 1} \mathbf{C}^\mathsf{T}_{\ell} & \mathbf{A}_{\ell + 1} \\\ & & & \ddots & \ddots \\end{pmatrix}`

    :math:`\mathbf{B} = \\begin{pmatrix} \mathbf{B}_1 \\\ \mathbf{0}_{(N - L) \\times L} \\end{pmatrix}`

    :math:`\mathbf{C}^\mathsf{T} = \\begin{pmatrix} \mathbf{0}_{(\\tilde{N}) \\times (N - \\tilde{N})} & \mathbf{C}^\mathsf{T}_{-1} \\end{pmatrix}`

    :math:`\mathbf{\Gamma} = \\begin{pmatrix} \ddots \\\ & \mathbf{\Gamma}_\ell  \\\ &  & \mathbf{\Gamma}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix}`

    :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T} = \\begin{pmatrix} \ddots \\\ & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_\ell  \\\ &  & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix}`

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
    N: int = np.sum(
        np.array([analog_system.N for analog_system in analog_systems]))
    M: int = np.sum(
        np.array([analog_system.M for analog_system in analog_systems]))
    M_tilde: int = np.sum(
        np.array([analog_system.M_tilde for analog_system in analog_systems]))
    L: int = analog_systems[0].L
    N_tilde: int = analog_systems[-1].N_tilde

    A = np.zeros((N, N))
    B = np.zeros((N, L))
    CT = np.zeros((N_tilde, N))
    Gamma = np.zeros((N, M))
    Gamma_tilde = np.zeros((M_tilde, N))
    previous_system = analog_systems[0]

    B[:analog_systems[0].B.shape[0],
        :analog_systems[0].B.shape[1]] = analog_systems[0].B

    CT[:, -analog_systems[-1].N:] = analog_systems[-1].CT

    n: int = 0
    m: int = 0
    m_tilde: int = 0
    l: int = 0
    for analog_system in analog_systems:
        n_end = n + analog_system.N
        m_end = m + analog_system.M
        m_tilde_end = m_tilde + analog_system.M_tilde

        A[n: n_end, n:n_end] = analog_system.A

        # From the second system and on connect input to output.
        if n > 0:
            if analog_system.B.shape[1] != previous_system.CT.shape[0]:
                raise BaseException(
                    f"System {previous_system} and {analog_system} don't have compatiable input output dimension")
            A[n: n_end, l:n] = np.dot(analog_system.B, previous_system.CT)

        Gamma[n:n_end, m:m_end] = analog_system.Gamma

        Gamma_tilde[m_tilde:m_tilde_end, n:n_end] = analog_system.Gamma_tildeT

        previous_system = analog_system
        l = n
        n += analog_system.N
        m += analog_system.M
        m_tilde += analog_system.M_tilde
    return AnalogSystem(A, B, CT, Gamma, Gamma_tilde)


def stack(analog_systems: List[AnalogSystem]) -> AnalogSystem:
    """Construct an analog system by stacking several analog systems in parallel.

    The parallel stack is achieved by constructing

    :math:`\mathbf{A} = \\begin{pmatrix}\ddots \\\ & \mathbf{A}_\ell \\\ & &  \mathbf{A}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{N \\times N}`

    :math:`\mathbf{B} = \\begin{pmatrix} \ddots \\\ & \mathbf{B}_\ell \\\ & &  \mathbf{B}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{N \\times L}`

    :math:`\mathbf{C}^\mathsf{T} = \\begin{pmatrix} \ddots \\\ & \mathbf{C}_\ell \\\ & &  \mathbf{C}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{\\tilde{N} \\times N}`

    :math:`\mathbf{\Gamma} = \\begin{pmatrix} \ddots \\\ & \mathbf{\Gamma}_\ell  \\\ &  & \mathbf{\Gamma}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{N \\times M}`

    :math:`\\tilde{\mathbf{\Gamma}}^\mathsf{T} = \\begin{pmatrix} \ddots \\\ & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_\ell  \\\ &  & \\tilde{\mathbf{\Gamma}}^\mathsf{T}_{\ell + 1} \\\ & & & \ddots \\end{pmatrix} \in \mathbb{R}^{\\tilde{M} \\times N}`

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
    N: int = np.sum(
        np.array([analog_system.N for analog_system in analog_systems]))
    M: int = np.sum(
        np.array([analog_system.M for analog_system in analog_systems]))
    M_tilde: int = np.sum(
        np.array([analog_system.M_tilde for analog_system in analog_systems]))
    L: int = np.sum(
        np.array([analog_system.L for analog_system in analog_systems]))
    N_tilde: int = np.sum(
        np.array([analog_system.N_tilde for analog_system in analog_systems]))

    A = np.zeros((N, N))
    B = np.zeros((N, L))
    CT = np.zeros((N_tilde, N))
    Gamma = np.zeros((N, M))
    Gamma_tilde = np.zeros((M_tilde, N))

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

        A[n: n_end, n:n_end] = analog_system.A
        B[n: n_end, l:l_end] = analog_system.B
        CT[n_tilde:n_tilde_end, n:n_end] = analog_system.CT
        Gamma[n:n_end, m:m_end] = analog_system.Gamma
        Gamma_tilde[m_tilde:m_tilde_end, n:n_end] = analog_system.Gamma_tildeT

        l += analog_system.L
        n += analog_system.N
        n_tilde += analog_system.N_tilde
        m += analog_system.M
        m_tilde += analog_system.M_tilde
    return AnalogSystem(A, B, CT, Gamma, Gamma_tilde)
