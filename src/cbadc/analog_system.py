"""Analog systems

This module provides a general :py:class:`cbadc.analog_system.AnalogSystem`
class with the necessary functionality to do transient simulations, compute
transfer functions, and exposing the relevant system parameters as
attributes. Additionally, several derived convenience classes are defined
to quickly initialize analog systems of particular structures.
"""
import numpy as np
import numpy.typing as npt


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
        self.Gamma = np.array(Gamma, dtype=np.double)
        self.Gamma_tildeT = np.array(Gamma_tildeT, dtype=np.double)

        self.N = self.A.shape[0]

        if self.A.shape[0] != self.A.shape[1]:
            raise InvalidAnalogSystemError(self, "system matrix not square")

        # ensure matrices
        if len(self.B.shape) == 1:
            self.B = self.B.reshape((self.N, 1))
        if len(self.CT.shape) == 1:
            self.CT = self.CT.reshape((1, self.N))

        self.M = self.Gamma.shape[1]
        self.M_tilde = self.Gamma_tildeT.shape[0]
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

        if self.Gamma.shape[0] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self, "N does not agree with control input matrix Gamma."
            )

        if self.Gamma_tildeT.shape[1] != self.A.shape[0]:
            raise InvalidAnalogSystemError(
                self,
                """N does not agree with control observation matrix
                Gamma_tilde.""",
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
        return np.dot(
            np.linalg.pinv(complex(0, _omega) * np.eye(self.N) - self.A),
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
        resp = np.einsum('ij,jkl', self.CT, result)
        return np.asarray(resp)

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
