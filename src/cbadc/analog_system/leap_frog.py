import numpy as np
from .analog_system import AnalogSystem, InvalidAnalogSystemError


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
    >>> system = LeapFrog(beta, rho, kappa)

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
