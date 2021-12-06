import numpy as np
from .analog_system import AnalogSystem, InvalidAnalogSystemError
from ..fom import enob_to_snr, snr_from_dB


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
