"""The leap-frog analog filter."""

import numpy as np
from scipy.signal import StateSpace


class LeapFrog(StateSpace):
    """A leap-frog analog filter.


    Parameters
    ----------
    beta : `array_like`, shape=(N,)
        vector with per integrator signal amplification :math:`\\begin{pmatrix}\\beta_1 & \cdots & \\beta_N \\end{pmatrix}`.
    alpha : `array_like`, shape=(N-1,)
        feedback factor vector :math:`\\begin{pmatrix}\\alpha_1 & \cdots & \\alpha_{N-1} \\end{pmatrix}`.
    rho : `array_like`, shape=(N,)
        local feedback factor vector :math:`\\begin{pmatrix}\\rho_1 & \cdots & \\rho_N \\end{pmatrix}`.
    kappa : `array_like`, shape=(N,)
        control gain vector :math:`\\begin{pmatrix}\\kappa_1 & \cdots & \\kappa_N \\end{pmatrix}`.

    See also
    --------
    :py:class:`cbadc.analog_filter.ChainOfIntegrators`

    """

    def __init__(
        self, beta: np.ndarray, alpha: np.ndarray, rho: np.ndarray, kappa: np.ndarray
    ):
        """Create an leap-frog analog system."""
        if beta.shape[0] != beta.size:
            raise ValueError("beta must be a one dimensional vector")
        if alpha.shape[0] != alpha.size:
            raise ValueError("alpha must be a one dimensional vector")
        if rho.shape[0] != rho.size:
            raise ValueError("rho must be a one dimensional vector")
        if beta.size != rho.size and rho.size != kappa.size:
            raise ValueError("beta, rho, kappa vector must be of same size")

        # State space order
        N = beta.size
        M = N

        # Analog system parameters
        A = np.diag(alpha, k=1) + np.diag(beta[1:], k=-1) + np.diag(rho)
        B = np.zeros((N, 1 + N))
        B[0, 0] = beta[0]
        C = np.eye(M)
        D = np.zeros((M, N + 1))
        # Check if Kappa is specified as a vector
        if kappa.shape[1] == 1:
            B[:, 1:] = np.diag(kappa.flatten())
        else:
            B[:, 1:] = np.array(kappa, dtype=np.double)

        # initialize parent class
        super().__init__(A, B, C, D)
