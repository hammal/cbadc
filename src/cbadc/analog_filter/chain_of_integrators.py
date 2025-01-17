"""The chain-of-integrators analog filter."""

import numpy as np
from scipy.signal import StateSpace


class ChainOfIntegrators(StateSpace):
    """A chain-of-integrators analog filter.

    Parameters
    ----------
    beta : `array_like`, shape=(N,)
        vector with per integrator signal amplification.
    rho : `array_like`, shape=(N,)
        local feedback factor vector.
    kappa : `array_like`, shape=(N,) or shape=(M, N),
        control gain vector.

    """

    def __init__(self, beta: np.ndarray, rho: np.ndarray, kappa: np.ndarray):
        """Create an chain-of-integrators analog system."""
        if len(beta.shape) > 1:
            raise ValueError("beta must be a one dimensional vector")
        if len(rho.shape) > 1:
            raise ValueError("rho must be a one dimensional vector")
        if kappa.shape[0] != rho.size:
            raise ValueError(
                "kappa must be a one dimensional vector of size N or matrix with N rows"
            )
        if beta.size != rho.size and rho.size != kappa[:, 0].size:
            raise ValueError("beta, rho, kappa vector must be of same size")

        N = beta.size
        A = np.diag(rho) + np.diag(beta[1:], k=-1)
        B = np.zeros((N, 1 + N), dtype=float)
        B[0, 0] = beta[0]
        # Check if Kappa is specified as a vector
        if kappa.shape[1] == 1:
            B[:, 1:] = np.diag(kappa.flatten())
        else:
            B[:, 1:] = np.array(kappa, dtype=float)
        C = -np.sign(B[0, 1]) * np.eye(N)
        D = np.zeros((N, N + 1), dtype=float)
        super().__init__(A, B, C, D)
