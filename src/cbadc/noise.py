"""Thermal-noise helpers for control-bounded converters.

This module bridges a *circuit-level noise specification* (device thermal noise,
input-referred spectral density) and the continuous-time state-noise intensity
that :class:`cbadc.analog_frontend.AnalogFrontend` injects during simulation.

Two distinct conventions are supported, and keeping them apart is the whole point:

``per-state`` / integrator-output-referred
    Each integrator node has its own noise density.  The continuous-time
    intensity is ``Q = diag(densities**2)``.  This is what
    :attr:`cbadc.analog_frontend.GmC.v_n` implements.

``input-referred``
    Noise is specified as a power spectral density ``S_in`` [V**2/Hz] referred
    to the converter *input*.  The equivalent state intensity is
    ``Q = b_u S_in b_u^T`` where ``b_u`` is the input column of ``B``, so the
    injected noise is statistically identical to driving the input port.  See
    :meth:`cbadc.analog_frontend.AnalogFrontend.input_referred_covariance_matrix`.

Either way, the continuous-time intensity ``Q`` must be discretised with the
van-Loan integral (:func:`discrete_process_noise_cov`) to obtain the per-step
covariance the simulator actually samples -- doing this consistently is what
makes the noise magnitude independent of *when* you discretise.
"""

import numpy as np
from scipy import constants
from scipy.linalg import expm

k_B = constants.Boltzmann

__all__ = [
    "resistor_density",
    "ota_density",
    "kTC_rms",
    "combine_densities",
    "per_state_intensity",
    "discrete_process_noise_cov",
    "psd_factor",
]


# ---------------------------------------------------------------------------
# Device noise densities -> V/sqrt(Hz)
# ---------------------------------------------------------------------------
def resistor_density(R: float, Temp: float = 300.0) -> float:
    """Thermal (Johnson) noise voltage density of a resistor.

    Parameters
    ----------
    R : float
        resistance in Ohm.
    Temp : float, optional
        temperature in Kelvin, defaults to 300 K.

    Returns
    -------
    float
        noise voltage density ``sqrt(4 k_B T R)`` in V/sqrt(Hz).
    """
    return np.sqrt(4 * k_B * Temp * R)


def ota_density(gm: float, Temp: float = 300.0, gamma: float = 1.5) -> float:
    """Input-referred noise voltage density of a transconductor.

    Parameters
    ----------
    gm : float
        transconductance in Siemens.
    Temp : float, optional
        temperature in Kelvin, defaults to 300 K.
    gamma : float, optional
        excess-noise factor, defaults to 1.5.

    Returns
    -------
    float
        ``sqrt(4 k_B T gamma / gm)`` in V/sqrt(Hz).
    """
    return np.sqrt(4 * k_B * Temp * gamma / gm)


def kTC_rms(C: float, Temp: float = 300.0) -> float:
    """Total (broadband) sampled ``kT/C`` noise in V rms.

    Parameters
    ----------
    C : float
        capacitance in Farad.
    Temp : float, optional
        temperature in Kelvin, defaults to 300 K.

    Returns
    -------
    float
        ``sqrt(k_B T / C)`` in V rms.
    """
    return np.sqrt(k_B * Temp / C)


def combine_densities(*densities) -> np.ndarray:
    """Root-sum-square combine independent noise densities [V/sqrt(Hz)]."""
    return np.sqrt(np.sum(np.square(np.asarray(densities, dtype=float)), axis=0))


# ---------------------------------------------------------------------------
# Continuous-time intensity construction
# ---------------------------------------------------------------------------
def per_state_intensity(densities: np.ndarray) -> np.ndarray:
    """Continuous-time state-noise intensity from per-state densities.

    Use this for noise referred to each integrator's own node -- the semantics
    :attr:`cbadc.analog_frontend.GmC.v_n` implements.

    Parameters
    ----------
    densities : numpy.ndarray, shape=(N,)
        per-state noise densities [V/sqrt(Hz)].

    Returns
    -------
    numpy.ndarray, shape=(N, N)
        ``diag(densities**2)``.
    """
    d = np.asarray(densities, dtype=float).flatten()
    return np.diag(d**2)


# ---------------------------------------------------------------------------
# Continuous-time -> discrete-time process-noise covariance (van Loan)
# ---------------------------------------------------------------------------
def discrete_process_noise_cov(
    A: np.ndarray, Q_ct: np.ndarray, dt: float
) -> np.ndarray:
    """Discrete per-step process-noise covariance via the van-Loan method.

    For the stochastic differential equation ``dx = A x dt + dW`` with
    ``cov(dW) = Q_ct dt``, the covariance accumulated over one step ``dt`` is

    .. math::

        Q_d = \\int_0^{dt} e^{A\\tau} Q_{ct} e^{A^T\\tau} \\, d\\tau

    evaluated by exponentiating the augmented ``2N x 2N`` matrix.

    Parameters
    ----------
    A : numpy.ndarray, shape=(N, N)
        continuous-time state matrix.
    Q_ct : numpy.ndarray, shape=(N, N)
        continuous-time noise intensity.
    dt : float
        step length.

    Returns
    -------
    numpy.ndarray, shape=(N, N)
        the symmetrised discrete-time per-step covariance ``Q_d``.
    """
    A = np.asarray(A, dtype=float)
    Q_ct = np.asarray(Q_ct, dtype=float)
    n = A.shape[0]
    M = np.vstack(
        (
            np.hstack((-A, Q_ct)),
            np.hstack((np.zeros((n, n)), A.T)),
        )
    )
    E = expm(M * dt)
    Qd = E[n:, n:].T @ E[:n, n:]
    return 0.5 * (Qd + Qd.T)  # symmetrise against round-off


def psd_factor(cov: np.ndarray, rel_floor: float = 1e-12) -> np.ndarray:
    """Return a factor ``L`` such that ``L @ L.T`` reproduces ``cov``.

    A strict Cholesky factorisation is tried first (fast, and preserves the
    historical lower-triangular factor for full-rank positive-definite inputs).
    Single-source (e.g. input-referred) noise yields a *rank-deficient* ``cov``
    and van-Loan round-off can leave tiny negative eigenvalues, both of which
    break Cholesky; in that case fall back to an eigen-decomposition with the
    spectrum floored to a negligible positive fraction of the largest
    eigenvalue.

    Parameters
    ----------
    cov : numpy.ndarray, shape=(N, N)
        a symmetric positive-semidefinite covariance matrix.
    rel_floor : float, optional
        eigenvalues below ``rel_floor * max_eigenvalue`` are clipped, defaults
        to 1e-12.

    Returns
    -------
    numpy.ndarray, shape=(N, N)
        a factor ``L`` with ``L @ L.T == cov`` (to floating-point tolerance).
    """
    cov = 0.5 * (cov + cov.T)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, w.max() * rel_floor, None)
        return V * np.sqrt(w)
