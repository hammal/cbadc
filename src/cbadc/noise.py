"""Circuit noise densities and covariance calculations.

Density helpers use one-sided amplitudes; state-noise intensities and their
continuous-time integration use two-sided covariance conventions.
"""

from __future__ import annotations

import numpy as np
from scipy import constants
from scipy.linalg import expm, matrix_balance
from scipy.signal import fftconvolve

k_B = constants.Boltzmann


def flicker_taps(size: int, alpha: float) -> np.ndarray:
    """Kasdin FIR taps with low-frequency power spectrum proportional to ``1/f**alpha``.

    Parameters
    ----------
    size : int
        Number of noise samples in the run.
    alpha : float
        Spectral exponent in ``(0, 2)``.

    Returns
    -------
    ndarray of float, shape (size,)
        Coefficients of ``(1 - z**-1)**(-alpha/2)``.
    """
    taps = np.ones(size)
    if size > 1:
        k = np.arange(1, size)
        taps[1:] = np.cumprod((k - 1 + alpha / 2) / k)
    return taps


def flicker_samples(
    taps: np.ndarray,
    intensity: float,
    f_corner: float,
    dt: float,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Filter white samples to match ``Q * (f_corner/f)**alpha`` at low frequencies.

    Parameters
    ----------
    taps : ndarray of float, shape (size,)
        Kasdin filter from :func:`flicker_taps`.
    intensity : float
        Two-sided white state-noise intensity ``Q`` in state units squared/s.
    f_corner : float
        Frequency in Hz where flicker and white noise have equal density.
    dt : float
        Noise-sample interval in seconds.
    alpha : float
        Spectral exponent in ``(0, 2)``.
    rng : numpy.random.Generator
        Source of independent standard-normal samples.

    Returns
    -------
    ndarray of float, shape (size,)
        Held state-derivative noise, one value per clock interval.
    """
    if not taps.size:
        return np.empty(0)
    scale = np.sqrt(intensity * (2 * np.pi * f_corner) ** alpha * dt ** (alpha - 1))
    return scale * fftconvolve(rng.normal(size=taps.size), taps)[: taps.size]


def resistor_density(R: float, Temp: float = 300.0) -> float:
    """One-sided thermal noise density ``sqrt(4 k T R)`` [V/sqrt(Hz)].

    ``R`` is a positive scalar resistance in ohms; ``Temp`` is a scalar
    absolute temperature in kelvin. Returns a scalar voltage density.

    Parameters
    ----------
    R : float
        Positive resistance in ohms.
    Temp : float, optional
        Absolute temperature in kelvin.
        Defaults to ``300.0``.

    >>> round(float(resistor_density(1e3)) * 1e9, 2)  # nV/sqrt(Hz)
    4.07

    Returns
    -------
    float
        One-sided voltage-noise density in V/√Hz.
    """
    return np.sqrt(4 * k_B * Temp * R)


def ota_density(gm: float, Temp: float = 300.0, gamma: float = 1.5) -> float:
    """Input-referred OTA noise density ``sqrt(4 k T gamma / gm)`` [V/sqrt(Hz)].

    ``gm`` is positive scalar transconductance in siemens, ``Temp`` is
    scalar absolute temperature in kelvin, and ``gamma`` is a dimensionless
    scalar excess-noise factor. Returns a scalar voltage density.

    Parameters
    ----------
    gm : float
        Positive transconductance in siemens.
    Temp : float, optional
        Absolute temperature in kelvin.
        Defaults to ``300.0``.
    gamma : float, optional
        Dimensionless excess-noise factor.
        Defaults to ``1.5``.

    Returns
    -------
    float
        One-sided input-referred voltage-noise density in V/√Hz.
    """
    return np.sqrt(4 * k_B * Temp * gamma / gm)


def kTC_rms(C: float, Temp: float = 300.0) -> float:
    """Sampled ``kT/C`` noise ``sqrt(k T / C)`` [V rms].

    ``C`` is a positive scalar capacitance in farads and ``Temp`` a scalar
    absolute temperature in kelvin. Returns scalar rms voltage.

    Parameters
    ----------
    C : float
        Positive capacitance in farads.
    Temp : float, optional
        Absolute temperature in kelvin.
        Defaults to ``300.0``.

    >>> round(float(kTC_rms(1e-12)) * 1e6, 1)  # uV rms on 1 pF
    64.4

    Returns
    -------
    float
        Sampled root-mean-square voltage in volts.
    """
    return np.sqrt(k_B * Temp / C)


def combine_densities(*densities) -> np.ndarray:
    """Root-sum-square of independent noise densities.

    ``densities`` are scalar or equally shaped float arrays of one-sided
    densities (for example V/sqrt(Hz)); the returned array has their
    broadcast shape and the same units.

    Parameters
    ----------
    *densities : float or array_like of float
        Independent one-sided densities with compatible shapes and units.

    Returns
    -------
    ndarray of float, shape of each input density
        Root-sum-square density in the input units.
    """
    return np.sqrt(np.sum(np.square(np.asarray(densities, dtype=float)), axis=0))


def per_state_intensity(densities) -> np.ndarray:
    """Two-sided state-noise intensity ``diag(d**2 / 2)`` from one-sided densities ``d``.

    ``densities`` is a float sequence or array of shape ``(N,)`` in state
    units/sqrt(Hz); returns the diagonal covariance intensity ``(N, N)``.

    Parameters
    ----------
    densities : array_like of float, shape (N,)
        One-sided noise amplitudes in state units/√Hz.

    ``cov(dW) = Q dt`` is a two-sided spectral density, a measured density is
    one-sided; the factor of two is what makes an RC settle to ``kT/C``.

    >>> per_state_intensity([1e-9, 2e-9]).diagonal().tolist()
    [5e-19, 2e-18]

    Returns
    -------
    ndarray of float, shape (N, N)
        Diagonal two-sided state-noise intensity.
    """
    d = np.asarray(densities, dtype=float).ravel()
    return np.diag(d**2 / 2.0)


def _van_loan(A: np.ndarray, Q: np.ndarray, dt: float) -> np.ndarray:
    """Integrate continuous state-noise intensity over one step.

    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Continuous-time state matrix.
    Q : ndarray of float, shape (N, N)
        Symmetric state-noise intensity.
    dt : float
        Integration period in seconds.

    Returns
    -------
    ndarray of float, shape (N, N)
        Symmetric per-step process-noise covariance.
    """
    N = A.shape[0]
    E = expm(np.block([[-A, Q], [np.zeros((N, N)), A.T]]) * dt)
    Qd = E[N:, N:].T @ E[:N, N:]
    return 0.5 * (Qd + Qd.T)


def discrete_process_noise_cov(
    A: np.ndarray, Q: np.ndarray, dt: float, target_norm: float = 1.0
) -> np.ndarray:
    """Per-step covariance ``int_0^dt e^{A s} Q e^{A^T s} ds`` of a continuous intensity ``Q``.

    ``A`` is a float state matrix ``(N, N)`` in 1/s, ``Q`` a symmetric float
    intensity ``(N, N)`` in state units squared per second, ``dt`` a scalar
    period in seconds, and ``target_norm`` a positive dimensionless scalar
    threshold for subdivision. Returns covariance ``(N, N)``.

    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Continuous-time state matrix in s⁻¹.
    Q : ndarray of float, shape (N, N)
        Symmetric two-sided noise intensity in state units squared per second.
    dt : float
        Positive integration period in seconds.
    target_norm : float, optional
        Positive threshold for substepping the matrix exponential.
        Defaults to ``1.0``.

    Van Loan's augmented exponential is exact but contains ``expm(-A dt)``, which
    overflows for stiff ``A``. Above ``target_norm`` the matrix is balanced and
    the integral is computed over ``dt / 2**s`` and doubled ``s`` times.

    Returns
    -------
    ndarray of float, shape (N, N)
        Symmetric per-step process-noise covariance.
    """
    if np.linalg.norm(A * dt, 2) <= target_norm:
        return _van_loan(A, Q, dt)
    A_b, Dg = matrix_balance(A, permute=False)
    scale = np.outer(np.diag(Dg), np.diag(Dg))
    norm = np.linalg.norm(A_b * dt, 2)
    s = 0 if norm <= target_norm else int(np.ceil(np.log2(norm / target_norm)))
    h = dt / 2**s
    Qd, Ah = _van_loan(A_b, Q / scale, h), expm(A_b * h)
    for _ in range(s):
        Qd = Qd + Ah @ Qd @ Ah.T
        Ah = Ah @ Ah
    Qd = Qd * scale
    return 0.5 * (Qd + Qd.T)


def positive_semidefinite(cov: np.ndarray) -> np.ndarray:
    """Factor a covariance as ``F F^T = cov``; use eigenvectors if Cholesky fails.

    ``cov`` is a symmetric positive-semidefinite float array ``(N, N)``;
    returns a float factor ``(N, N)``.

    Parameters
    ----------
    cov : ndarray of float, shape (N, N)
        Symmetric positive-semidefinite covariance matrix.

    A single source leaves ``cov`` rank-deficient and the van Loan integral
    can leave round-off negatives, which Cholesky rejects; the eigen path
    zeroes those, so a null direction gets no noise.

    >>> cov = np.outer([1.0, 2.0], [1.0, 2.0])  # rank one
    >>> F = positive_semidefinite(cov)
    >>> np.allclose(F @ F.T, cov)
    True

    Returns
    -------
    ndarray of float, shape (N, N)
        Factor whose product with its transpose reconstructs ``cov``.
    """
    cov = 0.5 * (cov + cov.T)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(cov)
        return V * np.sqrt(np.clip(w, 0.0, None))
