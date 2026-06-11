"""Tests for :mod:`cbadc.noise` and noise propagation through discretisation."""

import numpy as np
import pytest

import cbadc
from cbadc import AnalogFrontend, noise

N = 4
ENOB = 12
BW = 1e5
xi = 1


def test_device_densities_positive():
    assert noise.resistor_density(1e3) > 0
    assert noise.ota_density(1e-3) > 0
    assert noise.kTC_rms(1e-12) > 0


def test_combine_densities_rss():
    a, b = 3.0, 4.0
    assert noise.combine_densities(a, b) == pytest.approx(5.0)


def test_per_state_intensity_is_diag_of_squares():
    d = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(noise.per_state_intensity(d), np.diag(d**2))


def test_discrete_process_noise_cov_zero_dynamics():
    # With A = 0 the van-Loan integral reduces to Q_ct * dt.
    Q = np.diag([1.0, 2.0])
    dt = 0.5
    Qd = noise.discrete_process_noise_cov(np.zeros((2, 2)), Q, dt)
    np.testing.assert_allclose(Qd, Q * dt, atol=1e-12)


def test_discrete_process_noise_cov_symmetric():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    M = rng.normal(size=(3, 3))
    Q = M @ M.T
    Qd = noise.discrete_process_noise_cov(A, Q, 1e-3)
    np.testing.assert_allclose(Qd, Qd.T, atol=1e-15)


def test_psd_factor_full_rank_matches_cholesky():
    rng = np.random.default_rng(1)
    M = rng.normal(size=(4, 4))
    cov = M @ M.T + np.eye(4)
    L = noise.psd_factor(cov)
    np.testing.assert_allclose(L @ L.T, cov, atol=1e-10)
    # full-rank positive-definite input keeps the lower-triangular Cholesky
    np.testing.assert_allclose(L, np.linalg.cholesky(cov))


def test_psd_factor_rank_deficient_does_not_raise():
    # A single-source (rank-1) covariance breaks strict Cholesky.
    v = np.array([1.0, 2.0, 0.5, 0.0]).reshape((-1, 1))
    cov = v @ v.T
    L = noise.psd_factor(cov)
    # the null direction is floored to a negligible positive eigenvalue, so the
    # reconstruction matches only to that floor (~max_eig * 1e-12), not exactly.
    np.testing.assert_allclose(L @ L.T, cov, atol=1e-9)


def test_state_noise_is_discretisation_order_invariant():
    """Setting noise before vs. after ``discretize`` must agree.

    This is the regression guard for the historical bug where ``discretize``
    propagated the continuous-time intensity straight onto the discrete
    frontend, skipping the van-Loan integral.
    """
    af, _ = AnalogFrontend.chain_of_integrators(N=N, ENOB=ENOB, BW=BW, xi=xi)
    dt = af.dt
    Q = np.diag(np.linspace(1e-6, 4e-6, N))

    af.state_covariance = Q  # set on the continuous-time frontend
    cov_before = af._state_cov_cholesky @ af._state_cov_cholesky.T

    af2, _ = AnalogFrontend.chain_of_integrators(N=N, ENOB=ENOB, BW=BW, xi=xi)
    af2.state_covariance = Q
    af2d = af2.discretize(dt)  # discretise (propagates the noise)
    cov_after = af2d._state_cov_cholesky @ af2d._state_cov_cholesky.T

    np.testing.assert_allclose(cov_before, cov_after, atol=1e-18)


def test_input_referred_noise_factorises():
    """Rank-deficient input-referred intensity must not break the setter."""
    af, _ = AnalogFrontend.chain_of_integrators(N=N, ENOB=ENOB, BW=BW, xi=xi)
    b_u = af.B[0][:, : af.L]
    Q_in = (b_u @ b_u.T) * 1e-9  # rank-1 continuous-time intensity
    af.state_covariance = Q_in  # would have raised LinAlgError before the fix
    L = af._state_cov_cholesky
    expected = noise.discrete_process_noise_cov(af.A[0], Q_in, af.dt)
    np.testing.assert_allclose(L @ L.T, expected, atol=1e-15)
