import cbadc
import numpy as np
import cbadc.digital_estimator._filter_coefficients
import pytest

eta2 = 1e3
N = 4
A = np.eye(N, k=1) * 1e-2 + np.eye(N, k=-1) * 1e3
B = np.random.randn(N, 1)
B[0, 0] = 1e2
CT = np.zeros((1, N))
CT[-1] = 1.0


A = A.transpose()
Q = np.outer(B, B)
B = CT.transpose()
R = eta2 * np.eye(1)


def test_care():
    cbadc.digital_estimator._filter_coefficients.care(A, B, Q, R)
    cbadc.digital_estimator._filter_coefficients.care(-A, B, Q, R)


def test_analytical_and_scipy_care():
    Vf = cbadc.digital_estimator._filter_coefficients.care(A, B, Q, R)
    Vb = cbadc.digital_estimator._filter_coefficients.care(-A, B, Q, R)

    Vf2 = cbadc.digital_estimator._filter_coefficients._scipy_care(A, B, Q, R)
    Vb2 = cbadc.digital_estimator._filter_coefficients._scipy_care(-A, B, Q, R)

    print(Vf, Vf2)
    np.testing.assert_allclose(Vf, Vf2)
    print(Vb, Vb2)
    np.testing.assert_allclose(Vb, Vb2)


def test_analytical_and_numpy_care():
    Vf = cbadc.digital_estimator._filter_coefficients.care(A, B, Q, R)
    Vb = cbadc.digital_estimator._filter_coefficients.care(-A, B, Q, R)

    Vf2 = cbadc.digital_estimator._filter_coefficients._numpy_care(A, B, Q, R)
    Vb2 = cbadc.digital_estimator._filter_coefficients._numpy_care(-A, B, Q, R)

    print(Vf, Vf2)
    np.testing.assert_allclose(Vf, Vf2)
    print(Vb, Vb2)
    np.testing.assert_allclose(Vb, Vb2)


@pytest.mark.skip("Currently not working")
def test_analytical_and_mpmath_care():
    Vf = cbadc.digital_estimator._filter_coefficients.care(A, B, Q, R)
    Vb = cbadc.digital_estimator._filter_coefficients.care(-A, B, Q, R)

    Vf2 = cbadc.digital_estimator._filter_coefficients._mpmath_care(A, B, Q, R)
    Vb2 = cbadc.digital_estimator._filter_coefficients._mpmath_care(-A, B, Q, R)

    print(Vf, Vf2)
    np.testing.assert_allclose(Vf, Vf2)
    print(Vb, Vb2)
    np.testing.assert_allclose(Vb, Vb2)


def test_brute_force_care():
    Vf = cbadc.digital_estimator._filter_coefficients.care(A, B, Q, R)
    Vb = cbadc.digital_estimator._filter_coefficients.care(-A, B, Q, R)

    Vf2 = cbadc.digital_estimator._filter_coefficients.bruteForceCare(
        A, B, Q, R, tau=1e-12
    )
    Vb2 = cbadc.digital_estimator._filter_coefficients.bruteForceCare(
        -A, B, Q, R, tau=1e-12
    )

    print(Vf, Vf2)
    np.testing.assert_allclose(Vf, Vf2)
    print(Vb, Vb2)
    np.testing.assert_allclose(Vb, Vb2)
