import cbadc
import numpy as np

eta2 = 1e3
N = 15
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
    cbadc.digital_estimator.care(A, B, Q, R)
    cbadc.digital_estimator.care(-A, B, Q, R)


def test_brute_force_care():
    Vf = cbadc.digital_estimator.care(A, B, Q, R)
    Vb = cbadc.digital_estimator.care(-A, B, Q, R)

    Vf2 = cbadc.digital_estimator.bruteForceCare(A, B, Q, R)
    Vb2 = cbadc.digital_estimator.bruteForceCare(-A, B, Q, R)

    assert np.allclose(Vf, Vf2)
    assert np.allclose(Vb, Vb2)
