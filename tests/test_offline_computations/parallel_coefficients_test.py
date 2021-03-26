import numpy as np
import scipy
import scipy.linalg
import sys
import os

# this is python madness
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")
if myPath:
    from cbadc.offline_computations import care


def test_diagonal_elements_of_parallel_version():
    """
    This file was created to investigate the behavior of digitalization. especially
    with respect to imaginary and real numbers of the resulting filter coefficients
    """
    N = 2
    beta = 10.0
    rho = -1.0
    eta2 = 1e0

    A = beta * np.eye(N, k=-1) + rho * np.eye(N, k=1)
    B = np.zeros((N, 1))
    B[0, 0] = beta
    Ts = 1e-3

    CT = np.zeros((N,1))
    CT[0, -1] = 1.0
    R = np.array([[eta2]])

    CT = np.eye(N)
    R = np.eye(N) * eta2

    Vf, Vb = care(A.transpose(), CT, np.outer(B, B), R)
    eta2inv = np.eye(N) / eta2
    tempAf = A - np.dot(Vf, eta2inv)
    tempAb = A + np.dot(Vb, eta2inv)

    Af = scipy.linalg.expm(tempAf * Ts)
    Ab = scipy.linalg.expm(-tempAb * Ts)

    Df, Qf = scipy.linalg.eig(Af)
    Db, Qb = scipy.linalg.eig(Ab)

    assert (Df.imag != 0).any()
    assert (Db.imag != 0).any()
    assert (Qf.imag != 0).any()
    assert (Qb.imag != 0).any()
