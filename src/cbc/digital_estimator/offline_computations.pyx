import numpy as np
import scipy as sp
import scipy.linalg
from numpy.linalg import LinAlgError
import time
#cython: language_level=3

def bruteForceCare(A, B, Q, R):
    timelimit = 10 * 60
    start_time = time.time()
    V = np.eye(A.shape[0])
    V_tmp = np.zeros_like(V)
    tau = 1e-5
    RInv = np.linalg.inv(R)

    while not np.allclose(V, V_tmp, rtol=1e-5, atol=1e-8):
        if time.time() - start_time > timelimit:
            raise Exception("Brute Force CARE solver ran out of time")
        V_tmp = V
        try:
            V = V + tau * (
                np.dot(A, V)
                + np.transpose(np.dot(A, V))
                + Q
                - np.dot(V, np.dot(B, np.dot(RInv, np.dot(B.transpose(), V))))
            )
        except FloatingPointError:
            print("V_frw:\n{}\n".format(V))
            print("V_frw.dot(V_frw):\n{}".format(np.dot(V, V)))
            raise FloatingPointError
    return V


def care(A, B, Q, R):
    """
    This function solves the forward and backward continuous Riccati equation.
    """
    A = np.array(A, dtype=np.double)
    B = np.array(B, dtype=np.double)
    Q = np.array(Q, dtype=np.double)
    R = np.array(R, dtype=np.double)

    try:
        Vf = sp.linalg.solve_continuous_are(A, B, Q, R)
    except LinAlgError:
        print(
            """Cholesky Method Failed for computing the CARE of Vf.
            Starting brute force"""
        )
        Vf = bruteForceCare(A, B, Q, R)

    try:
        Vb = sp.linalg.solve_continuous_are(-A, B, Q, R)
    except LinAlgError:
        print(
            """Cholesky Method Failed for computing the CARE of Vb.
            Starting brute force"""
        )
        Vb = bruteForceCare(-A, B, Q, R)
    # RInv = np.linalg.inv(R)
    # cdef double tau = 1e-15
    # Vf = np.array(Vf, dtype=np.longdouble)
    # Vb = np.array(Vb, dtype=np.longdouble)
    # for _ in range(1000):
    #     Vf = Vf + tau * (
    #         np.dot(A, Vf)
    #         + np.transpose(np.dot(A, Vf))
    #         + Q
    #         - np.dot(Vf, np.dot(B, np.dot(RInv, np.dot(B.transpose(), Vf))))
    #     )
    #     Vb = Vb + tau * (
    #         np.dot(-A, Vb)
    #         + np.transpose(np.dot(-A, Vb))
    #         + Q
    #         - np.dot(Vb, np.dot(B, np.dot(RInv, np.dot(B.transpose(), Vb))))
    #     )
    return np.array(Vf, dtype=np.double), np.array(Vb, dtype=np.double)
