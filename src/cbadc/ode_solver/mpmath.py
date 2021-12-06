import cbadc
import sympy as sp
from mpmath import mp
from typing import List, Tuple


def invariant_system_solver(
    A: mp.matrix,
    B: mp.matrix,
    signals: List[cbadc.analog_signal._AnalogSignal],
    t_span: Tuple[mp.mpf, mp.mpf],
    homogeneous: bool = True,
    tol: float = 1e-12,
):
    N = A.cols
    M = B.cols
    non_hom_sol = mp.matrix(N, M)
    if homogeneous:
        hom_sol = mp.expm(A * t_span[1], method='taylor')
    else:
        hom_sol = mp.matrix(N, N)

    for m in range(M):

        def diff_equation(x, y):
            res = mp.matrix(N, 1)
            for m in range(M):
                res += B[:, m] * signals[m]._mpmath(x)
            for n in range(N):
                res += A[:, n] * y[n]
            return res

        y = mp.odefun(diff_equation, t_span[0], [0 for _ in range(N)], tol=tol)
        non_hom_sol[:, m] = mp.matrix(y(t_span[1]))

    return hom_sol, non_hom_sol
