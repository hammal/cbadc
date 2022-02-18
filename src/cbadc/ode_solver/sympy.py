import logging
import sympy as sp
from typing import List
import multiprocessing
import copy

logger = logging.getLogger(__name__)

# Current problem with result not being pickable...
PARALLEL_PROCESSES = False


def _non_hom_solver(eq, func, t0=0):
    hint = 'nth_linear_constant_coeff_undetermined_coefficients'
    sol = sp.dsolve(eq, func, hint=hint, doit=True)
    constants = sp.numbered_symbols(prefix='C', cls=sp.Symbol, start=1)
    Cons = [next(constants) for _ in func]
    t = sp.Symbol('t', real=True)
    zero_solution = [s.subs(t, t0).rhs for s in sol]
    initial_condition = sp.solve([t, *zero_solution], dict=True)
    tmp = [
        s.subs([(c, initial_condition[0][c]) for c in Cons])
        # .rewrite(
        #    sp.sin
        # )  # .simplify()
        for s in sol
    ]
    # tmp not pickable :(
    return tmp


class Process:
    def __init__(self, target, timeout: float, *args):
        self._target = target
        self._args = args
        self.queue = multiprocessing.Queue()
        self.timeout = timeout

        def modified_target(queue: multiprocessing.Queue, target, *args):
            res = target(*args)
            queue.put(res)

        self.process = multiprocessing.Process(
            target=modified_target, args=(self.queue, target, *args)
        )

        if PARALLEL_PROCESSES:
            self.process.start()

    def get(self):
        if PARALLEL_PROCESSES:
            res = self.queue.get()
            # self.process.join()
            if self.process.exitcode:
                raise Exception(f"Process ended with exitcode {self.process.exitcode}")
        else:
            res = self._target(*self._args)
        return copy.deepcopy(res)


def invariant_system_solver(
    A: sp.Matrix,
    B: sp.Matrix,
    signals: List[sp.Function],
    initial_condition_time: List[float],
    timeout: float = 60.0,
):
    # using mpmath to do SVD
    # U, S, VH = A.bidiagonal_decomposition()

    # B_new = VH * B
    N = A.cols
    M = B.cols

    t = sp.Symbol('t', real=True)

    non_hom_sol = []

    processes = []

    hom_sol = sp.solvers.ode.systems.matrix_exp(A, t)
    # hom_sol = sp.re(sp.simplify(hom_sol))

    # Start processes
    variables = sp.numbered_symbols(prefix='x', cls=sp.Function, start=1)
    x = [next(variables) for _ in range(N)]
    func = [f(t) for f in x]

    for m in range(M):
        eqs = []
        for n in range(N):
            expr = B[n, m] * signals[m]
            for nn in range(N):
                expr += A[n, nn] * func[nn]
            eqs.append(sp.Eq(func[n].diff(t), expr))
        processes.append(
            Process(_non_hom_solver, timeout, eqs, func, initial_condition_time[m])
        )

    # Retrive results
    for m, process in enumerate(processes):
        sol_m = process.get()
        non_hom_sol.append(sol_m)
        # for n in range(N):
        #     non_hom_sol[m][n] = sol_m[n]  # .expand(complex=True)
        # .subs(
        # t, evaluation_time).rhs.rewrite(sp.exp)

    # non_hom_sol = (non_hom_sol).expand(complex=True)
    # non_hom_sol = sp.re(sp.simplify(non_hom_sol))

    return hom_sol, non_hom_sol, t
