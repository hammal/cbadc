"""Quick try: numba @njit the simulate_dt recursion; verify + measure speedup.

Mirrors the exact discrete-time loop (slew clip, state clip, inlined quantizer +
cyclic hold) with explicit scalar loops, runs it on the SAME buffers the Python
path uses, checks bit-identity, and times both.
"""
import logging
import time

import numpy as np
from numba import njit

logging.disable(logging.CRITICAL)
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import ZeroOrderHold


@njit(cache=True, fastmath=False)
def _recursion(A, B, C, D, states, inputs, outputs, slew, smin, smax,
               o_scale, pre_gain, mid_thread, mid_rise, qmin, qmax, update, L):
    size, N, Jn = states.shape
    M = outputs.shape[1]
    cA, cB, cC, cD, cU = A.shape[0], B.shape[0], C.shape[0], D.shape[0], update.shape[0]
    for i in range(1, size):
        ai, bi, ci, di, ui = i % cA, i % cB, i % cC, i % cD, i % cU
        # linear part via BLAS (np.dot) so rounding matches the numpy matrix path
        lin = np.dot(A[ai], states[i - 1]) + np.dot(B[bi], inputs[i - 1])  # (N, Jn)
        for r in range(N):
            s = slew[r]
            for j in range(Jn):
                v = lin[r, j]
                if v > s:
                    v = s
                elif v < -s:
                    v = -s
                v += states[i, r, j]  # pre-init (noise)
                if v < smin[r]:
                    v = smin[r]
                elif v > smax[r]:
                    v = smax[r]
                states[i, r, j] = v
        out = np.dot(C[ci], states[i]) + np.dot(D[di], inputs[i])  # (M, Jn)
        for r in range(M):
            for j in range(Jn):
                o = outputs[i, r, j] + out[r, j]  # pre-init (noise) + linear
                outputs[i, r, j] = o
                q = 2.0 * np.floor(pre_gain[r] * o + mid_thread[r]) + mid_rise[r]
                if q < qmin[r]:
                    q = qmin[r]
                elif q > qmax[r]:
                    q = qmax[r]
                q *= o_scale[r]
                if update[ui, r]:
                    inputs[i, L + r, j] = q
                else:
                    inputs[i, L + r, j] = inputs[i - 1, L + r, j]


def numba_sim(af, size):
    inputs, states, outputs, t = af._simulate_alloc(size, None, 0.0, np.double)
    af._quantize(outputs[0], 0)  # ensure params populated
    slew = np.broadcast_to(af.slew_rate * af.digital_control.dt, (af.N,)).copy()
    _recursion(
        af.A, af.B, af.C, af.D, states, inputs, outputs,
        slew, af.state_min.copy(), af.state_max.copy(),
        af._q_o_scale.ravel().copy(), af._q_pre_gain.ravel().copy(),
        af._q_mid_thread.ravel().copy(), af._q_mid_rise.ravel().copy(),
        af._q_min.ravel().copy(), af._q_max.ravel().copy(),
        af._q_update, af.L,
    )
    return {"v": inputs[:, af.L:, :], "x": states}


for name, (N, E) in {"chain N=4": (4, 12), "leapfrog N=6": (6, 20)}.items():
    af0, OSR = (AnalogFrontend.chain_of_integrators if "chain" in name
                else AnalogFrontend.leapfrog)(N=N, ENOB=E, BW=1e6)
    # discretize once so BOTH paths use the same discrete-time matrices
    # (simulate_dt would otherwise re-discretize internally each call)
    af = af0 if af0.is_discrete_time else af0.discretize(dt=af0.digital_control.dt)
    size = 1 << 18
    ref = ZeroOrderHold.uniform_reference_signal(
        af.dt, -np.ones((1, 8)), np.ones((1, 8)), size=size, seed=1)
    af.analog_signal = ref

    # warm up numba JIT (compile) on a tiny run
    numba_sim(af, 256)

    t0 = time.time(); base = af.simulate_dt(size); tb = time.time() - t0
    t0 = time.time(); nb = numba_sim(af, size); tn = time.time() - t0

    dv = np.abs(base["v"] - nb["v"]).max()
    dx = np.abs(base["x"] - nb["x"]).max()
    print(f"{name:14} size={size}  python={tb:6.2f}s  numba={tn:6.3f}s  "
          f"speedup={tb/tn:5.1f}x  max|dv|={dv:.1e} max|dx|={dx:.1e}", flush=True)
