"""Find the first step where the numba kernel diverges from the Python loop."""
import logging
import numpy as np
from numba import njit

logging.disable(logging.CRITICAL)
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import ZeroOrderHold


@njit(cache=True)
def _recursion(A, B, C, D, states, inputs, outputs, slew, smin, smax,
               o_scale, pre_gain, mid_thread, mid_rise, qmin, qmax, update, L):
    size, N, Jn = states.shape
    M = outputs.shape[1]
    W = inputs.shape[1]
    cA, cB, cC, cD, cU = A.shape[0], B.shape[0], C.shape[0], D.shape[0], update.shape[0]
    for i in range(1, size):
        ai, bi, ci, di, ui = i % cA, i % cB, i % cC, i % cD, i % cU
        for j in range(Jn):
            for r in range(N):
                acc = 0.0
                for c in range(N):
                    acc += A[ai, r, c] * states[i - 1, c, j]
                for w in range(W):
                    acc += B[bi, r, w] * inputs[i - 1, w, j]
                s = slew[r]
                if acc > s:
                    acc = s
                elif acc < -s:
                    acc = -s
                acc += states[i, r, j]
                if acc < smin[r]:
                    acc = smin[r]
                elif acc > smax[r]:
                    acc = smax[r]
                states[i, r, j] = acc
            for r in range(M):
                acc = outputs[i, r, j]
                for c in range(N):
                    acc += C[ci, r, c] * states[i, c, j]
                for w in range(W):
                    acc += D[di, r, w] * inputs[i, w, j]
                outputs[i, r, j] = acc
            for r in range(M):
                q = 2.0 * np.floor(pre_gain[r] * outputs[i, r, j] + mid_thread[r]) + mid_rise[r]
                if q < qmin[r]:
                    q = qmin[r]
                elif q > qmax[r]:
                    q = qmax[r]
                q *= o_scale[r]
                if update[ui, r]:
                    inputs[i, L + r, j] = q
                else:
                    inputs[i, L + r, j] = inputs[i - 1, L + r, j]


af, OSR = AnalogFrontend.chain_of_integrators(N=2, ENOB=10, BW=1e6)
size = 12
ref = ZeroOrderHold.uniform_reference_signal(
    af.dt, -np.ones((1, 1)), np.ones((1, 1)), size=size, seed=1)
af.analog_signal = ref

base = af.simulate_dt(size)
inputs, states, outputs, t = af._simulate_alloc(size, None, 0.0, np.double)
slew = np.broadcast_to(af.slew_rate * af.digital_control.dt, (af.N,)).astype(float).copy()
_recursion(
    af.A, af.B, af.C, af.D, states, inputs, outputs,
    slew, af.state_min.astype(float).copy(), af.state_max.astype(float).copy(),
    af._q_o_scale.ravel().copy(), af._q_pre_gain.ravel().copy(),
    af._q_mid_thread.ravel().copy(), af._q_mid_rise.ravel().copy(),
    af._q_min.ravel().copy(), af._q_max.ravel().copy(), af._q_update, af.L,
)
nb_v = inputs[:, af.L:, :]

print("quant params: o_scale", af._q_o_scale.ravel(), "pre_gain", af._q_pre_gain.ravel(),
      "mid_thread", af._q_mid_thread.ravel(), "mid_rise", af._q_mid_rise.ravel(),
      "qmin", af._q_min.ravel(), "qmax", af._q_max.ravel())
print("\nstep |       x ref        |  x numba          | v ref | v numba")
for i in range(min(size, 9)):
    print(f"{i:3d}  | {np.ravel(base['x'][i]).round(4)} | {np.ravel(states[i]).round(4)} "
          f"| {np.ravel(base['v'][i]).round(2)} | {np.ravel(nb_v[i]).round(2)}")
