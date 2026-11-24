"""
The analog frontend
===================

Everything in :mod:`cbadc.frontend` is one object, a linear state space model.
A constant, a tone and a decaying exponential are state spaces; so is a
filter, the DAC pulse that drives a converter, and the converter itself. This
tutorial follows the waveforms up from that one object: autonomous, driven,
restarted at the clock, then composed into hierarchies, then the converter,
and finally the indexing that lets one set of matrices stand for a whole
batch of simulations of a switched circuit.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.linalg import expm

from cbadc.frontend import (
    DAC,
    SOS,
    Constant,
    Exp,
    Filter,
    Frontend,
    Pole,
    Sinusoid,
    StateSpace,
    cascade,
    parallel,
)

# %%
# A state space model
# -------------------
# The state ``x`` is a vector of ``N`` entries, the input ``u`` a vector of
# ``L`` and the output ``y`` a vector of ``M``; ``x_n``, ``u_l`` and ``y_m``
# are single entries. Four matrices connect them::
#
#     x' = A x + B u        A: (N, N)   B: (N, L)
#     y  = C x + D u        C: (M, N)   D: (M, L)
#
# Row ``n`` of ``A`` and ``B`` is the derivative of ``x_n``: ``A[n, n']`` weighs
# state ``n'`` and ``B[n, l]`` weighs input ``l``. Row ``m`` of ``C`` and ``D``
# is the output ``y_m``. :class:`~cbadc.frontend.StateSpace` takes the four
# matrices as scalars, lists or any array-like such as a numpy array, and
# stores them behind two leading axes; what those axes are for comes last.
# Printing a system shows its ``[0, 0]`` slice as the block matrix
# ``[[A, B], [C, D]]`` with rules where the states end, so ``N``, ``L`` and
# ``M`` can be read off the picture. Every system in this tutorial is shown
# this way. An integrator, ``x' = u`` with ``y = x``, is the smallest example.
integrator = StateSpace(A=0.0, B=1.0, C=1.0, D=0.0)
print(integrator)

# %%
# Autonomous: the sources
# -----------------------
# With ``B = 0`` and ``D = 0`` nothing drives the system; it runs from a start
# state ``x0`` on its own and its output is ``y(t) = C e^{A t} x0``. Every
# signal the library needs is of this form. A constant ``c`` is one state that
# does not move; a rotation of two states at ``w = 2 pi f`` carries ``cos(w t)``
# and ``sin(w t)`` together, the start state picking the phase; a decay with
# time constant ``tau`` is one state leaking to zero::
#
#     Constant(c)              A = [0]                 C = [1]      x0 = [c]       y = c
#     Sinusoid(1, f)           A = [[0, -w], [w, 0]]   C = [1, 0]   x0 = [1, 0]    y = cos(w t)
#     Sinusoid(1, f, -pi/2)    A = [[0, -w], [w, 0]]   C = [1, 0]   x0 = [0, -1]   y = sin(w t)
#     Exp(tau)                 A = [-1 / tau]          C = [1]      x0 = [1]       y = exp(-t / tau)
#
# The constructors fill in the matrices and ``x0``, which the print adds as a
# column to the right of ``B``.
sources = {
    "Constant(0.5)": Constant(0.5),
    "cos, 100 kHz": Sinusoid(1.0, 1e5),
    "sin, 100 kHz": Sinusoid(1.0, 1e5, -np.pi / 2),
    "Exp(2 us)": Exp(2e-6),
}
for name, w in sources.items():
    print(f"{name}\n{w}")

# %%
# Evaluating ``C e^{A t} x0`` on a time grid gives the waveforms themselves;
# nothing was sampled or integrated to draw them.
t = np.linspace(0, 20e-6, 400)
fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
for name, w in sources.items():
    ax.plot(t * 1e6, w.value(t)[:, 0], lw=1.5, label=name)
ax.set_xlabel("t [us]")
ax.set_ylabel("y(t) = C exp(A t) x0")
ax.grid(alpha=0.3)
ax.legend(ncol=4, fontsize=9, loc="lower center")
plt.show()

# %%
# Driven: the filters
# -------------------
# With ``B != 0`` the system is driven, and the same matrices have three faces::
#
#     H(s) = C (s I - A)^{-1} B + D         transfer function
#     h(t) = C e^{A t} B                    impulse response: the source with x0 = B
#     Constant(1) >> f                      step response: a constant through the filter
#
# The simplest filter is one pole, ``Pole(tau)``: ``A = [-1 / tau]``,
# ``B = [1 / tau]``, ``C = [1]``, so ``H(s) = 1 / (1 + s tau)``. Any rational
# design fits the same form: here fourth-order Butterworth, Chebyshev I (1 dB
# passband ripple), Chebyshev II (40 dB stopband) and elliptic (1 dB, 40 dB)
# low-passes at 1 MHz from :mod:`scipy.signal`, handed over as analog
# second-order sections. :func:`~cbadc.frontend.SOS` builds one two-state
# :func:`~cbadc.frontend.Filter` per section and chains them, so each biquad's
# poles sit on their own diagonal block of ``A``.
f_c = 2 * np.pi * 1e6
designs = {
    "Butterworth": signal.butter(4, f_c, analog=True, output="zpk"),
    "Chebyshev I": signal.cheby1(4, 1, f_c, analog=True, output="zpk"),
    "Chebyshev II": signal.cheby2(4, 40, f_c, analog=True, output="zpk"),
    "elliptic": signal.ellip(4, 1, 40, f_c, analog=True, output="zpk"),
}
filters = {
    name: SOS(signal.zpk2sos(*zpk, analog=True)) for name, zpk in designs.items()
}
pole = Pole(1 / f_c)
print(pole)
print(
    np.round(filters["Butterworth"].A[0, 0] / f_c, 3)
)  # (4, 4) in units of w_c: two 2 x 2 blocks

# %%
# Why sections and balancing: one companion matrix for the same Butterworth
# has entries spanning ``w_c`` to ``w_c^4``, which ``expm`` and the products
# downstream pay for in precision. ``Filter`` balances by default and
# ``SOS`` balances the whole cascade.
b, a = signal.zpk2tf(*designs["Butterworth"])
companion = Filter(*signal.tf2ss(b, a), balance=False)
print(
    f"max |A|: companion {np.abs(companion.A).max():.1e}, sections {np.abs(filters['Butterworth'].A).max():.1e}"
)

# %%
# The transfer functions on a log grid, the step responses from
# ``Constant(1) >> f`` and the impulse responses from the sources
# ``f.impulse()``, all in closed form.
f = np.logspace(5, 7, 400)
t = np.linspace(0, 4e-6, 400)
fig, (ax_h, ax_t) = plt.subplots(1, 2, figsize=(10, 3.2), constrained_layout=True)
for i, (name, flt) in enumerate(filters.items()):
    H = flt.transfer_function(2j * np.pi * f)[:, 0]
    ax_h.semilogx(f, 20 * np.log10(np.abs(H)), lw=1.5, color=f"C{i}", label=name)
    step = (Constant(1.0) >> flt).value(t)[:, 0]
    ax_t.plot(t * 1e6, step, lw=1.5, color=f"C{i}", label=name)
    if not np.any(
        flt.D
    ):  # a feed-through D is a delta: no impulse response as a source
        h = flt.impulse().value(t)[:, 0] / f_c
        ax_t.plot(t * 1e6, h, lw=1, ls="--", color=f"C{i}")
ax_h.set(xlabel="f [Hz]", ylabel="|H| [dB]", ylim=(-80, 5))
ax_t.set(xlabel="t [us]", ylabel="step (solid), impulse / w_c (dashed)")
for ax in (ax_h, ax_t):
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
plt.show()

# %%
# The Chebyshev I and elliptic steps settle at ``10^(-1/20) = 0.891``, the 1 dB
# ripple at DC of an even-order design. Chebyshev II and elliptic have as many
# zeros as poles, so ``D != 0`` and their impulse response is a delta plus a
# source; only the all-pole designs have an impulse response that is a source
# alone.

# %%
# Restarted at the clock: the DAC pulses
# --------------------------------------
# A DAC pulse is a source with one more rule. At every clock edge ``i`` the
# states of its shape are reset to the start state scaled by the decision,
# ``x[i dt] = x0 v[i]`` for an undelayed pulse, so the shape replays with the sign
# of the bit; a ``window = (delay, width)`` in periods gates it to
# ``[delay, delay + width)``. :class:`~cbadc.frontend.DAC` wraps the shape,
# and ``>> f`` appends a filter whose states carry over instead of restarting.
# Four pulses, the first three one state each::
#
#     NRZ      DAC(Constant(1))                          A = [0]          x0 = [1]
#     RZ       DAC(Constant(1)).with_window(0.25, 0.5)   A = [0]          x0 = [1]   window = (0.25, 0.5)
#     SC       DAC(Exp(tau)).with_window(0, 0.5)         A = [-1 / tau]   x0 = [1]   window = (0, 0.5)
#     NRZ, RC  DAC(Constant(1)) >> Pole(tau)             A = [[-1/tau, 1/tau], [0, 0]]   x0 = [0, 1]   restart = [False, True]
#
# The last one is a hierarchy already: the constant restarts, the filter state
# behind it does not, so the settling carries over from one bit to the next.
dt, tau = 1e-7, 2e-8
pulses = {
    "NRZ": DAC(Constant(1.0)),
    "RZ": DAC(Constant(1.0)).with_window(0.25, 0.5),
    "SC": DAC(Exp(tau)).with_window(0.0, 0.5),
    "NRZ through RC": DAC(Constant(1.0)) >> Pole(tau),
}
for name, w in pulses.items():
    print(f"{name:<15} N = {w.N}  restart = {w.restart.tolist()}  window = {w.window}")

# %%
# The pulse train of a decision sequence is the superposition of the replayed
# pulses, ``sum_i v[i] p(t - i dt)``: what the DAC puts on the wire. (The
# simulator never forms it; it integrates each pulse exactly over its window.)
v = np.array([1, -1, -1, 1, 1, -1])
t = np.linspace(0, v.size * dt, 1200, endpoint=False)
fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
for name, w in pulses.items():
    delay, width = w.window or (0.0, 1.0)

    def y(tau, w=w):  # C e^{A tau} x0 for tau >= 0, zero before
        return np.where(tau >= 0, w.value(np.maximum(tau, 0))[:, 0], 0.0)

    train = np.zeros_like(t)
    for i, v_i in enumerate(v):
        tau_i = t - i * dt
        if w.restart.all():  # a gated source: its output is on inside the window
            p = y(tau_i) * (tau_i >= delay * dt) * (tau_i < (delay + width) * dt)
        else:  # a constant switched on and off through the filter: two steps
            p = y(tau_i - delay * dt) - y(tau_i - (delay + width) * dt)
        train += v_i * p
    ax.plot(t / dt, train, lw=1.5, label=name)
ax.set(xlabel="t / dt", ylabel="DAC output", xticks=range(v.size + 1))
ax.grid(alpha=0.3)
ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.25))
plt.show()

# %%
# Hierarchies: cascade, parallel and the operators
# ------------------------------------------------
# Two compositions are the only algebra in the module, and every larger
# object is built from them. :func:`~cbadc.frontend.cascade` chains two
# systems, ``a -> b``: the single output of ``a`` replaces input ``col`` of
# ``b``. The states stack as ``[x_b; x_a]`` so the rows of ``b`` keep their
# place, and with ``b_c = B_b[:, col]`` and ``d_c = D_b[:, col]``::
#
#     A = [[A_b, b_c C_a], [0, A_a]]    B = [[B_b with column col = b_c D_a], [B_a in column col]]
#     C = [C_b, d_c C_a]                D = D_b with column col = d_c D_a
#
# Two integrators in a row give ``N = 2``: the chain ``x_1' = x_2``,
# ``x_2' = u`` read out at ``x_1``, so row ``n = 1`` of ``A`` picks up state 2.
chain = cascade(integrator, integrator)
print(chain)

# %%
# ``col`` is the input index ``l`` and matters when ``b`` has ``L > 1``. A
# model ``x' = u - v`` has the signal on ``l = 0`` and the control on ``l = 1``;
# cascading a filter onto ``col = 1`` shapes only the control path. ``B`` keeps
# both columns, column 1 now feeding the filter state.
model = StateSpace(A=0.0, B=[[1.0, -1.0]], C=1.0, D=[[0.0, 0.0]])
filtered = cascade(Pole(1.0), model, col=1)
print(filtered)

# %%
# :func:`~cbadc.frontend.parallel` puts two systems on the same input and sums
# their outputs, ``y = y_a + y_b``; the states stack as ``[x_a; x_b]`` and
# nothing couples them::
#
#     A = [[A_a, 0], [0, A_b]]    B = [B_a; B_b]
#     C = [C_a, C_b]              D = D_a + D_b
#
# Two poles in parallel are a second-order filter with a block-diagonal ``A``.
two_poles = parallel(Pole(0.5), Pole(1.0))
print(two_poles)

# %%
# The two compositions are operators too, plus one more. ``a >> f``
# is ``cascade`` onto a filter (the RC-settled pulse above), ``a + b`` is
# ``parallel`` with the start states stacked, and ``a * b`` multiplies two
# sources: the Kronecker product ``x_a (x) x_b`` with ``A = A_a (x) I + I (x) A_b``,
# so a damped tone or a modulated carrier is still one source, simulated
# exactly. An IQ carrier is ``I cos(w t) - Q sin(w t)`` in four factors.
f_m, f_car = 1e5, 1e6
cos_c, msin_c = Sinusoid(1.0, f_car), Sinusoid(1.0, f_car, np.pi / 2)  # -sin(w t)
composed = {
    "two tones: a + b": Sinusoid(0.5, f_m) + Sinusoid(0.5, 3 * f_m),
    "damped tone: a * Exp": Sinusoid(1.0, 3 * f_m) * Exp(4e-6),
    "IQ carrier": Sinusoid(1.0, f_m) * cos_c + Sinusoid(1.0, f_m, -np.pi / 2) * msin_c,
}
t = np.linspace(0, 20e-6, 2000)
fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
for name, w in composed.items():
    ax.plot(t * 1e6, w.value(t)[:, 0], lw=1, label=f"{name} (N = {w.N})")
ax.set(xlabel="t [us]", ylabel="y(t)")
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
plt.show()

# %%
# The Frontend
# ------------
# A converter is a driven system whose input has two parts, the signal ``u``
# and the control ``v`` that its own quantiser feeds back::
#
#     x' = A x + B [u; v]        B = [B_L | B_M]: L signal columns, M control columns
#     y  = C x + D [u; v]        D = [D_L | D_M]
#     v  = Q(y)                  levels codes per channel, v in [-1, 1]
#
# ``Q`` reads ``y`` in units of full scale and returns the nearest of
# ``levels`` codes, mid-rise for even and mid-tread for odd ``levels``; the
# DAC gain lives in ``B_M``. :class:`~cbadc.frontend.Frontend` takes the four
# matrices and the clock period ``dt``, and the synthesis constructors
# (``leapfrog``, ``chain_of_integrators``, ``ctsdm``, ``dtsdm``, ``mash``)
# build the standard loops. A third-order leap-frog for 1 MHz at OSR 32:
# ``B_L`` is the first column and ``B_M`` the three DAC columns, ``C = I``
# compares every state, ``D = 0``.
af, OSR = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
print(af)
print(f"fs = {af.fs / 1e6:.1f} MHz, levels = {af.levels.tolist()}")

# %%
# The inputs and the DACs are the hierarchies above, cascaded onto the
# columns of ``B``: ``af.input`` holds ``L`` sources (or
# :class:`~cbadc.frontend.Samples`, predetermined DAC drive values),
# ``af.dac`` holds ``M``
# pulses, and a single one is repeated over the columns. Every non-ideality
# is an attribute, off by default: the white state noise intensity
# ``state_covariance``, the per-decision comparator-input ``output_covariance`` and
# ``comparator_input_offset``, the swing ``state_min``/``state_max``, and ``slew_rate``.
af.input = Sinusoid(0.5, 1e5)
af.dac = DAC(Constant(1.0)).with_window(0.0, 0.5)  # return-to-zero on all three
af.state_covariance = 1e-12 * np.eye(3)
af.comparator_input_offset = [0.01, 0.0, 0.0]
print(
    f"L = {af.L} input, N = {[w.N for w in af.input]};  M = {af.M} DACs, windows {[w.window for w in af.dac]}"
)

# %%
# ``simulate(size)`` runs ``size`` clock periods and returns the records,
# time first and batch last: decisions ``v``, input ``u`` and digital-twin
# ``y_hat`` by default. ``record="all"`` adds ``t``, ``x``, ``y`` and ``x_hat``.
# The control keeps every state inside its bound
# while the density of ones in ``v`` follows the input.
sim = af.simulate(1 << 12, record="all", verbose=False)
print({key: value.shape for key, value in sim.items()})

fig, (ax_x, ax_v) = plt.subplots(
    2, 1, figsize=(7, 4), sharex=True, constrained_layout=True
)
i = slice(0, 640)  # one period of the tone
for n in range(3):
    ax_x.plot(sim["t"][i] * 1e6, sim["x"][i, n, 0], lw=1, label=f"$x_{n + 1}$")
ax_x.plot(sim["t"][i] * 1e6, sim["u"][i, 0, 0], "k--", lw=1, label="$u$")
ax_v.step(sim["t"][i] * 1e6, sim["v"][i, 0, 0], lw=0.6, where="post")
ax_x.set(ylabel="states"), ax_x.legend(ncol=4, fontsize=9, loc="upper right")
ax_v.set(xlabel="t [us]", ylabel="$v_1$", yticks=[-1, 1])
plt.show()

# %%
# The digital twin
# ----------------
# When requested, the twin replays the frontend matrices on its decisions:
# ``x_hat' = A x_hat + B_M v``, ``y_hat = C_hat x_hat + D_hat v_held``.
# ``v_held`` is the preceding decision; here ``D_hat = 0``.
# By linearity ``x = x_u + x_hat`` with ``x_u`` the response to the input
# alone, so ``-x_hat`` estimates ``x_u`` and the bounded state ``x`` is the
# exact error; with the default ``C_hat = -e_N``, the last state, ``y_hat``
# estimates ``x_u,N`` and ``x_N`` is its error. Unit-circle modes retain
# switch-on transients, making raw-twin tone SNDR unsuitable; use the Wiener
# estimate ``u_hat`` instead. Single-comparator DSMs set ``C_hat = 0`` and
# ``D_hat = 1``, reading the preceding decision directly. Here a lightly
# damped leapfrog lets the input-only response settle:
lf, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
af = Frontend(
    lf.A[0, 0] - 2 * np.pi * 1e5 * np.eye(3), lf.B[0, 0], lf.C[0, 0], dt=lf.dt
)
af.input = Sinusoid(0.5, 1e5)
sim = af.simulate(1 << 12, record="all", verbose=False)
model = StateSpace(af.A[0, 0], af.B[0, 0, :, :1], np.eye(3), np.zeros((3, 1)))
x_u = cascade(af.input[0], model).trajectory(af.dt, 1 << 12)[:, 0, :3]
print(
    "max |x - x_hat - x_u| / max |x_u| =",
    np.abs(sim["x"][..., 0] - sim["x_hat"][..., 0] - x_u).max() / np.abs(x_u).max(),
)

fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
i = slice(2048, 2048 + 640)  # one period of the tone
ax.plot(sim["t"][i] * 1e6, x_u[i, 2], "k", lw=3, alpha=0.3, label="$x_{u,3}$")
ax.plot(sim["t"][i] * 1e6, sim["y_hat"][i, 0, 0], "C0", label=r"$\hat y = -\hat x_3$")
ax.plot(
    sim["t"][i] * 1e6,
    sim["x"][i, 2, 0],
    "C3",
    lw=0.6,
    label=r"$x_3 = x_{u,3} - \hat y$",
)
ax.set(xlabel="t [µs]", ylabel="last state"), ax.legend(ncol=3, loc="upper right")
plt.show()

# %%
# The four indexes ``(J, K, ., .)``
# ---------------------------------
# Every matrix in the module is stored as ``(J, K, rows, cols)``; the two
# leading axes stayed hidden so far because they had length one. ``j`` is the
# batch index, one independent system per batch member simulated side by side. A
# list of amplitudes gives a batch of tones: one source whose ``x0`` has a
# row per ``j``, and a frontend fed with it has ``J = 2`` and every record a
# last axis of two.
A, B, C = lf.A[0, 0], lf.B[0, 0], lf.C[0, 0]
tone = Sinusoid([0.1, 0.5], 1e5)
print(tone)
batch = Frontend(A, B, C, dt=lf.dt, input=tone)
sim = batch.simulate(1 << 12, record=("v", "x"), verbose=False)
print(
    f"J = {batch.J}, v {sim['v'].shape}, max |x_3| per batch member {np.abs(sim['x'][:, 2]).max(0).round(2).tolist()}"
)

# %%
# A matrix given as ``(J, K, ., .)`` runs one frontend per batch member: a Monte
# Carlo over mismatch is ``A`` with a batch axis of ``J`` perturbed copies,
# while ``B`` and ``C`` keep their length-one axes and are shared.
# ``J_frontend`` counts matrix variants, ``J`` includes the inputs, and
# ``frontend[j]`` selects one complete batch member for the analytic tools.
# The selected frontend keeps all ``K`` clock phases; a phase-specific matrix
# is available directly as ``frontend.A[j, k]`` when inspecting the model.
rng = np.random.default_rng(1)
spread = 1.0 + 0.05 * rng.standard_normal((16, 3, 3))
mc = Frontend(lf.A * spread[:, None], B, C, dt=lf.dt, input=Sinusoid(0.5, 1e5))
sim = mc.simulate(1 << 12, record=("v", "x"), verbose=False)
print(
    f"A {mc.A.shape}, B {mc.B.shape}: J_frontend = {mc.J_frontend}, J = {mc.J}, v {sim['v'].shape}"
)
print("max |x_3| per frontend:", np.abs(sim["x"][:, 2]).max(0).round(2).tolist())

# %%
# ``k`` is the cyclic step: ``A[:, i % K]`` applies at step ``i``, so a converter
# whose matrices repeat with period ``K`` is one frontend with ``K > 1``, and a
# plain ``(N, N)`` matrix is the same at every step. A channel whose ``C`` and
# ``D`` rows are zero at step ``k`` does not look and holds its decision;
# :meth:`~cbadc.frontend.Frontend.with_trigger` writes such rows. Channels
# 0 and 1 deciding on alternate clocks, channel 2 on every one:
ti = lf.with_trigger([2, 2, 1], [0, 1, 0])
print(f"K = {ti.K}, C {ti.C.shape}")
print("C[0]:", ti.C[0, 0].tolist(), "\nC[1]:", ti.C[0, 1].tolist())

# %%
# A chopper is the same mechanism on signs. Step ``i`` runs from edge ``i-1``
# to edge ``i`` with ``B[i % K]`` and decides with ``C[i % K]``, so ``C[k]``
# alternates at the edges and the DAC undoes it during the following period:
# the sign of the decision taken with ``C[k]`` sits in ``B[(k + 1) % K]``.
# The states are those of the unchopped loop; the control alternates.
B_neg = B.copy()
B_neg[:, 1:] *= -1
chopped = Frontend(A, np.stack([B_neg, B]), np.stack([C, -C]), dt=lf.dt)
chopped.input = plain = Sinusoid(0.5, 1e5)
ref = Frontend(A, B, C, dt=lf.dt, input=plain).simulate(
    1 << 10, record=("v", "x"), verbose=False
)
chop = chopped.simulate(1 << 10, record=("v", "x"), verbose=False)
print(
    f"K = {chopped.K}: x identical {np.allclose(ref['x'], chop['x'])}, v[1::2] negated {np.array_equal(chop['v'][1::2], -ref['v'][1::2])}"
)

# %%
# The rules, then. A matrix is ``(J, K, rows, cols)`` and an axis of length
# one is shared, so a plain matrix, a cyclic ``(K, ., .)`` stack and a batch
# ``(J, K, ., .)`` each mean what they look like; ``A[:, i % K]`` applies at step
# ``i``; records are ``(size, ., J)``, time first and batch last; a source
# carries ``J`` in ``x0``, and in ``A`` only when the frequencies differ.
print(f"tone     A {tone.A.shape}  x0 {tone.x0.shape}")
print(f"trigger  C {ti.C.shape}")
print(f"MC       A {mc.A.shape}  B {mc.B.shape}")
print(f"records  v {sim['v'].shape}  x {sim['x'].shape}")

# %%
# Discretization
# --------------
# Between two edges nothing is decided, so the frontend with every waveform
# cascaded in is integrated in closed form, one matrix exponential per cyclic
# step ``k`` and batch member ``j``. The result is a one-step map on the
# stacked state and the quantiser input::
#
#     [z[i]; y[i]] = ABCD[j, k] [z[i-1]; v[i-1]] + excitation[i],    v[i] = Q(y[i]),    k = i % K
#
# The ``N_x`` state rows are the ``N`` design states, then the states the
# decisions alone drive (DAC filter states, latch memory for a pulse longer
# than a period), then sampled-input pulse, filter and source states and a
# constant carrying
# the offsets. The simulation is this one product per step with the
# quantiser inline. The default RZ pulse stays within one period, so a bare
# leap-frog has ``N_x = N``. The ``y`` rows read the
# frontend states through ``C`` since ``D = 0``.
rz, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
dz = rz.discretize()
print(dz)
print(np.round(dz.ABCD[0, 0], 3))

# %%
# The control columns of ``B_d`` are the pulse integrals
# ``int e^{A (dt - s)} B_M p(s) ds`` over the window, so a DAC shape changes
# those columns. Compare the default equal-area RZ pulse with an explicit
# delayed NRZ pulse: the latter spills into the next step and needs ``M``
# decision-memory states. A source input adds its states to
# ``N_x``, and the diagnostics say
# how stiff the step was and that one step agrees with two half steps.
delayed, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
delayed.dac = DAC(Constant(1.0), window=(0.5, 1.0))
print("Default RZ:\n", np.round(dz.B_d[0, 0, :3, 1:], 3))
print("Delayed NRZ:\n", np.round(delayed.discretize().B_d[0, 0, :3, 1:], 3))
delayed.input = Sinusoid(0.5, 1e5)
print(delayed.discretize())

# %%
# What is not a closed-form function of the step is the ``excitation``: held
# samples through the zero-order-hold map ``B_u``, white state noise as a
# factor of its per-step covariance (:func:`~cbadc.noise.positive_semidefinite`) on
# the ``x`` rows and as the comparators see it on the ``y`` rows, all drawn
# ahead of the loop. Comparator noise adds independent per-decision draws.
rz.state_covariance = 1e-12 * np.eye(3)
dz_rz = rz.discretize()
exc = np.zeros((1 << 10, dz_rz.N_x + rz.M, rz.J))
dz_rz.precompute_excitation(exc, np.random.default_rng(0))
rms = np.sqrt((exc[:, :, 0] ** 2).mean(0))
print(exc.shape, "rms per row:", " ".join(f"{r:.0e}" for r in rms))

# %%
# Tricks
# ------
# Three habits keep the module small. Doubling: ``trajectory`` needs
# ``e^{A i dt} x0`` for ``size`` steps and gets them in ``log2(size)`` passes,
# each one batched product ``X @ P^T`` appending the next block and one
# squaring ``P = P^2``; the recursion is ``size`` small products.
tone = Sinusoid(1.0, 1e5)
t0 = time.perf_counter()
X = tone.trajectory(1e-7, 1 << 20)
t1 = time.perf_counter()
P, x = expm(tone.A[0, 0] * 1e-7), [tone.x0[0]]
for _ in range((1 << 16) - 1):
    x.append(P @ x[-1])
t2 = time.perf_counter()
print(
    f"doubling, 2^20 steps: {1e3 * (t1 - t0):.1f} ms;  recursion, 2^16 steps: {1e3 * (t2 - t1):.1f} ms"
)

# %%
# The Kronecker product ``a * b`` of two sources has
# ``A = A_a (x) I + I (x) A_b``, so the eigenvalues add: a damped tone owns
# ``-1 / tau +- j w`` and stays a source, simulated exactly with no sampling.
damped = Sinusoid(1.0, 1e5) * Exp(2e-6)
print("eigenvalues:", np.linalg.eigvals(damped.A[0, 0]).round(0).tolist())
print("expected:   ", -1 / 2e-6 + 2j * np.pi * 1e5)

# %%
# One hold mechanism: a channel with zero ``C`` and ``D`` rows at step ``k``
# does not look, and the discretisation writes ``D_M = I`` on that channel so
# ``y[i] = v[i-1]`` reproduces the held decision inside the same product;
# ``with_trigger`` and rows zeroed by hand simulate identically. And every
# exponential and product runs on the stacked ``(J, K, ., .)`` arrays at
# once, so ``J`` frontends use one batched GEMM per step.
dz = ti.discretize()
print(
    "decision-to-comparator block at k = 1:\n",
    np.round(dz.ABCD[0, 1, dz.N_x :, dz.N_x :], 2),
)
print("one expm for the J frontends:", expm(mc.A[:, 0] * mc.dt).shape)

# %%
# Summary
# -------
rows = (
    [(name, w.N) for name, w in sources.items()]
    + [(name, f.N) for name, f in filters.items()]
    + [(name, w.N) for name, w in pulses.items()]
    + [(name, w.N) for name, w in composed.items()]
    + [("cascade of two integrators", chain.N), ("parallel poles", two_poles.N)]
)
print(f"{'system':<28}{'N':>3}")
for name, N in rows:
    print(f"{name:<28}{N:>3}")
