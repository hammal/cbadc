"""Control-bounded converter: model, discretisation and simulation.

Three layers, each a state space with leading batch and cyclic axes ``(J, K, ., .)``:

- ``J``: independent batch models.
- ``K``: cyclic clock phases.
- ``L``: external input channels in ``u``.
- ``M``: comparator outputs in ``y`` and decisions in ``v = Q(y)``.
- ``N``: frontend matrix states in ``x``.

- :class:`StateSpace`: ``x' = A x + B u``, ``y = C x + D u`` from ``x0``;
  :func:`cascade` and :func:`parallel` are the only algebra. Sources and
  filters are plain state spaces; :class:`DAC` adds the replay and window of a
  pulse; :class:`Samples` supplies predetermined amplitudes to the same pulse.
- :class:`Frontend`: its matrices give ``dx/dt = A x + B [u; v]`` and
  ``y = C x + D [u; v]`` has ``N`` states and ``L`` input channels in ``u``.
  The ``M`` comparator outputs ``y`` produce ``M`` decisions ``v = Q(y)``.
  It is clocked at ``dt``; non-idealities are attributes, synthesis constructors
  build the standard chains.
- :class:`Discretization`: the exact one-step maps
  ``[z[i]; y[i]] = ABCD[j, k] [z[i-1]; v[i-1]] + excitation[i]`` with every
  waveform cascaded in; :mod:`cbadc.simulation` runs them, together with the
  digital twin.

The digital twin replays the frontend matrices on the decisions alone,
``dx_hat/dt = A x_hat + B_M v`` from rest, read out as
``y_hat = C_hat x_hat + D_hat v``.
By linearity ``x = x_u + x_hat`` with ``x_u`` the input-only response. Thus
``-x_hat``, reconstructed from ``v``, estimates ``x_u`` with the bounded state
``x`` as its error. ``C_hat`` and ``D_hat`` select the reported quantity;
the generic defaults are ``-e_N`` and zero. Delta-sigma constructors instead
use the comparator readout, including its direct decision term. Estimators of
``u`` itself (such as Wiener reconstruction) operate on ``v`` outside the core.

The four axes of a matrix ``A[j, k, r, c]``: ``j`` selects one independent
frontend in a batch (a parameter sweep or Monte Carlo over mismatch),
simulated side by side; ``k`` is the cyclic step, the matrices repeat with
period ``K`` and ``A[:, i % K]`` applies at step ``i``, so a channel triggered
every other clock or a multiplexed input is one frontend with ``K > 1``; ``r``
and ``c`` are the ordinary matrix rows and columns (``A`` is ``N x N``, ``B`` is
``N x (L + M)``, ``C`` is ``M x N``, ``D`` is ``M x (L + M)``). An axis of length
one is shared: a plain ``(N, N)`` matrix is the same at every step and for every
batch member. Records are ``(size, ., J)``, time first and batch last. The quantiser has
``levels`` codes per channel, reads ``y`` in units of full scale and returns
``v`` in ``[-1, 1]``, mid-rise for even and mid-tread for odd ``levels``; the
DAC gain lives in ``B``. A channel whose ``C`` and ``D`` rows are zero at step
``k`` holds its previous decision; :meth:`Frontend.with_trigger` builds such rows.

>>> af, OSR = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
>>> af.input = Sinusoid(0.5, 1e5)
>>> sim = af.simulate(1 << 10)
>>> sim["v"].shape, np.unique(sim["v"]).tolist()
((1024, 3, 1), [-1, 1])
>>> sim["y_hat"].shape  # the twin's estimate of the last state of x_u, error x_N
(1024, 1, 1)
"""

from __future__ import annotations

import logging
import math
import operator
import warnings
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm, matrix_balance

from .noise import (
    combine_densities,
    discrete_process_noise_cov,
    flicker_samples,
    flicker_taps,
    k_B,
    kTC_rms,
    ota_density,
    per_state_intensity,
    positive_semidefinite,
    resistor_density,
)

if TYPE_CHECKING:
    from scipy.signal import lti

    from .estimator import WienerFilter

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# State space
# --------------------------------------------------------------------------- #


def _batched(a: ArrayLike, nd: int) -> np.ndarray:
    """``a`` with ``nd`` axes; a 3-D state-space matrix carries the batch axis.

    Parameters
    ----------
    a : array_like of float, shape (...)
        Input array with at most ``nd`` axes.
    nd : int
        Target number of axes, two or four.

    Returns
    -------
    ndarray of float, ndim nd
        Input with leading singleton axes added where needed.
    """
    a = np.asarray(a, dtype=float)
    if nd == 4 and a.ndim == 3:
        return a[:, np.newaxis]
    return a.reshape((1,) * (nd - a.ndim) + a.shape) if a.ndim < nd else a


def _cat(axis: int, *ms: np.ndarray) -> np.ndarray:
    """Concatenate after broadcasting the leading ``(J, K)`` axes.

    Parameters
    ----------
    axis : int
        Concatenation axis.
    *ms : ndarray of float, shape (J, K, rows, cols)
        Matrices with broadcastable leading axes.

    Returns
    -------
    ndarray of float, shape (J, K, rows, cols)
        Concatenated blocks with broadcast batch and phase axes.
    """
    lead = np.broadcast_shapes(*(m.shape[:2] for m in ms))
    return np.concatenate([np.broadcast_to(m, lead + m.shape[2:]) for m in ms], axis)


def _block(A: np.ndarray, X: np.ndarray, A_x: np.ndarray) -> np.ndarray:
    """``[[A, X], [0, A_x]]``.

    Parameters
    ----------
    A : ndarray, shape (J, K, N_a, N_a)
        Upper-left state block.
    X : ndarray, shape (J, K, N_a, N_x)
        Upper-right coupling block.
    A_x : ndarray, shape (J, K, N_x, N_x)
        Lower-right state block; leading axes broadcast.

    Returns
    -------
    ndarray of float, shape (J, K, N_a + N_x, N_a + N_x)
        Upper-triangular block state matrix.
    """
    zero = np.zeros((1, 1, A_x.shape[-2], A.shape[-1]))
    return _cat(-2, _cat(-1, A, X), _cat(-1, zero, A_x))


def _table(
    *rows: Sequence[Optional[np.ndarray]],
    labels: Sequence[Sequence[Optional[str]]] = (),
) -> str:
    """Rows of 2-D blocks as text, ``%.4g`` cells, rules between the rows.

    The blocks of a row share their height and the columns line up across
    rows; ``None`` leaves a block blank. ``labels`` names the blocks of the
    first row above the table and those of the last row below it. An entry
    below the rounding floor of its block, ``eps`` times the largest magnitude
    there, prints as ``0``.

    Parameters
    ----------
    *rows : sequence of 2-D ndarray or None
        Block rows of the formatted matrix.
    labels : sequence of sequences of str or None, optional
        Labels for the first and last block row.
        Defaults to ``()``.

    Returns
    -------
    str
        Formatted block matrix table.
    """
    widths = [b.shape[1] for b in rows[0]]

    def cell(v: float, floor: float) -> str:
        """Format one matrix value.

        Parameters
        ----------
        v : float
            Matrix entry.
        floor : float
            Absolute cutoff for printing zero.

        Returns
        -------
        str
            One formatted matrix entry.
        """
        return "0" if abs(v) <= floor else f"{v:.4g}"

    def cells(blocks: Sequence[Optional[np.ndarray]]) -> list[list[list[str]]]:
        """Format one row of matrix blocks.

        Parameters
        ----------
        blocks : sequence of 2-D ndarray or None
            Blocks with equal row count when present.

        Returns
        -------
        list of lists of lists of str
            Formatted cells grouped by row and block.
        """
        height = max(b.shape[0] for b in blocks if b is not None)
        floors = [
            np.finfo(float).eps * np.abs(b).max() if b is not None else 0.0
            for b in blocks
        ]
        return [
            [
                [cell(v, floor) for v in b[i]] if b is not None else [""] * w
                for b, w, floor in zip(blocks, widths, floors)
            ]
            for i in range(height)
        ]

    grid = [cells(blocks) for blocks in rows]
    width = max(len(c) for row in grid for line in row for block in line for c in block)
    spans = [w * (width + 1) - 1 for w in widths]

    def text(line: list[list[str]]) -> str:
        """Join formatted block cells into a table line.

        Parameters
        ----------
        line : list of lists of str
            One formatted matrix row grouped by block.

        Returns
        -------
        str
            One formatted table row.
        """
        return (
            "│ " + " │ ".join(" ".join(c.rjust(width) for c in b) for b in line) + " │"
        )

    def names(row: Sequence[Optional[str]]) -> str:
        """Center block names above or below the table.

        Parameters
        ----------
        row : sequence of str or None
            One optional name per matrix block.

        Returns
        -------
        str
            Centered block-name row.
        """
        return "  " + "   ".join((n or "").center(span) for n, span in zip(row, spans))

    rule = "├─" + "─┼─".join("─" * span for span in spans) + "─┤"
    body = f"\n{rule}\n".join("\n".join(text(line) for line in row) for row in grid)
    head, foot = (names(row).rstrip() for row in labels) if labels else ("", "")
    return "\n".join(line for line in (head, body, foot) if line)


class StateSpace:
    """``x' = A x + B u,  y = C x + D u`` from ``x0``, with leading batch and cyclic axes ``(J, K, ., .)``.

    Dimensions:

    - ``J``: independent batch columns; length-one axes are shared.
    - ``K``: cyclic phases.
    - ``L``: input channels in ``u``.
    - ``M``: output components in ``y``; in a :class:`Frontend`, these feed
      ``M`` comparators and produce ``M`` decisions ``v``.
    - ``N``: state coordinates in ``x``.

    The matrix shapes are ``A: (N, N)``, ``B: (N, L)``, ``C: (M, N)``, and
    ``D: (M, L)``. The initial state ``x0`` has shape ``(1 or J, N)``.
    ``A[j, k]`` applies at step ``i`` with ``k = i % K`` in batch column ``j``.
    :func:`cascade` and :func:`parallel` compose systems; ``a >> f`` feeds
    ``a`` into ``f``, ``a + b`` adds systems, and ``a * b`` multiplies two
    sources. ``print`` shows the ``[0, 0]`` slice as ``[[A, B], [C, D]]`` with
    ``x0`` beside ``B``
    when it is not zero, the rules marking ``N``, ``L`` and ``M``:

    >>> integrator = StateSpace(A=0.0, B=1.0, C=1.0, D=0.0)
    >>> print(cascade(integrator, integrator))
       A    B
    │ 0 1 │ 0 │
    │ 0 0 │ 1 │
    ├─────┼───┤
    │ 1 0 │ 0 │
       C    D

    Every excitation is such a system. A source (:func:`Constant`,
    :func:`Sinusoid`, :func:`Exp`, :func:`Poly`) is autonomous, ``B = D = 0``,
    and its output ``y = C e^{A t} x0`` runs from ``x0``; a filter (:func:`Pole`,
    :func:`Filter`, :func:`SOS`) is driven, and ``source >> filter`` is
    again a source. :class:`Discretization` cascades them into the frontend, so
    the matrix-exponential one-step map ``expm(A dt)`` has no time-stepping
    truncation error for these linear waveforms. Finite-precision error in
    ``expm`` and subsequent matrix operations remains. Held :class:`Samples`
    and noise enter as per-step excitations.

    ``A`` and ``x0`` carry the batch axis ``J`` so a tone may differ per column:

    >>> tone = Sinusoid([0.1, 0.5], 1e5)  # two batch columns (J=2), one amplitude each
    >>> rc = Constant(1.0) >> Pole(2e-9)  # a step settling through an RC
    >>> tone.J, rc.N, rc.autonomous
    (2, 2, True)

    Parameters
    ----------
    A, B, C, D : array_like of float
        State, input, output and feed-through matrices with trailing shapes
        ``(N, N)``, ``(N, L)``, ``(M, N)`` and ``(M, L)``. Leading axes may be
        absent, cyclic ``(K, rows, cols)``, or batched ``(J, K, rows, cols)``;
        a length-one axis broadcasts. Scalar zero is accepted for ``B`` and
        ``D``.
    x0 : array_like of float, shape (N,) or (J, N), optional
        Initial state of each batch column; defaults to zero.

    Returns
    -------
    StateSpace
        Initialized state-space system.
    """

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
        x0: Optional[ArrayLike] = None,
    ):
        scalar_zero_B = np.ndim(B) == 0 and not np.any(B)
        scalar_zero_D = np.ndim(D) == 0 and not np.any(D)
        self.A, self.B, self.C, self.D = (_batched(m, 4) for m in (A, B, C, D))
        self.x0 = np.zeros((1, self.N)) if x0 is None else _batched(x0, 2)
        if any(m.ndim != 4 for m in self.abcd) or self.x0.ndim != 2:
            raise ValueError("A, B, C, D need at most four axes and x0 at most two")
        N, M = self.N, self.C.shape[-2]
        L = max(self.B.shape[-1], self.D.shape[-1])
        if scalar_zero_B:
            self.B = np.zeros((1, 1, N, L))
        if scalar_zero_D:
            self.D = np.zeros((1, 1, M, L))
        if (
            self.A.shape[-2] != N
            or self.B.shape[-2:] != (N, L)
            or self.C.shape[-1] != N
            or self.D.shape[-2:] != (M, L)
            or self.x0.shape[-1] != N
        ):
            raise ValueError(
                "incompatible state-space dimensions: expected A (N, N), "
                "B (N, L), C (M, N), D (M, L), and x0 (J, N)"
            )
        try:
            np.broadcast_shapes(*(m.shape[:1] for m in self.abcd), self.x0.shape[:1])
        except ValueError as exc:
            raise ValueError("incompatible J batch dimensions") from exc

    @property
    def abcd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The four state-space matrices.

        Returns
        -------
        A, B, C, D : tuple of ndarray
            Matrices with shapes ``(J, K, N, N)``, ``(J, K, N, L)``,
            ``(J, K, M, N)``, and ``(J, K, M, L)`` respectively.
        """
        return self.A, self.B, self.C, self.D

    @property
    def N(self) -> int:
        """Number of state coordinates.

        Returns
        -------
        int
            State dimension ``N``.
        """
        return self.A.shape[-1]

    @property
    def K(self) -> int:
        """Combined cyclic period of the matrices.

        Returns
        -------
        int
            Number of cyclic phases ``K``.
        """
        return math.lcm(*(m.shape[1] for m in self.abcd))

    @property
    def J(self) -> int:
        """Broadcast batch width of the matrices and initial state.

        Returns
        -------
        int
            Number of parallel batch columns ``J``.
        """
        axes = [m.shape[:1] for m in self.abcd] + [self.x0.shape[:1]]
        return np.broadcast_shapes(*axes)[0]

    @property
    def autonomous(self) -> bool:
        """Whether the system has no driven inputs.

        Returns
        -------
        bool
            True when both ``B`` and ``D`` are zero.
        """
        return not np.any(self.B) and not np.any(self.D)

    def __str__(self) -> str:
        """Format the phase-zero, first-column matrix blocks.

        Returns
        -------
        str
            Labeled state-space block matrix.
        """
        A, B, C, D = (m[0, 0] for m in self.abcd)
        top, bottom, names = [A, B], [C, D], ["A", "B"]
        if np.any(self.x0):
            top, bottom, names = (
                top + [self.x0[0][:, None]],
                bottom + [None],
                names + ["x0"],
            )
        text = _table(top, bottom, labels=(names, ("C", "D")))
        return (
            text
            if self.K == self.J == 1
            else f"{text}\n[0, 0] of J={self.J}, K={self.K}"
        )

    def __add__(self, other: "StateSpace") -> "StateSpace":
        """Form the parallel sum of two systems.

        Parameters
        ----------
        other : StateSpace
            System with matching input and output widths.

        Returns
        -------
        StateSpace
            Parallel sum of the two systems.
        """
        return parallel(self, other)

    def __rshift__(self, f: "StateSpace") -> "StateSpace":
        """Cascade this system into another.

        Parameters
        ----------
        f : StateSpace
            Downstream system.

        Returns
        -------
        StateSpace
            Cascade of this system into ``f``.
        """
        return cascade(self, f)

    def __mul__(self, other: "StateSpace") -> "StateSpace":
        """Multiply two autonomous source outputs using a tensor-product state.

        Let ``⊗`` denote the Kronecker product. The combined state starts at
        ``x0_a ⊗ x0_b`` and evolves with
        ``A = A_a ⊗ I_b + I_a ⊗ A_b``. Its readout is
        ``C = C_a ⊗ C_b``, giving ``y(t) = y_a(t) y_b(t)``. This produces an
        autonomous source with ``N_a * N_b`` states; driven filters cannot be
        multiplied this way because their product is nonlinear in the input.

        The identity ``cos(a) cos(b) = (cos(a - b) + cos(a + b)) / 2`` explains
        the first example: a 300 kHz cosine times a 100 kHz cosine produces
        tones at their difference (200 kHz) and sum (400 kHz), each at half
        amplitude. Multiplying a cosine by a decaying exponential instead
        makes its amplitude decay over time.

        Parameters
        ----------
        other : StateSpace
            Autonomous source to multiply, with a broadcastable batch axis ``J``.

        >>> t = np.arange(8) * 1e-7
        >>> product = Sinusoid(1.0, 300e3) * Sinusoid(1.0, 100e3)
        >>> expected = 0.5 * np.cos(2 * np.pi * 200e3 * t) + 0.5 * np.cos(2 * np.pi * 400e3 * t)
        >>> product.N, np.allclose(product.value(t)[:, 0], expected)
        (4, True)
        >>> damped = Sinusoid(1.0, 1e6) * Exp(1e-6)  # a decaying tone
        >>> np.allclose(damped.value(t)[:, 0], np.cos(2 * np.pi * 1e6 * t) * np.exp(-t / 1e-6))
        True

        Returns
        -------
        StateSpace
            Autonomous product source with ``N_a * N_b`` states.
        """
        if not (self.autonomous and other.autonomous):
            raise ValueError("only sources multiply; a filter is linear in its input")
        J, N_a, N_b = max(self.J, other.J), self.N, other.N
        A_a, A_b = (np.broadcast_to(w.A[:, 0], (J, w.N, w.N)) for w in (self, other))
        x0_a, x0_b = (np.broadcast_to(w.x0, (J, w.N)) for w in (self, other))
        I_a, I_b = np.eye(N_a), np.eye(N_b)
        # kron(A_a, I_b) + kron(I_a, A_b) in every batch column
        A = np.einsum("jab,cd->jacbd", A_a, I_b) + np.einsum("ab,jcd->jacbd", I_a, A_b)
        return StateSpace(
            A.reshape(J, 1, N_a * N_b, N_a * N_b),
            np.zeros((N_a * N_b, 1)),
            np.kron(self.C[0, 0], other.C[0, 0]),
            0.0,
            np.einsum("ji,jk->jik", x0_a, x0_b).reshape(J, N_a * N_b),
        )

    def transfer_function(self, jw: ArrayLike) -> np.ndarray:
        """``H(jw) = C (jw I - A)^-1 B + D`` of a single-input single-output system.

        ``jw`` is a complex angular-frequency scalar or array-like of shape
        ``(size,)`` in rad/s; any higher-dimensional input is flattened. The
        return is a complex array of shape ``(size, J)``. The ``k = 0`` matrix
        slice is used.

        Parameters
        ----------
        jw : complex scalar or array_like, shape (size,)
            Angular frequencies ``1j * omega`` in rad/s; higher-dimensional
            arrays are flattened.

        >>> jw = 2j * np.pi * np.array([0.0, 1e6, 1e8])
        >>> np.abs(Pole(1 / (2 * np.pi * 1e6)).transfer_function(jw)[:, 0]).round(3).tolist()
        [1.0, 0.707, 0.01]

        Returns
        -------
        ndarray of complex, shape (size, J)
            Frequency response for each batch column at cyclic phase zero.
        """
        jw = np.asarray(jw, dtype=complex).reshape(-1)
        J, N = self.J, self.N
        A = np.broadcast_to(self.A[:, 0], (J, N, N))
        sI_A = jw[None, :, None, None] * np.eye(N) - A[:, None]  # (J, size, N, N)
        X = np.linalg.solve(sI_A, np.broadcast_to(self.B[0, 0], (J, jw.size, N, 1)))
        return ((self.C[0, 0] @ X)[..., 0, 0] + self.D[0, 0, 0, 0]).T

    def impulse(self) -> "StateSpace":
        """The impulse response ``h(t) = C e^{A t} B`` of a filter as a source: ``x0 = B``.

        >>> t = np.array([0.0, 1e-9, 2e-9])
        >>> Pole(1e-9).impulse().value(t)[:, 0].round(1).tolist()  # e^{-t / tau} / tau
        [1000000000.0, 367879441.2, 135335283.2]

        Returns
        -------
        StateSpace
            Autonomous source with initial state set from the filter input matrix.
        """
        if np.any(self.D):
            raise ValueError(
                "a feed-through D is a delta; no impulse response as a source"
            )
        x0 = np.broadcast_to(self.B[:, 0, :, 0], (self.J, self.N))
        return StateSpace(self.A, 0.0, self.C, 0.0, x0)

    def trajectory(self, dt: float, size: int) -> np.ndarray:
        """Zero-input state ``e^{A i dt} x0`` at ``t = i dt``, ``0 <= i < size``: shape ``(size, J, N)``.

        ``dt`` is the scalar time increment in seconds; ``size`` is the
        positive integer number of time samples. The result is a float array.

        The response from the start state alone, what a source *is*; for a
        filter it is the decay of ``x0`` with nothing applied, zero when ``x0``
        is, and ``impulse()`` sets ``x0 = B`` to make it the impulse response.

        The naive recursion is ``size`` small products::

            P = expm(A * dt)
            x[0] = x0
            for i in range(1, size):
                x[i] = P @ x[i - 1]

        Doubling computes the same values in ``log2(size)`` large ones: with
        ``x[0 .. 2^b - 1]`` known and ``P`` raised to ``2^b``, one product gives
        the next block ``x[2^b .. 2^(b+1) - 1] = P^(2^b) x[0 .. 2^b - 1]``, and
        squaring ``P`` prepares the next pass::

            before   X = [x0]                              P = e^{A dt}
            pass 1   X = [x0, P x0]           = x[0..1]    P = P^2
            pass 2   X = [x[0..1], P^2 x[0..1]] = x[0..3]  P = P^4
            pass 3   X = [x[0..3], P^4 x[0..3]] = x[0..7]  P = P^8

        ``X`` stores states as rows, so advancing them is ``X @ P^T``.

        >>> tone = Sinusoid(1.0, 1e5)
        >>> X = tone.trajectory(1e-7, 8)  # (size, J, N)
        >>> P = expm(tone.A[0, 0] * 1e-7)
        >>> np.allclose(X[3, 0], P @ P @ P @ tone.x0[0])
        True

        Parameters
        ----------
        dt : float
            Time increment in seconds.
        size : int
            Positive number of state samples.

        Returns
        -------
        ndarray of float, shape (size, J, N)
            Zero-input state trajectory.
        """
        J, N = self.J, self.N
        X = np.broadcast_to(self.x0, (J, N))[:, np.newaxis].astype(float)  # (J, i, N)
        P = expm(np.broadcast_to(self.A[:, 0], (J, N, N)) * dt)  # (J, N, N): e^{A dt}
        while X.shape[1] < size:  # doubling: X[2^b:2^(b+1)] = P^(2^b) X[:2^b]
            X = np.concatenate((X, X @ P.transpose(0, 2, 1)), axis=1)  # batched GEMM
            P = P @ P
        return X[:, :size].transpose(1, 0, 2)  # a view; each X[:, j] stays contiguous

    def value(self, t: np.ndarray) -> np.ndarray:
        """Zero-input output ``C e^{A t} x0`` at arbitrary times ``t``: shape ``(len(t), J)``.

        ``t`` is a one-dimensional float array of times in seconds. One
        ``expm`` is evaluated per time; ``trajectory`` is the state on the
        clock grid. The result is a float array.

        Parameters
        ----------
        t : array_like of float, shape (size,)
            Evaluation times in seconds.

        Returns
        -------
        ndarray of float, shape (size, J)
            Zero-input output at the requested times.
        """
        J, N = self.J, self.N
        A = np.broadcast_to(self.A[:, 0], (J, N, N))
        x0 = np.broadcast_to(self.x0, (J, N))
        t = np.asarray(t, dtype=float)
        P = expm(
            A[:, np.newaxis] * t[np.newaxis, :, np.newaxis, np.newaxis]
        )  # (J, t, N, N)
        return np.einsum("k,jtkl,jl->tj", self.C[0, 0, 0], P, x0)


def _stacked_x0(*ss: StateSpace) -> np.ndarray:
    """The start states of several systems side by side, ``(J, sum N)``.

    Parameters
    ----------
    *ss : StateSpace
        Systems with broadcastable batch axes ``J``.

    Returns
    -------
    ndarray of float, shape (J, sum N)
        Concatenated initial states of all systems.
    """
    J = max(s.x0.shape[0] for s in ss)
    return np.concatenate([np.broadcast_to(s.x0, (J, s.N)) for s in ss], axis=1)


def cascade(a: StateSpace, b: StateSpace, col: int = 0) -> StateSpace:
    """``a -> b``: the output of ``a`` replaces input ``col`` of ``b``, ``u_b[col] = y_a``.

    ``a`` and ``b`` are :class:`StateSpace` systems with broadcastable ``J``
    and ``K`` axes. ``a`` must have one output; ``col`` is an integer input
    column in ``[0, L_b)``. The result is a :class:`StateSpace`.

    With ``b_c = B_b[:, col]`` and ``d_c = D_b[:, col]``::

        x = [x_b; x_a]
        A = [[A_b, b_c C_a], [0, A_a]]    B = [[B_b with column col = b_c D_a], [B_a in column col, else 0]]
        C = [C_b, d_c C_a]                D = D_b with column col = d_c D_a

    >>> rc = cascade(Constant(1.0), Pole(0.5))  # A_a = 0, C_a = 1; A_b = -2, b_c = 2
    >>> rc.N, rc.A[0, 0].tolist(), rc.x0.tolist()  # x_b' = -2 x_b + 2 x_a: the pole driven by the constant
    (2, [[-2.0, 2.0], [0.0, 0.0]], [[0.0, 1.0]])

    Parameters
    ----------
    a : StateSpace
        Upstream system with one output.
    b : StateSpace
        Downstream system with a compatible batch and cyclic axes.
    col : int, optional
        Zero-based input column of ``b`` replaced by the output of ``a``.
        Defaults to ``0``.

    Returns
    -------
    StateSpace
        System formed by feeding ``a`` into one input of ``b``.
    """
    B_c, D_c = b.B[..., col : col + 1], b.D[..., col : col + 1]
    L_b = b.B.shape[-1]
    B_a = _cat(
        -1, np.zeros((1, 1, a.N, col)), a.B, np.zeros((1, 1, a.N, L_b - col - 1))
    )
    return StateSpace(
        _block(b.A, B_c @ a.C, a.A),
        _cat(-2, _cat(-1, b.B[..., :col], B_c @ a.D, b.B[..., col + 1 :]), B_a),
        _cat(-1, b.C, D_c @ a.C),
        _cat(-1, b.D[..., :col], D_c @ a.D, b.D[..., col + 1 :]),
        _stacked_x0(b, a),
    )


def parallel(a: StateSpace, b: StateSpace) -> StateSpace:
    """``a`` and ``b`` on the same input with their outputs summed, ``y = y_a + y_b``.

    Both arguments are :class:`StateSpace` systems with the same ``L`` and
    ``M`` and broadcastable ``J`` and ``K`` axes. The return is a
    :class:`StateSpace` with ``N_a + N_b`` states.

    Parameters
    ----------
    a, b : StateSpace
        Systems with equal input and output widths and broadcastable ``J``
        and ``K`` axes.

    ::

        x = [x_a; x_b]
        A = [[A_a, 0], [0, A_b]]    B = [B_a; B_b]
        C = [C_a, C_b]              D = D_a + D_b

    >>> two = parallel(Pole(0.5), Pole(1.0))  # A_a = -2, B_a = 2; A_b = -1, B_b = 1
    >>> two.A[0, 0].tolist(), two.B[0, 0].tolist(), two.C[0, 0].tolist()
    ([[-2.0, 0.0], [0.0, -1.0]], [[2.0], [1.0]], [[1.0, 1.0]])

    Returns
    -------
    StateSpace
        Parallel sum with ``N_a + N_b`` states.
    """
    zero = np.zeros((1, 1, a.N, b.N))
    return StateSpace(
        _block(a.A, zero, b.A),
        _cat(-2, a.B, b.B),
        _cat(-1, a.C, b.C),
        a.D + b.D,
        _stacked_x0(a, b),
    )


# --------------------------------------------------------------------------- #
# Sources, filters, DAC pulses and samples
# --------------------------------------------------------------------------- #


def Constant(c: ArrayLike = 1.0) -> StateSpace:
    """An autonomous state-space model of a constant source.

    It produces one value per batch column.

    State-space realization for each batch column ``j``::

        A     = [[0]]
        B     = [[0]]
        C     = [[1]]
        D     = [[0]]
        x0[j] = [c_j]

    Thus ``x_j(t) = c_j`` and ``y_j(t) = c_j``. The stored ``A``, ``B``,
    ``C``, and ``D`` each have shape ``(1, 1, 1, 1)`` and broadcast across
    the ``J`` rows of ``x0``, whose shape is ``(J, 1)``.

    Parameters
    ----------
    c : float or array_like, shape (J,), optional
        Constant output value in each batch column.
        Defaults to ``1.0``.

    Returns
    -------
    StateSpace
        Autonomous constant source, one value per batch column.
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    return StateSpace(0.0, 0.0, 1.0, 0.0, c[:, np.newaxis])


def Sinusoid(
    amplitude: ArrayLike, frequency: ArrayLike, phase: ArrayLike = 0.0
) -> StateSpace:
    """An autonomous state-space model of a sinusoidal source.

    Its output is ``amplitude cos(2 pi frequency t + phase)``, with ``t`` in
    seconds.

    ``amplitude`` is a float scalar or ``(J,)`` array in output units;
    ``frequency`` is a float scalar or ``(J,)`` array in Hz; ``phase`` is a
    float scalar or ``(J,)`` array in radians. All three broadcast to ``(J,)``.
    Returns a two-state-per-column :class:`StateSpace` source.

    State-space realization for each batch column ``j``, with
    ``omega_j = 2 pi frequency_j``::

        A_j   = [[0, -omega_j], [omega_j, 0]]
        B     = [[0], [0]]
        C     = [[1, 0]]
        D     = [[0]]
        x0[j] = [amplitude_j cos(phase_j), amplitude_j sin(phase_j)]

    The two states rotate at ``omega_j`` and ``C`` selects the cosine state.
    ``x0`` has shape ``(J, 2)``. The stored ``A`` has shape
    ``(J_A, 1, 2, 2)``, where ``J_A = 1`` when all frequencies agree and
    ``J_A = J`` otherwise. The shared ``B``, ``C``, and ``D`` have shapes
    ``(1, 1, 2, 1)``, ``(1, 1, 1, 2)``, and ``(1, 1, 1, 1)``.

    Parameters
    ----------
    amplitude : float or array_like, shape (J,)
        Peak amplitude in output units.
    frequency : float or array_like, shape (J,)
        Frequency in Hz.
    phase : float or array_like, shape (J,)
        Initial phase in radians; defaults to zero.

    Returns
    -------
    StateSpace
        Autonomous sinusoidal source with two states per column.
    """
    a, f, p = np.broadcast_arrays(
        *(
            np.atleast_1d(np.asarray(v, dtype=float))
            for v in (amplitude, frequency, phase)
        )
    )
    w = 2.0 * np.pi * (f[:1] if np.all(f == f[0]) else f)  # shared A when one frequency
    A = np.zeros((w.size, 2, 2))
    A[:, 0, 1], A[:, 1, 0] = -w, w
    x0 = np.stack((a * np.cos(p), a * np.sin(p)), axis=1)
    return StateSpace(A, [[0.0], [0.0]], [[1.0, 0.0]], 0.0, x0)


def Exp(tau: float) -> StateSpace:
    """An autonomous state-space model of ``exp(-t / tau)``.

    This describes a switched-capacitor DAC discharge. ``tau`` is a positive,
    finite scalar time constant in seconds; the model has one state.

    State-space realization::

        A     = [[-1/tau]]
        B     = [[0]]
        C     = [[1]]
        D     = [[0]]
        x0[0] = [1]

    Thus ``x(t) = y(t) = exp(-t / tau)``. The four matrices each have shape
    ``(1, 1, 1, 1)`` and ``x0`` has shape ``(1, 1)``.

    Parameters
    ----------
    tau : float
        Positive, finite decay time constant in seconds.

    >>> np.round(Exp(1e-9).value(np.array([0.0, 1e-9]))[:, 0], 3).tolist()
    [1.0, 0.368]

    Returns
    -------
    StateSpace
        Autonomous exponential source with one state.
    """
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")
    return StateSpace(-1.0 / tau, 0.0, 1.0, 0.0, [[1.0]])


def Poly(coefficients: ArrayLike) -> StateSpace:
    """An autonomous state-space model of a polynomial in time.

    For ``N`` coefficients, the output is
    ``y(t) = c_0 + c_1*t + ... + c_(N-1)*t**(N-1)``: the powers run from
    ``k = 0`` through ``k = N-1``. For example, ``Poly([1, -1/tp])`` gives
    ``y(t) = 1 - t/tp``. It starts at 1, reaches 0 at ``t = tp``, and
    continues below 0 afterward.

    ``coefficients`` is a one-dimensional float array ``(N,)`` ordered from
    constant to highest power; ``t`` is in seconds. Returns an ``N``-state
    :class:`StateSpace` source.

    State-space realization, with ``c_k = coefficients[k]``::

        A     = [[0, 1, 0, ...], [0, 0, 1, ...], ..., [0, ..., 0]]
        B     = [[0], ..., [0]]
        C     = [[1, 0, ..., 0]]
        D     = [[0]]
        x0[0] = [0! c_0, 1! c_1, ..., (N-1)! c_(N-1)]

    ``A`` makes each state's derivative equal the next state; the last
    state is constant and ``C`` reads the first. With ``N = 3``, the
    initial states are ``[c_0, c_1, 2*c_2]``. The last state stays at
    ``2*c_2``, the middle becomes ``c_1 + 2*c_2*t``, and the output becomes
    ``c_0 + c_1*t + c_2*t**2``. For degree ``k``, the ``k!`` in the initial
    state cancels the ``1/k!`` introduced by integrating ``k`` times.
    The stored ``A``, ``B``, ``C``, ``D``, and ``x0`` have shapes ``(1, 1, N, N)``,
    ``(1, 1, N, 1)``, ``(1, 1, 1, N)``, ``(1, 1, 1, 1)``, and ``(1, N)``.

    Parameters
    ----------
    coefficients : array_like of float, shape (N,)
        Polynomial coefficients from constant to highest power.

    >>> Poly([1.0, -1e9]).value(np.array([0.0, 0.5e-9]))[:, 0].round(12).tolist()
    [1.0, 0.5]

    Returns
    -------
    StateSpace
        Autonomous polynomial source with ``N`` states.
    """
    c = np.asarray(coefficients, dtype=float)
    N = c.size
    x0 = c * np.array([math.factorial(k) for k in range(N)])
    return StateSpace(np.eye(N, k=1), np.zeros((N, 1)), np.eye(N)[:1], 0.0, x0[None])


def Pole(tau: float) -> StateSpace:
    """The one-state filter ``1 / (1 + s tau)``.

    Parameters
    ----------
    tau : float
        Positive, finite pole time constant in seconds.

    Returns
    -------
    StateSpace
        Single-pole filter with one state.
    """
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")
    return Filter([[-1.0 / tau]], [[1.0 / tau]], [[1.0]])


def Filter(
    A: ArrayLike, B: ArrayLike, C: ArrayLike, D: ArrayLike = 0.0, balance: bool = True
) -> StateSpace:
    """A single-input single-output filter in state-space form.

    For transfer-function coefficients ``b`` and ``a`` in descending powers
    of ``s``, SciPy computes ``A, B, C, D``. For example, with a two-pole
    analog Butterworth low-pass::

        >>> from scipy import signal
        >>> b, a = signal.butter(2, 2 * np.pi * 1e6, analog=True, output="ba")
        >>> A, B, C, D = signal.tf2ss(b, a)
        >>> f = Filter(A, B, C, float(D[0, 0]))
        >>> f.N
        2

    By default, ``balance`` rescales the realization for better numerical
    conditioning without changing its transfer function.

    Parameters
    ----------
    A : array_like of float, shape (N, N)
        Filter state matrix.
    B : array_like of float, shape (N,) or (N, 1)
        Input matrix or vector.
    C : array_like of float, shape (N,) or (1, N)
        Output matrix or vector.
    D : float
        Direct feed-through gain, default zero.
    balance : bool
        Balance the realization for numerical conditioning, default true.

    Returns
    -------
    StateSpace
        Balanced or unbalanced single-input, single-output filter.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    N = A.shape[0]
    B = np.asarray(B, dtype=float).reshape(N, 1)
    C = np.asarray(C, dtype=float).reshape(1, N)
    if balance and N > 1:
        A, T = matrix_balance(A)
        B, C = np.linalg.solve(T, B), C @ T
    return StateSpace(A, B, C, D)


def SOS(sos: ArrayLike) -> StateSpace:
    """A single-input single-output filter from analog second-order sections.

    Each row ``[b0, b1, b2, a0, a1, a2]`` represents
    ``(b0*s**2 + b1*s + b2) / (a0*s**2 + a1*s + a2)``. SciPy can design
    these sections directly; for example, a fourth-order Butterworth low-pass::

        >>> from scipy import signal
        >>> sos = signal.butter(4, 2 * np.pi * 1e6, analog=True, output="sos")
        >>> f = SOS(sos)
        >>> f.N
        4

    Parameters
    ----------
    sos : array_like of float, shape (sections, 6) or (6,)
        Analog numerator and denominator coefficients in descending powers of ``s``.

    Returns
    -------
    StateSpace
        Single-input, single-output cascade of analog sections.
    """
    from scipy.signal import tf2ss

    f = None
    for row in np.atleast_2d(np.asarray(sos, dtype=float)):
        b, a = row[:3], row[3:]
        lead = int(np.argmax(a != 0))  # a first-order section carries leading zeros
        A, B, C, D = tf2ss(np.trim_zeros(b[lead:], "f"), a[lead:])
        section = Filter(A, B, C, float(np.squeeze(D)), balance=False)
        f = section if f is None else f >> section
    return Filter(
        f.A[0, 0], f.B[0, 0], f.C[0, 0], float(f.D[0, 0, 0, 0])
    )  # balance the whole cascade


class DAC(StateSpace):
    """A clocked DAC pulse built from an autonomous state-space source.

    At each clock edge, the decision scales the ``shape``'s initial state
    and restarts its states. Between edges the pulse evolves autonomously;
    an optional settling ``filter`` has states that carry across edges.
    ``window = (delay, width)`` gates the shape in clock-period units.
    The DAC coupling gain remains in the frontend's ``B``.

    >>> nrz = DAC(Constant(1.0))  # constant pulse over the whole period
    >>> rz = nrz.with_window(0.25, 0.5)  # return to zero
    >>> settling = nrz >> Pole(2e-9)  # add an RC filter
    >>> nrz.autonomous, rz.window, settling.restart.tolist()
    (True, (0.25, 0.5), [False, True])

    Parameters
    ----------
    shape : StateSpace
        Autonomous pulse source with one output, ``J = K = 1``.
    filter : StateSpace, optional
        Driven single-input settling filter.
        Defaults to ``None``.
    window : tuple of float, shape (2,), optional
        Nonnegative delay and positive width in clock periods; defaults to
        the full period.

    Returns
    -------
    DAC
        Initialized DAC pulse.
    """

    def __init__(
        self,
        shape: StateSpace,
        filter: Optional[StateSpace] = None,
        window: Optional[tuple[float, float]] = None,
    ):
        if not shape.autonomous or shape.K != 1 or shape.J != 1:
            raise ValueError("a DAC shape is a source without cyclic or batch axis")
        if window is not None and (window[0] < 0 or window[1] <= 0):
            raise ValueError("window needs delay >= 0 and width > 0")
        whole = shape if filter is None else cascade(shape, filter)
        super().__init__(*whole.abcd, whole.x0)
        self.shape, self.filter, self.window = shape, filter, window

    @property
    def restart(self) -> np.ndarray:
        """``(N,)`` bool: the shape states, replayed at each edge; the filter states carry over.

        Returns
        -------
        ndarray of bool, shape (N,)
            Mask of states restarted at each clock edge.
        """
        return np.arange(self.N) >= self.N - self.shape.N

    def __rshift__(self, f: StateSpace) -> "DAC":
        """Append a settling filter to this DAC pulse.

        Parameters
        ----------
        f : StateSpace
            Driven single-input settling filter.

        Returns
        -------
        DAC
            Pulse with the downstream settling filter appended.
        """
        through = f if self.filter is None else cascade(self.filter, f)
        return DAC(self.shape, through, self.window)

    def with_window(self, delay: float, width: float) -> "DAC":
        """Gate the shape to ``[delay, delay + width)`` clock periods.

        ``delay`` is a nonnegative float and ``width`` a positive float;
        returns a new :class:`DAC`.

        Parameters
        ----------
        delay : float
            Nonnegative delay in clock periods.
        width : float
            Positive pulse width in clock periods.

        Returns
        -------
        DAC
            Pulse gated to the requested window.
        """
        return DAC(self.shape, self.filter, (float(delay), float(width)))


class Samples(DAC):
    """Predetermined DAC drive values, one per clock period.

    Each value scales a restarted autonomous ``shape`` at its clock edge.
    The default ``Constant(1)`` shape holds the value for one period. As with
    :class:`DAC`, ``with_window`` gates the pulse and ``>>`` adds a settling
    filter whose states persist between edges. The sequence repeats when the
    simulation is longer than the stored values.

    Parameters
    ----------
    values : array_like of float, shape (size,) or (size, J)
        Repeating pulse amplitudes, with one column per parallel simulation.
    shape : StateSpace, optional
        Autonomous single-output pulse shape; defaults to ``Constant(1)``.

    >>> s = Samples(np.random.default_rng(0).uniform(-0.5, 0.5, (64, 2)))
    >>> s.values.shape, s.J, s.N
    ((64, 2), 2, 1)
    >>> Samples([0.5, -0.5]).held(6)[:, 0].tolist()
    [0.5, -0.5, 0.5, -0.5, 0.5, -0.5]
    >>> (Samples([1.0, 0.0]) >> Pole(1e-7)).N
    2

    Returns
    -------
    Samples
        Initialized repeating sample stream.
    """

    def __init__(self, values: ArrayLike, shape: Optional[StateSpace] = None):
        v = np.asarray(values, dtype=float)
        if v.ndim not in (1, 2):
            raise ValueError("samples need shape (size,) or (size, J)")
        self.values = v[:, np.newaxis] if v.ndim == 1 else v
        if self.values.shape[0] == 0:
            raise ValueError("an input has no samples")
        super().__init__(Constant(1.0) if shape is None else shape)

    @property
    def J(self) -> int:
        """Number of parallel pulse or sampled-input columns.

        Returns
        -------
        int
            Broadcast batch width of the pulse and stored sequence.
        """
        return max(super().J, self.values.shape[1])

    def _same_pulse(self, other: "Samples") -> bool:
        """Whether two sample streams have the same pulse response.

        Parameters
        ----------
        other : Samples
            Pulse response to compare with this one.

        Returns
        -------
        bool
            True when the shape, filter and window are identical.
        """
        if self.window != other.window or (self.filter is None) != (
            other.filter is None
        ):
            return False
        pairs = [(self.shape, other.shape)]
        if self.filter is not None:
            pairs.append((self.filter, other.filter))
        return all(
            all(np.array_equal(a, b) for a, b in zip(left.abcd, right.abcd))
            and np.array_equal(left.x0, right.x0)
            for left, right in pairs
        )

    def __add__(self, other: "Samples") -> "Samples":
        """Add drive sequences when both pass through the same pulse.

        Parameters
        ----------
        other : Samples
            Sequence with matching pulse response and broadcastable values.

        Returns
        -------
        Samples
            Sum of the drive values through their common pulse.
        """
        if not isinstance(other, Samples):
            return NotImplemented
        if not self._same_pulse(other):
            raise ValueError("sample streams need the same pulse to be added")
        result = copy(self)
        result.values = self.values + other.values
        return result

    def __mul__(self, other: "Samples | float") -> "Samples":
        """Multiply drive values, or two unfiltered pulses in one period.

        Parameters
        ----------
        other : Samples or float
            Scalar gain or another unfiltered sample stream with the same window.

        Returns
        -------
        Samples
            Product of the sampled waveforms.
        """
        if np.isscalar(other):
            result = copy(self)
            result.values = self.values * float(other)
            return result
        if not isinstance(other, Samples):
            return NotImplemented
        if (
            self.filter is not None
            or other.filter is not None
            or self.window != other.window
            or sum(self.window or (0.0, 1.0)) > 1.0
        ):
            raise ValueError(
                "sample products need unfiltered pulses in the same period"
            )
        result = Samples(self.values * other.values, self.shape * other.shape)
        return result if self.window is None else result.with_window(*self.window)

    __rmul__ = __mul__

    def __rshift__(self, f: StateSpace) -> "Samples":
        """Pass held samples through a filter.

        Parameters
        ----------
        f : StateSpace
            Driven single-input sample filter.

        Returns
        -------
        Samples
            Sample stream with the downstream filter appended.
        """
        through = f if self.filter is None else cascade(self.filter, f)
        new = copy(self)
        DAC.__init__(new, self.shape, through, self.window)
        return new

    def with_window(self, delay: float, width: float) -> "Samples":
        """Return the sequence with its pulse gated to a clock window.

        Parameters
        ----------
        delay : float
            Nonnegative delay in clock periods.
        width : float
            Positive pulse width in clock periods.

        Returns
        -------
        Samples
            Sequence with the requested pulse window.
        """
        new = copy(self)
        DAC.__init__(new, self.shape, self.filter, (float(delay), float(width)))
        return new

    def held(self, size: int, log: bool = True) -> np.ndarray:
        """Repeat the drive values, not the shaped analog waveform.

        ``size`` is the integer number of requested clock periods; ``log`` is
        a bool controlling the wrap notice.

        Parameters
        ----------
        size : int
            Number of requested clock periods.
        log : bool, optional
            Log when the sample sequence wraps.
            Defaults to ``True``.

        Returns
        -------
        ndarray of float, shape (size, J)
            Pulse drive values, wrapping the stored sequence as necessary.
        """
        period = self.values.shape[0]
        if size > period:
            if log:
                logger.info(
                    "%d samples cycled %.2f times over %d steps",
                    period,
                    size / period,
                    size,
                )
            return self.values[np.arange(size) % period]
        return self.values[:size]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _zoh(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """``expm([[A, B], [0, 0]] dt)`` split into zero-order-hold maps.

    Parameters
    ----------
    A : ndarray, shape (..., N, N)
        State matrix in inverse seconds.
    B : ndarray, shape (..., N, L)
        Input matrix; leading axes broadcast with ``A``.
    dt : float
        Clock period in seconds.

    Returns
    -------
    A_d : ndarray of float, shape (..., N, N)
        One-step state transition.
    B_d : ndarray of float, shape (..., N, L)
        Zero-order-hold input map.
    """
    N, L = B.shape[-2:]
    Z = np.zeros(A.shape[:-2] + (N + L, N + L))
    Z[..., :N, :N], Z[..., :N, N:] = A, B
    E = expm(Z * dt)
    return E[..., :N, :N], E[..., :N, N:]


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #


class Frontend(StateSpace):
    """A control-bounded converter.

    Dimensions:

    - ``J``: parallel batch members.
    - ``K``: cyclic clock phases.
    - ``L``: external input channels in ``u``.
    - ``M``: comparator inputs in ``y``, decisions in ``v``, and DACs.
    - ``N``: frontend matrix states in ``x``.
    - ``M_hat``: digital-twin readout channels; one by default.

    The frontend matrices give ``dx/dt = A x + B [u; v]`` and ``y = C x + D [u; v]``,
    with ``B = [B_L | B_M]`` and ``D = [D_L | D_M]``. ``v = Q(y)`` is clocked at ``dt``. Assignment
    normalises: matrices become ``(J, K, ., .)``, one input or :class:`DAC` is
    repeated over the ``L`` inputs or ``M`` DACs. Non-idealities are attributes:
    ``state_covariance`` has shape ``(J, K, N, N)`` (two-sided intensity, or
    per-step covariance when ``discrete``). ``output_covariance`` is the
    ``(M, M)`` comparator-input noise covariance per decision;
    ``comparator_input_offset`` sets comparator thresholds,
    ``state_min``/``state_max`` (swing) and ``slew_rate``. The digital-twin
    readout is ``y_hat = C_hat x_hat + D_hat v``; see the module overview.

    >>> A = [[0.0, 0.0], [1e6, 0.0]]  # two integrators in a chain
    >>> B = [[1e6, -1e6, 0.0], [0.0, 0.0, -1e6]]  # [B_L | B_M]: input, then one DAC per control
    >>> af = Frontend(A, B, np.eye(2), dt=1e-7)
    >>> af.J, af.K, af.L, af.M, af.N, af.A.shape
    (1, 1, 1, 2, 2, (1, 1, 2, 2))
    >>> print(af)
    Frontend(J=1, K=1, L=1, M=2, N=2, dt=1.000e-07, continuous)
            A                  B
    │      0      0 │  1e+06 -1e+06      0 │
    │  1e+06      0 │      0      0 -1e+06 │
    ├───────────────┼──────────────────────┤
    │      1      0 │      0      0      0 │
    │      0      1 │      0      0      0 │
            C                  D

    Parameters
    ----------
    A : array_like, shape (N, N), (K, N, N), or (J, K, N, N)
        Frontend state matrix.
    B : array_like, shape (N, L + M), (K, N, L + M), or (J, K, N, L + M)
        Input columns followed by control columns.
    C : array_like, shape (M, N), (K, M, N), or (J, K, M, N)
        Comparator input matrix.
    D : array_like, shape (M, L + M), (K, M, L + M), or (J, K, M, L + M), optional
        Direct feed-through; defaults to zero. Leading ``J`` and ``K`` axes of
        all matrices broadcast independently.
    dt : float
        the clock period in seconds.
    levels : int or array_like, shape (M,)
        Quantiser codes per channel, default 2.
    input : StateSpace, Samples, or sequence of length L, optional
        Input sources, default zero; one object is reused for all ``L`` inputs.
    dac : DAC or sequence of length M, optional
        DAC pulses; defaults to an equal-area RZ pulse on ``[T/4, 3T/4]``
        with amplitude 2 for continuous time, or immediate NRZ for discrete
        time. One object is reused for all
        ``M`` controls.
    discrete : bool, optional
        ``A`` and ``B`` are already the one-step maps of a discrete-time system.
        Defaults to ``False``.
    C_hat : array_like, shape (M_hat, N), (K, M_hat, N), or (J, K, M_hat, N), optional
        Digital-twin readout; when omitted, the simulator uses ``-e_N``.
        Defaults to ``None``.
    D_hat : array_like, shape (M_hat, M), (K, M_hat, M), or (J, K, M_hat, M), optional
        Direct readout from the held decisions; defaults to zero.
    Returns
    -------
    Frontend
        Initialized clocked converter.
    """

    _per_state = dict(state_min=-np.inf, state_max=np.inf, slew_rate=np.inf)

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: Optional[ArrayLike] = None,
        *,
        dt: float,
        levels: ArrayLike = 2,
        input: Optional[StateSpace | Samples | Sequence] = None,
        dac: Optional[DAC | Sequence[DAC]] = None,
        discrete: bool = False,
        C_hat: Optional[ArrayLike] = None,
        D_hat: Optional[ArrayLike] = None,
    ):
        self.A, self.B, self.C, self.C_hat, self.D_hat = A, B, C, C_hat, D_hat
        N, M, L = self.N, self.M, self.L
        self.D = np.zeros((M, L + M)) if D is None else D
        matrices = self.abcd + tuple(
            matrix for matrix in (self.C_hat, self.D_hat) if matrix is not None
        )
        if any(m.ndim != 4 for m in matrices):
            raise ValueError("A, B, C, D, C_hat, and D_hat need at most four axes")
        if (
            self.A.shape[-2:] != (N, N)
            or self.B.shape[-2] != N
            or self.C.shape[-1] != N
            or (self.C_hat is not None and self.C_hat.shape[-1] != N)
        ):
            raise ValueError("A must be (N, N), B (N, L + M), C and C_hat (., N)")
        if self.D.shape[-2:] != (M, L + M) or L < 0:
            raise ValueError("D must be (M, L + M)")
        M_hat = 1 if self.C_hat is None else self.C_hat.shape[-2]
        if self.D_hat is not None and self.D_hat.shape[-2:] != (M_hat, M):
            raise ValueError("D_hat must be (M_hat, M)")
        try:
            np.broadcast_shapes(*(m.shape[:1] for m in matrices))
        except ValueError as exc:
            raise ValueError("incompatible J batch dimensions") from exc
        if not dt > 0:
            raise ValueError("dt must be positive")
        self.dt, self.discrete = float(dt), bool(discrete)
        self.x0 = np.zeros((1, N))
        self.levels, self.input, self.dac = levels, input, dac
        self.state_covariance = None
        self._flicker = {}  # node -> (corner frequency [Hz], spectral exponent)
        self.output_covariance = None
        self.comparator_input_offset = np.zeros(M)
        for name, fill in self._per_state.items():
            setattr(self, name, np.full(N, fill))

    def __setattr__(self, name: str, value) -> None:
        """Normalize and validate mutable converter attributes.

        Parameters
        ----------
        name : str
            Attribute to assign, including matrix, input, noise, or limit fields.
        value : array_like, StateSpace, Samples, DAC, sequence, or scalar
            New value. Matrix fields use trailing state-space dimensions and
            optional ``(J, K)`` axes. State covariance accepts ``(N, N)``,
            ``(K, N, N)``, or ``(J, K, N, N)`` and stores four axes;
            comparator-input covariance is symmetric ``(M, M)``
            arrays, and per-state limits broadcast to ``(N,)``.

        Returns
        -------
        None
            Assigns the normalized attribute in place.
        """
        if name in ("a2", "a3"):
            # Legacy nonlinear settings must fail rather than be silently ignored.
            raise AttributeError("polynomial a2/a3 nonlinearity is not supported")
        if name == "output_offset":
            raise AttributeError("use comparator_input_offset for comparator offsets")
        if name in ("clock_jitter", "comparator_jitter"):
            raise AttributeError("clock jitter is not supported")
        if name in ("A", "B", "C", "D") or (
            name in ("C_hat", "D_hat") and value is not None
        ):  # (., .), cyclic (K, ., .) or (J, K, ., .)
            value = np.asarray(value, dtype=float)
            value = value[np.newaxis] if value.ndim == 3 else _batched(value, 4)
        elif name in ("input", "dac"):
            count = self.L if name == "input" else self.M
            if value is None:
                value = (
                    Constant(0.0)
                    if name == "input"
                    else DAC(
                        Constant(1.0 if self.discrete else 2.0),
                        window=None if self.discrete else (0.25, 0.5),
                    )
                )
            value = (
                [value] * count
                if isinstance(value, (StateSpace, Samples))
                else list(value)
            )
            if len(value) != count:
                raise ValueError(f"{name} needs {count} entries")
            if name == "dac" and not all(
                isinstance(w, DAC) and not isinstance(w, Samples) for w in value
            ):
                raise ValueError("a DAC is a DAC(shape) pulse")
            if name == "input" and not all(
                isinstance(w, Samples) or (w.autonomous and w.K == 1) for w in value
            ):
                raise ValueError("an input is a source (B = D = 0, K = 1) or Samples")
        elif name == "levels":
            value = np.broadcast_to(np.asarray(value, dtype=int), (self.M,)).copy()
            if (value < 2).any():
                raise ValueError("levels must be >= 2 for each of the M channels")
        elif name == "state_covariance" and value is not None:
            value = np.asarray(value, dtype=float)
            if value.ndim == 2:
                value = value[None, None]
            elif value.ndim == 3:
                value = value[None]
            if (
                value.ndim != 4
                or not all(value.shape[:2])
                or value.shape[-2:] != (self.N, self.N)
                or not np.allclose(value, value.swapaxes(-1, -2))
            ):
                raise ValueError("state_covariance must be symmetric (J, K, N, N)")
            try:
                np.broadcast_shapes(
                    (self.J_frontend,),
                    *(
                        matrix.shape[:1]
                        for matrix in (self.C_hat, self.D_hat)
                        if matrix is not None
                    ),
                    *((w.J,) for w in self.input),
                    *((w.J,) for w in self.dac),
                    value.shape[:1],
                )
            except ValueError as exc:
                raise ValueError(
                    "incompatible J batch dimensions for state_covariance"
                ) from exc
            value = value.copy()
        elif name == "output_covariance" and value is not None:
            value = np.asarray(value, dtype=float)
            if value.shape != (self.M, self.M) or not np.allclose(value, value.T):
                raise ValueError(
                    f"output_covariance must be a symmetric ({self.M}, {self.M}) matrix"
                )
        elif name == "comparator_input_offset" or name in self._per_state:
            size = self.M if name == "comparator_input_offset" else self.N
            value = np.broadcast_to(np.asarray(value, dtype=float), (size,)).copy()
        object.__setattr__(self, name, value)

    # -- shape ----------------------------------------------------------------
    @property
    def M(self) -> int:
        """Number of comparator channels.

        Returns
        -------
        int
            Comparator-input row count ``M``.
        """
        return self.C.shape[-2]

    @property
    def L(self) -> int:
        """Number of external input channels.

        Returns
        -------
        int
            Input-column count ``L`` before the ``M`` control columns.
        """
        return self.B.shape[-1] - self.M

    @property
    def K(self) -> int:
        """Combined period of the matrices, twin and state noise.

        Returns
        -------
        int
            Least common multiple of their cyclic periods.
        """
        K = super().K
        if self.C_hat is not None:
            K = math.lcm(K, self.C_hat.shape[1])
        if self.D_hat is not None:
            K = math.lcm(K, self.D_hat.shape[1])
        covariance = getattr(self, "state_covariance", None)
        if covariance is not None:
            K = math.lcm(K, covariance.shape[1])
        return K

    @property
    def J_frontend(self) -> int:
        """Parallel frontend matrix sets along the ``J`` batch axis.

        Returns
        -------
        int
            Number of distinct frontend matrix sets.
        """
        return super().J

    @property
    def J(self) -> int:
        """Batch width: matrices, sources, DACs, twin and state noise.

        Returns
        -------
        int
            Total simulation batch width.
        """
        return max(
            [self.J_frontend]
            + (
                [self.state_covariance.shape[0]]
                if self.state_covariance is not None
                else []
            )
            + [
                matrix.shape[0]
                for matrix in (self.C_hat, self.D_hat)
                if matrix is not None
            ]
            + [w.J for w in self.input]
            + [w.J for w in self.dac]
        )

    def __getitem__(self, j: int) -> "Frontend":
        """Select one complete batch member, retaining every cyclic phase.

        Each leading ``J`` axis is selected independently; a length-one axis
        remains shared. State noise selects its matching member too. The
        selected converter is a deep copy. Clock phase
        ``k`` is not an index here: it advances as ``i % K`` during simulation.

        Parameters
        ----------
        j : int
            Batch index in ``[-J, J)``.

        Returns
        -------
        Frontend
            Independent converter with ``J = 1`` and the original ``K``.

        >>> af, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
        >>> af.input = Sinusoid([0.1, 0.5], 1e5)
        >>> af[1].J, af[1].K, af[1].input[0].x0.shape
        (1, 1, (1, 2))
        """
        if isinstance(j, tuple):
            raise TypeError("use frontend[j] for a batch member; clock phase is i % K")
        j = range(self.J)[operator.index(j)]
        new = deepcopy(self)

        def select(system: StateSpace) -> None:
            for name, matrix in zip("ABCD", system.abcd):
                setattr(system, name, matrix[[j % matrix.shape[0]]])
            system.x0 = system.x0[[j % system.x0.shape[0]]]
            if isinstance(system, Samples):
                system.values = system.values[:, [j % system.values.shape[1]]]
            if isinstance(system, DAC) and system.filter is not None:
                select(system.filter)

        select(new)
        if new.C_hat is not None:
            new.C_hat = new.C_hat[[j % new.C_hat.shape[0]]]
        if new.D_hat is not None:
            new.D_hat = new.D_hat[[j % new.D_hat.shape[0]]]
        if new.state_covariance is not None:
            new.state_covariance = new.state_covariance[
                [j % new.state_covariance.shape[0]]
            ]
        for source in new.input:
            select(source)
        for pulse in new.dac:
            select(pulse)
        return new

    @property
    def fs(self) -> float:
        """Sampling frequency.

        Returns
        -------
        float
            Reciprocal clock period in Hz.
        """
        return 1.0 / self.dt

    @property
    def is_discrete_time(self) -> bool:
        """Whether the frontend matrices are already one-step maps.

        Returns
        -------
        bool
            True for a discrete-time model.
        """
        return self.discrete

    def input_referred_covariance(self, input_psd: ArrayLike) -> np.ndarray:
        """State intensity ``B_L (S / 2) B_L^T`` of a one-sided input PSD.

        Here ``B_L = B[0, 0, :, :L]`` is the external-input block for the
        first batch column and clock phase; ``S`` is ``input_psd``.

        Parameters
        ----------
        input_psd : array_like, shape (L, L)
            One-sided input noise PSD ``S`` in V²/Hz.

        Returns
        -------
        ndarray of float, shape (N, N)
            Continuous-time state-noise intensity referred from the input PSD.
        """
        S = np.asarray(input_psd, dtype=float).reshape(self.L, self.L)
        B_L = self.B[0, 0, :, : self.L]
        return B_L @ (S / 2.0) @ B_L.T

    def __str__(self) -> str:
        """Summary line, then ``[[A, B], [C, D]]`` if it fits in 100 columns, else ``A`` to ``D`` one by one.

        Returns
        -------
        str
            Formatted converter summary and matrix table.
        """
        kind = "discrete" if self.discrete else "continuous"
        head = f"Frontend(J={self.J}, K={self.K}, L={self.L}, M={self.M}, N={self.N}, dt={self.dt:.3e}, {kind})"
        table = StateSpace.__str__(self)
        hat = [
            f"{name} {matrix.shape[2:]}:\n{_table([matrix[0, 0]])}"
            for name, matrix in (("C_hat", self.C_hat), ("D_hat", self.D_hat))
            if matrix is not None
        ]
        if max(map(len, table.splitlines())) <= 100:
            return "\n".join([head, table, *hat])
        blocks = (
            f"{name} {m.shape[2:]}:\n{_table([m[0, 0]])}"
            for name, m in zip("ABCD", self.abcd)
        )
        return "\n".join([head, *blocks, *hat])

    # -- scaling ----------------------------------------------------------------
    def with_trigger(self, period: ArrayLike, phase: ArrayLike = 0) -> "Frontend":
        """Channel ``m`` decides only at steps ``i % period[m] == phase[m]``.

        Its ``C`` and ``D`` rows are zero at every other step, so it holds its
        decision; the matrices become cyclic over the least common period.

        Parameters
        ----------
        period : int or array_like, shape (M,)
            Positive decision period in clock steps for each comparator.
        phase : int or array_like, shape (M,), optional
            Decision offset in clock steps; each value is in ``[0, period)``.
            Defaults to ``0``.

        >>> af, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
        >>> ti = af.with_trigger([2, 2, 1], [0, 1, 0])  # channels 0 and 1 alternate
        >>> ti.K, ti.C[0, 1, 0].tolist(), not np.any(ti.D[0, 1, 0])
        (2, [0.0, 0.0, 0.0], True)

        Returns
        -------
        Frontend
            Copy with the requested comparator decision schedule.
        """
        period, phase = (np.broadcast_to(a, (self.M,)) for a in (period, phase))
        if (period < 1).any() or (phase < 0).any() or (phase >= period).any():
            raise ValueError("trigger needs period >= 1 and 0 <= phase < period")
        K = math.lcm(self.K, *period.tolist())
        quiet = (np.arange(K)[:, None] % period != phase)[None, :, :, None]
        new = deepcopy(self)
        new.C = np.where(quiet, 0.0, self.C[:, np.arange(K) % self.C.shape[1]])
        new.D = np.where(quiet, 0.0, self.D[:, np.arange(K) % self.D.shape[1]])
        return new

    def with_dt(self, dt: float) -> "Frontend":
        """The same converter at clock period ``dt``: identical decisions, scaled bandwidth.

        ``A``, ``B``, ``slew_rate`` and the noise intensity scale with
        ``r = dt_old / dt``.

        Parameters
        ----------
        dt : float
            Positive new clock period in seconds.

        >>> af, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
        >>> unit = af.with_dt(1.0)
        >>> unit.fs, bool(np.allclose(unit.A * unit.dt, af.A * af.dt))
        (1.0, True)

        Returns
        -------
        Frontend
            Copy scaled to the new clock period.
        """
        if self.discrete:
            raise ValueError("a discrete-time model has no time scale to change")
        if not dt > 0:
            raise ValueError("dt must be positive")
        r = self.dt / dt
        new = deepcopy(self)
        new.dt, new.A, new.B = float(dt), self.A * r, self.B * r
        new.slew_rate = self.slew_rate * r
        if self.state_covariance is not None:
            new.state_covariance = self.state_covariance * r
        new._flicker = {
            node: (corner * r, alpha) for node, (corner, alpha) in self._flicker.items()
        }
        return new

    def with_flicker(
        self,
        node: int,
        f_corner: float,
        alpha: float = 1.0,
    ) -> "Frontend":
        """Add sampled ``1/f**alpha`` state noise to a copy of the converter.

        At low frequency its two-sided density is
        ``Q[j, 0, node, node] * (f_corner/f)**alpha`` where ``Q`` is
        ``state_covariance``; this intensity must be phase-invariant. A
        finite record is drawn and filtered before
        each simulation; it adds no states. Its lowest frequency is set by
        the simulation length and its sample interval is ``dt``.

        Parameters
        ----------
        node : int
            Zero-based state index receiving flicker noise, in ``[0, N)``.
        f_corner : float
            Positive frequency in Hz where flicker and white spectra cross.
        alpha : float, optional
            Ngspice's ``NALPHA`` spectral exponent, in ``(0, 2)``; defaults to ``1.0``.

        >>> af, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
        >>> af.state_covariance = np.diag([1e-8, 0.0, 0.0])
        >>> af.with_flicker(0, f_corner=1e5).N == af.N
        True

        Returns
        -------
        Frontend
            Copy with a flicker source on the chosen state.
        """
        if self.discrete:
            raise ValueError("flicker noise needs a continuous-time model")
        if self.state_covariance is None:
            raise ValueError("set state_covariance (the white intensity) first")
        if not isinstance(node, (int, np.integer)) or not 0 <= node < self.N:
            raise ValueError("node must index a frontend state")
        if not np.isfinite(f_corner) or f_corner <= 0:
            raise ValueError("f_corner must be positive and finite")
        if not np.isfinite(alpha) or not 0 < alpha < 2:
            raise ValueError("alpha must lie in (0, 2)")
        if np.any(self.state_covariance[..., node, node] <= 0):
            raise ValueError("the selected state needs positive white-noise intensity")
        new = deepcopy(self)
        new._flicker[node] = (float(f_corner), float(alpha))
        return new

    # -- frequency domain -----------------------------------------------------
    def transfer_function(
        self,
        jw: ArrayLike,
        open_loop: bool = True,
        input_index: Optional[int | Sequence[int]] = None,
        output_index: Optional[int | Sequence[int]] = None,
        state_output: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``H(jw) = C (jw I - A)^-1 B + D`` of the ``k = 0`` matrices (``z = e^{jw dt}`` if discrete).

        With ``open_loop=False`` the control feed-through ``D_M`` is closed.
        Returns ``(jw, H)`` with ``H`` of shape ``(size, M, L + M)``, or the
        states ``(size, N, L + M)`` when ``state_output``.

        Parameters
        ----------
        jw : array_like, shape (size,)
            Complex angular frequencies ``1j * omega``, with ``omega`` in rad/s.
            Supply a one-dimensional array, including for one frequency.
        open_loop : bool, optional
            If false, close the control feed-through; only the ``L`` input
            columns then remain.
            Defaults to ``True``.
        input_index : int or sequence of int, optional
            Input columns to retain, in ``[0, L + M)`` for open loop or
            ``[0, L)`` for closed loop; defaults to all columns.
        output_index : int or sequence of int, optional
            Comparator rows to retain, in ``[0, M)``; defaults to all rows.
        state_output : bool, optional
            Return state response instead of comparator response.
            Defaults to ``False``.

        >>> af, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
        >>> jw = 2j * np.pi * np.array([1e3, 1e6])
        >>> _, H = af.transfer_function(jw, input_index=0, output_index=-1)
        >>> H.shape, bool(abs(H[0, 0, 0]) > 100 * abs(H[1, 0, 0]))  # third-order low-pass
        ((2, 1, 1), True)

        Returns
        -------
        s : ndarray of complex, shape (size,)
            ``jw`` for continuous-time models or ``exp(jw * dt)`` for discrete-time models.
        H : ndarray of complex, shape (size, outputs, inputs)
            Selected comparator response, or ``(size, N, inputs)`` state response when ``state_output`` is true.
        """
        if self.J_frontend > 1:
            raise ValueError(
                "transfer_function needs one frontend matrix set; pick frontend[j]"
            )
        A, B, C, D = (m[0, 0] for m in self.abcd)
        if not open_loop:
            Bl, Bm, Dl, Dm = (
                B[:, : self.L],
                B[:, self.L :],
                D[:, : self.L],
                D[:, self.L :],
            )
            I_DM = np.linalg.inv(np.eye(self.M) - Dm)
            A, B, C, D = A + Bm @ I_DM @ C, Bl + I_DM @ Dl, I_DM @ C, I_DM @ Dl
        if input_index is not None:
            B, D = (
                B[:, input_index].reshape(self.N, -1),
                D[:, input_index].reshape(self.M, -1),
            )
        if output_index is not None:
            C, D = C[output_index].reshape(-1, self.N), D[output_index]
        jw = np.asarray(jw, dtype=complex)
        s = np.exp(jw * self.dt) if self.discrete else jw
        x = np.linalg.solve(
            s[:, None, None] * np.eye(self.N) - A[None],
            np.broadcast_to(B, (s.size,) + B.shape),
        )
        return (s, x) if state_output else (s, C @ x + D)

    def wiener_filter(
        self, eta2: Optional[float] = None, OSR: Optional[float] = None
    ) -> "WienerFilter":
        """The analytical Wiener filter; ``OSR`` picks ``eta2 = |G(j pi fs / OSR)|^2``.

        Parameters
        ----------
        eta2 : float, optional
            Positive regularization variance; required unless ``OSR`` is given.
            Defaults to ``None``.
        OSR : float, optional
            Oversampling ratio used to derive ``eta2`` when it is omitted.
            Defaults to ``None``.

        Returns
        -------
        WienerFilter
            Analytical estimator for the converter.
        """
        from .estimator import WienerFilter

        if eta2 is None:
            if OSR is None:
                raise ValueError("give eta2 or OSR")
            nominal = (
                self[0] if self.J_frontend > 1 else self
            )  # one eta2 for a Monte Carlo
            _, tf = nominal.transfer_function(
                np.array([1j * np.pi / (OSR * self.dt)]), input_index=0, output_index=-1
            )
            eta2 = float(np.abs(tf[0, 0, 0]) ** 2)
        return WienerFilter(self, float(eta2))

    # -- discretisation -------------------------------------------------------
    def discretize(self) -> "Discretization":
        """The one-step maps of the clocked system; see :class:`Discretization`.

        Returns
        -------
        Discretization
            One-step maps for each cyclic phase and frontend matrix set.
        """
        return Discretization(self)

    # -- simulation ------------------------------------------------------------
    def simulate(
        self,
        size: int,
        x0: Optional[ArrayLike] = None,
        record: str | Sequence[str] = ("v", "u", "y_hat"),
        engine: str = "numba",
        threads: Optional[int] = None,
        verbose: bool = True,
        seed: Optional[int] = 98123591265830293457639481,
    ) -> dict[str, np.ndarray]:
        """Simulate ``size`` clock periods from the model's ``A, B, C, D``.

        :meth:`discretize` combines those matrices with the input sources and
        DAC pulses into one-step ``ABCD`` maps. At step ``i > 0``, batch member
        ``j`` uses phase ``k = i % K``::

            [z[i]; y[i]] = ABCD[j, k] [z[i-1]; v[i-1]] + excitation[i]
            v[i] = Q(y[i])

        ``x[i]`` is the first ``N`` rows of ``z[i]``. ``u[i]`` is the input
        value at clock edge ``i``, represented by source states or
        ``excitation[i]``. The digital twin replays ``v`` from rest without
        ``u``, producing ``x_hat[i]`` and
        ``y_hat[i] = C_hat[j, k] x_hat[i] + D_hat[j, k] v[i-1]``.
        A length-one batch axis shares
        its matrix across batch members. Continuous-time maps integrate the
        linear waveforms over each clock period; discrete-time maps are already
        one-step matrices. Step zero quantizes the initial state.

        Parameters
        ----------
        size : int
            Positive number of clock steps.
        x0 : array_like, shape (N,), (N, 1), or (N, J), optional
            Initial frontend states; ``None`` starts them at zero (default).
        record : str or sequence of str, optional
            Names of records to return; defaults to
            ``("v", "u", "y_hat")``. Choose from ``"v"``, ``"u"``,
            ``"y_hat"``, ``"x"``, ``"y"``, ``"x_hat"`` and ``"t"``, or
            pass ``"all"`` to return every record.
        engine : {"numba", "numpy"}, optional
            Defaults to ``"numba"``. After compilation, its loop is expected
            to help most for long runs with small maps (``N_x + M <= 24``;
            up to 64 for per-batch maps). Large ``J`` also permits parallel
            blocks. ``"numpy"`` avoids compilation for short runs; both use
            BLAS for larger maps. Speed depends on the workload and hardware.
        threads : int, optional
            Numba worker count; ``None`` uses its runtime setting (default).
        verbose : bool, optional
            Print estimated time and memory to stderr; defaults to True.
        seed : int or None, optional
            Seed for this run's noise draws; defaults to
            ``98123591265830293457639481``. ``None`` draws a fresh seed.

        Returns
        -------
        dict[str, ndarray]
            Selected records: decisions ``v`` ``(size, M, J)``, input
            ``u`` ``(size, L, J)``, twin readout ``y_hat``
            ``(size, M_hat, J)``, frontend ``x`` and twin ``x_hat``
            ``(size, N, J)``, comparator ``y`` ``(size, M, J)``, and
            clock-edge time ``t[i] = i*dt`` ``(size,)``.
        """
        from .simulation import simulate

        return simulate(
            self,
            size,
            x0,
            record,
            engine,
            threads,
            verbose,
            seed,
        )

    # -- synthesis ------------------------------------------------------------
    @staticmethod
    def leapfrog(
        N: int,
        BW: float,
        ENOB: Optional[float] = None,
        OSR: Optional[float] = None,
        SNR: Optional[float] = None,
        xi: float = 4e-3,
        delta: float = 1.0,
    ) -> tuple["Frontend", float]:
        """A leap-frog converter of order ``N`` for bandwidth ``BW`` [Hz].

        Give ``ENOB`` (or ``SNR`` [dB]) to size it for a resolution, or ``OSR``
        to fix the clock. The twin reads the negative final state via
        ``C_hat = -e_N``. Feedback defaults to an equal-area RZ pulse on
        ``[T/4, 3T/4]``.
        Returns ``(frontend, OSR)``.

        Parameters
        ----------
        N : int
            Positive number of integrator states and comparator channels.
        BW : float
            Positive signal bandwidth in Hz.
        ENOB : float, optional
            Target effective number of bits; one of ``ENOB``, ``SNR`` or ``OSR`` is required.
            Defaults to ``None``.
        OSR : float, optional
            Oversampling ratio used when no resolution target is supplied.
            Defaults to ``None``.
        SNR : float, optional
            Target SNR in dB; takes precedence over ``ENOB``.
            Defaults to ``None``.
        xi : float, optional
            Dimensionless noise-shaping design factor.
            Defaults to ``4e-3``.
        delta : float, optional
            Dimensionless state-scaling factor.
            Defaults to ``1.0``.

        >>> af, OSR = Frontend.leapfrog(N=4, ENOB=12, BW=1e6)
        >>> af.N, round(OSR)
        (4, 14)

        Returns
        -------
        frontend : Frontend
            Synthesized leap-frog converter.
        OSR : float
            Effective oversampling ratio.
        """
        from .fom import enob_to_snr, snr_from_dB, snr_to_enob

        if SNR is not None:
            ENOB = snr_to_enob(SNR)
        omega_BW = 2.0 * np.pi * BW
        if ENOB is not None:
            gamma = (xi / _g_i_leapfrog(N) * snr_from_dB(enob_to_snr(float(ENOB)))) ** (
                1.0 / (2.0 * N)
            ) * delta
            omega_p = omega_BW / 2.0
            beta, alpha = -omega_p * 2.0 * gamma, omega_p / (2.0 * gamma)
            dt = 1.0 / abs(2.0 * omega_BW * gamma / delta)
            OSR = 1.0 / (2.0 * dt * BW)
        elif OSR is not None:
            dt = 1.0 / (2 * OSR * BW)
            gamma = delta * OSR / (2 * np.pi)
            beta, alpha = -gamma * omega_BW, omega_BW / (4 * gamma)
        else:
            raise ValueError("give ENOB, SNR or OSR")
        A = np.diag(alpha * np.ones(N - 1), 1) + np.diag(beta * np.ones(N - 1), -1)
        kappa = np.array([beta * delta**n for n in range(N)])
        return (
            Frontend(
                A, _b_matrix(beta, kappa), np.eye(N), dt=dt, C_hat=-np.eye(N)[-1:]
            ),
            OSR,
        )

    @staticmethod
    def chain_of_integrators(
        N: int,
        BW: float,
        ENOB: Optional[float] = None,
        OSR: Optional[float] = None,
        SNR: Optional[float] = None,
        xi: float = 2.3e-3,
        rho: float = 0.0,
        local_feedback: bool = False,
        finite_gain: bool = False,
        delta: float = 1.0,
    ) -> tuple["Frontend", float]:
        """A chain-of-integrators converter of order ``N`` for bandwidth ``BW`` [Hz].

        The twin reads the negative final state via ``C_hat = -e_N``;
        feedback defaults to an equal-area RZ pulse on ``[T/4, 3T/4]``.
        See :meth:`leapfrog`.

        Parameters
        ----------
        N : int
            Positive number of integrator states and comparator channels.
        BW : float
            Positive signal bandwidth in Hz.
        ENOB : float, optional
            Target effective number of bits.
            Defaults to ``None``.
        OSR : float, optional
            Oversampling ratio used when no resolution target is supplied.
            Defaults to ``None``.
        SNR : float, optional
            Target SNR in dB; takes precedence over ``ENOB``.
            Defaults to ``None``.
        xi : float, optional
            Dimensionless noise-shaping design factor.
            Defaults to ``2.3e-3``.
        rho : float, optional
            Integrator leakage rate in inverse seconds.
            Defaults to ``0.0``.
        local_feedback : bool, optional
            Derive local feedback from the bandwidth and resolution target.
            Defaults to ``False``.
        finite_gain : bool, optional
            Add finite-gain leakage to the state matrix.
            Defaults to ``False``.
        delta : float, optional
            Dimensionless state-scaling factor.
            Defaults to ``1.0``.

        Returns
        -------
        frontend : Frontend
            Synthesized chain-of-integrators converter.
        OSR : float
            Effective oversampling ratio.
        """
        from .fom import enob_to_snr, snr_from_dB, snr_to_enob

        if SNR is not None:
            ENOB = snr_to_enob(SNR)
        omega_3dB = 2.0 * np.pi * BW
        if ENOB is not None:
            gamma = (xi / (2.0 * N + 1.0) * snr_from_dB(enob_to_snr(float(ENOB)))) ** (
                1.0 / (2.0 * N)
            )
            beta = -gamma * omega_3dB
            rho = -omega_3dB / gamma if local_feedback else rho
            dt = 1.0 / abs(2.0 * beta)
            OSR = 1.0 / (2.0 * dt * BW)
            kappa = beta * np.ones(N)
        elif OSR is not None:
            dt = 1.0 / (2 * BW * OSR)
            beta = delta / (2 * dt)
            kappa = np.array([beta * delta**n for n in range(N)])
        else:
            raise ValueError("give ENOB, SNR or OSR")
        A = np.diag(rho * np.ones(N)) + np.diag(beta * np.ones(N - 1), -1)
        if finite_gain:
            A -= omega_3dB / gamma ** (2 * N) * np.eye(N)
        B, C = _b_matrix(beta, kappa), -np.sign(beta) * np.eye(N)
        return Frontend(A, B, C, dt=dt, C_hat=-np.eye(N)[-1:]), OSR

    @staticmethod
    def dtsdm(ABCD: ArrayLike, levels: int = 2, L: int = 1, M: int = 1) -> "Frontend":
        """A discrete-time delta-sigma modulator from a toolbox ``ABCD`` matrix.

        In the Schreier/deltasigma convention, ``levels`` (called ``nlev``
        there) is the number of output codes, not the number of bits. The
        codes are ``-(levels - 1), -(levels - 3), ..., levels - 1``. Thus
        ``levels=2`` has ``{-1, +1}``, ``levels=3`` has ``{-2, 0, +2}``,
        and ``levels=4`` has ``{-3, -1, +1, +3}``.

        This frontend divides the toolbox's input ``u``, states ``x``,
        comparator input ``y``, and decision ``v`` by ``levels - 1``.
        Decisions therefore always span ``[-1, 1]``: for ``levels=3`` they
        are ``{-1, 0, +1}``, and for ``levels=4`` they are
        ``{-1, -1/3, +1/3, +1}``. The quantizer thresholds scale by the
        same factor: with ``levels=3`` they are ``y = -1/2`` and ``+1/2``.
        ``ABCD`` itself needs no rescaling because all its
        signal units change together. An input of ``0.5`` here corresponds
        to ``0.5 * (levels - 1)`` in the toolbox.
        For one comparator, ``C_hat = 0`` and ``D_hat = 1`` give
        ``y_hat[i] = v[i-1]`` for ``i > 0``, with ``y_hat[0] = 0``.
        Multiple comparators retain ``C_hat = -C``, ``D_hat = -D[:, L:]``.

        Parameters
        ----------
        ABCD : array_like, shape (N + M, N + L + M)
            Toolbox state-space block matrix.
        levels : int, optional
            Number of quantizer levels per comparator.
            Defaults to ``2``.
        L : int, optional
            Number of external input columns.
            Defaults to ``1``.
        M : int, optional
            Number of comparator and DAC channels.
            Defaults to ``1``.

        Returns
        -------
        Frontend
            Discrete-time converter synthesized from the toolbox block matrix.
        """
        from .delsig import partitionABCD

        A, B, C, D = partitionABCD(ABCD, m=L + M, r=M)
        return Frontend(
            A,
            B,
            C,
            D,
            dt=1.0,
            levels=levels,
            discrete=True,
            dac=DAC(Constant(1.0)),
            C_hat=np.zeros_like(C) if M == 1 else -C,
            D_hat=np.ones((1, 1)) if M == 1 else -D[:, L:],
        )

    @staticmethod
    def ctsdm(
        ABCD: ArrayLike,
        tdac: ArrayLike,
        dt: float = 1.0,
        levels: int = 2,
        L: int = 1,
        M: int = 1,
    ) -> "Frontend":
        """A continuous-time delta-sigma modulator from a ``T = 1`` loop filter.

        ``levels`` uses the same output codes and full-scale normalization as
        :meth:`dtsdm`. The toolbox matrix describes one unit-length clock
        period; ``dt`` gives that period a duration in seconds, while
        ``tdac`` gates a :class:`DAC` with a ``Constant(1.0)`` shape. The
        feedback gain remains in ``B``.
        For one comparator, ``C_hat = 0`` and ``D_hat = 1`` give
        ``y_hat[i] = v[i-1]`` for ``i > 0``, with ``y_hat[0] = 0``.
        Multiple comparators retain ``C_hat = -C``, ``D_hat = -D[:, L:]``.

        Parameters
        ----------
        ABCD : array_like, shape (N + M, N + L + M)
            Toolbox loop-filter matrix for a unit clock period ``T = 1``.
        tdac : array_like, shape (2,) or (M, 2)
            Start and end of the constant DAC shape after a clock edge, in
            clock periods. ``(0, 1)`` uses the standard immediate, full-period
            pulse. A pair
            ``(t0, t1)`` otherwise becomes
            ``DAC(Constant(1.0), window=(t0, t1 - t0))``: ``tdac`` gives
            start and end, whereas ``DAC.window`` stores delay and width.
            Thus ``(0.25, 0.75)`` gates the shape from ``0.25 * dt`` to
            ``0.75 * dt`` after the edge. All active DAC channels must share
            the same timing.
        dt : float, optional
            Time between clock edges in seconds; defaults to ``1.0``. The
            ``A`` and ``B`` blocks are divided by ``dt`` to convert the
            toolbox's ``T = 1`` dynamics to this clock rate; ``C`` and ``D``
            are unchanged.
        levels : int, optional
            Number of output codes per comparator, as in :meth:`dtsdm`;
            defaults to ``2``.
        L : int, optional
            Number of external input columns.
            Defaults to ``1``.
        M : int, optional
            Number of comparator and DAC channels.
            Defaults to ``1``.

        Returns
        -------
        Frontend
            Continuous-time converter with the specified DAC timing.
        """
        from .delsig import partitionABCD

        A, B, C, D = partitionABCD(ABCD, m=L + M, r=M)
        tdac = np.atleast_2d(np.asarray(tdac, dtype=float))
        active = {
            (round(a, 12), round(b, 12)) for a, b in tdac if not (a < 0 and b < 0)
        }
        if len(active) > 1:
            raise NotImplementedError(
                f"ctsdm models one DAC timing, got {sorted(active)}"
            )
        t0, t1 = active.pop() if active else (0.0, 1.0)
        window = None if (t0, t1) == (0.0, 1.0) else (t0, t1 - t0)
        return Frontend(
            A / dt,
            B / dt,
            C,
            D,
            dt=dt,
            levels=levels,
            dac=DAC(Constant(1.0), window=window),
            C_hat=np.zeros_like(C) if M == 1 else -C,
            D_hat=np.ones((1, 1)) if M == 1 else -D[:, L:],
        )

    @staticmethod
    def from_delsig_ntf(
        ntf: lti,
        *,
        BW: float,
        OSR: float,
        form: str = "FB",
        tdac: ArrayLike = (0, 1),
        nlev: int = 2,
        normalize_stf: bool = True,
    ) -> tuple["Frontend", float]:
        """A continuous-time delta-sigma from a noise transfer function.

        Parameters
        ----------
        ntf : scipy.signal.lti
            Discrete-time noise transfer function.
        BW : float
            Positive signal bandwidth in Hz.
        OSR : float
            Positive oversampling ratio.
        form : str, optional
            Delta-sigma toolbox realization form.
            Defaults to ``"FB"``.
        tdac : array_like, shape (2,) or (M, 2), optional
            DAC pulse start and end times in normalized clock periods.
            Defaults to ``(0, 1)``.
        nlev : int, optional
            Number of quantizer levels.
            Defaults to ``2``.
        normalize_stf : bool, optional
            Normalize the signal transfer function at DC.
            Defaults to ``True``.

        Returns
        -------
        frontend : Frontend
            Realized converter.
        OSR : float
            Effective oversampling ratio.
        """
        from .delsig import realizeNTF_ct

        ABCDc, tdac2 = realizeNTF_ct(ntf, form=form, tdac=tdac)
        dt = 1.0 / (2.0 * float(BW) * float(OSR))
        af = Frontend.ctsdm(ABCDc, tdac2, dt=dt, levels=int(nlev))
        if normalize_stf:
            # Use sampled maps so B_v includes the chosen DAC window, including
            # return-to-zero or delayed pulses. Linearise the quantiser as
            # v = C x. Then x_next = A_d x + B_u u + B_v v, so at DC
            # (I - A_d - B_v C) x = B_u u. C[-1] reads the last comparator.
            dz = af.discretize()
            N = af.N
            A_d, B_u, B_v = dz.A_d[0, 0], dz.B_u[0, 0], dz.ABCD[0, 0, :N, N:]
            C = af.C[0, 0]
            gain = C[-1] @ np.linalg.solve(np.eye(N) - A_d - B_v @ C, B_u)
            # Scale only the input coupling: |STF(0)| becomes one, while the
            # feedback dynamics and quantiser code levels stay as designed.
            af.B[..., 0] /= np.abs(gain[0])
        return af, float(OSR)

    @staticmethod
    def mash(
        stage_ntfs: Sequence[lti],
        *,
        BW: float,
        OSR: float,
        nlev: int = 2,
        inter_stage_gain: float = 1.0,
        form: str = "CRFB",
    ) -> tuple["Frontend", float]:
        """A discrete-time MASH cascade as one converter with ``M = len(stage_ntfs)`` controls.

        Stage ``k + 1`` integrates ``inter_stage_gain`` times the residue
        ``C_k x_k - v_k`` of stage ``k``. Reading the final stage's comparator
        output with ``C_hat = -C[-1:]`` includes digital cancellation.

        Parameters
        ----------
        stage_ntfs : sequence of scipy.signal.lti, length M
            One discrete-time noise transfer function per stage.
        BW : float
            Positive signal bandwidth in Hz.
        OSR : float
            Positive oversampling ratio.
        nlev : int, optional
            Number of quantizer levels per stage.
            Defaults to ``2``.
        inter_stage_gain : float, optional
            Gain applied to each preceding stage's residue.
            Defaults to ``1.0``.
        form : str, optional
            Delta-sigma toolbox realization form.
            Defaults to ``"CRFB"``.

        Returns
        -------
        frontend : Frontend
            Synthesized multistage converter.
        OSR : float
            Effective oversampling ratio.
        """
        from scipy.signal import ZerosPolesGain, dlti, lti

        from .delsig import partitionABCD, realizeNTF, stuffABCD

        stages = []
        for ntf in stage_ntfs:
            if isinstance(ntf, lti) and not isinstance(ntf, (ZerosPolesGain, dlti)):
                raise NotImplementedError("mash needs discrete-time NTFs")
            # The toolbox supplies each isolated stage's state-space matrices.
            a, g_, b, c = realizeNTF(ntf, form)
            A_k, B_k, C_k, _ = partitionABCD(stuffABCD(a, g_, b, c, form), m=2, r=1)
            stages.append((np.asarray(A_k), np.asarray(B_k), np.asarray(C_k)))
        if not stages:
            raise ValueError("mash needs at least one stage")
        g = float(inter_stage_gain)
        # Lay the stage states consecutively in the combined state vector.
        offsets = np.cumsum([0, *(A_k.shape[0] for A_k, *_ in stages)])
        N, M = int(offsets[-1]), len(stages)
        A, B, C = np.zeros((N, N)), np.zeros((N, 1 + M)), np.zeros((M, N))
        for k, (A_k, B_k, C_k) in enumerate(stages):
            rows = slice(offsets[k], offsets[k + 1])
            A[rows, rows], C[k, rows] = A_k, C_k[0]
            # Every stage has its own decision v_k in B column 1 + k.
            B[rows, 1 + k] = B_k[:, 1]
            if k == 0:
                # Only the first stage receives the external input u.
                B[rows, 0] = B_k[:, 0]
            else:
                # Later stages receive g(C_prev x_prev - v_prev). Substituting
                # this for their input gives the cross-stage state coupling
                # g B_k,u C_prev and the previous decision gain -g B_k,u.
                A[rows, offsets[k - 1] : offsets[k]] = g * np.outer(
                    B_k[:, 0], stages[k - 1][2][0]
                )
                B[rows, k] = -g * B_k[:, 0]
        return Frontend(
            A,
            B,
            C,
            dt=1.0 / (2.0 * OSR * BW),
            levels=nlev,
            discrete=True,
            dac=DAC(Constant(1.0)),
            C_hat=-C[-1:],
            D_hat=np.zeros((1, M)),
        ), float(OSR)


def _b_matrix(beta: float, kappa: np.ndarray) -> np.ndarray:
    """``[B_L | B_M]`` of a synthesised chain.

    Parameters
    ----------
    beta : float
        Input gain into the first state.
    kappa : ndarray of float, shape (N,)
        Diagonal DAC gains.

    Returns
    -------
    ndarray of float, shape (N, 1 + N)
        Input column followed by diagonal DAC coupling columns.
    """
    B = np.zeros((kappa.size, 1 + kappa.size))
    B[0, 0], B[:, 1:] = beta, np.diag(kappa)
    return B


def _g_i_leapfrog(N: int) -> float:
    """The leap-frog noise-shaping integral.

    Parameters
    ----------
    N : int
        Positive leap-frog order.

    Returns
    -------
    float
        Dimensionless leap-frog noise-shaping integral factor.
    """
    from numpy.polynomial import polynomial as P

    poly = np.array([1.0])
    for n in range(1, N + 1):
        poly = P.polymul(poly, [np.cos(n * np.pi / (N + 1)), 1.0])
    integral = P.polyval(1.0, P.polyint(P.polymul(poly, poly)))
    return 1.0 / integral


# --------------------------------------------------------------------------- #
# Discretisation
# --------------------------------------------------------------------------- #


class Discretization:
    r"""One-step maps of a :class:`Frontend` by batch variant and clock phase.

    Dimensions:

    - ``J``: total simulation batch members.
    - ``K``: cyclic clock phases.
    - ``L``: external input channels.
    - ``M``: comparators and decisions.
    - ``N``: frontend matrix states.
    - ``J_m``: matrix variants, either ``1`` (shared) or ``J``.
    - ``N_s``: decision-driven DAC states and pulse memory.
    - ``N_x``: all augmented states, with ``N_x >= N + N_s``.
    - ``N_p``: persistent continuous-time states, with ``N_p <= N_x``.

    For persistent state ``p`` (frontend, DAC/filter, source), form
    ``dp/dt=A_aug p+b(t)``. For each matrix variant and phase (``T=dt``),
    ``F=expm(A_aug*T)`` and ``G=∫_0^T expm(A_aug*(T-t)) b(t) dt``
    advance ``p`` and integrate a unit DAC or sampled-input pulse.
    Windows limit that integral; pulse memory spans periods. For discrete
    time, ``A``/``B`` already give one step. Append comparator readout to ``ABCD``.

    ``z`` holds frontend ``x``, persistent DAC/filter/source states, an optional
    offset state ``q=1``, and past ``v``/``u`` for long pulses:
    ``z_i=A_d z_{i-1}+B_v v_{i-1}+B_u u_{i-1}+w_i`` and
    ``y_i=C z_i+D_v v_{i-1}+D_L u_i+n_i``. ``w_i`` and ``n_i`` are state
    and comparator noise; held comparator rows copy the previous decision.

    Parameters
    ----------
    af : Frontend
        Clocked converter to discretize.

    Attributes
    ----------
    ABCD : ndarray, shape (J_m, K, N_x + M, N_x + M)
        ``[z[i]; y[i]] = ABCD[j, k] [z[i-1]; v[i-1]] + excitation[i]``.
    x0 : ndarray, shape (1 or J, N_x)
        Initial state, shared or one row per batch member.
    C : ndarray, shape (J_m, K, M, N_x)
        Comparator state readout, with held channels zeroed.
    D_L : ndarray, shape (J_m, K, M, L)
        Direct input-to-comparator map, with held channels zeroed.
    B_u : ndarray, shape (J_m, K, N_x, L)
        One-step input-to-state map, also used for sampled inputs.
    noise_factor : ndarray, shape (J_f, K, N_x, N), or None
        State-noise factor; ``J_f`` broadcasts frontend matrices and the
        covariance's batch axis. Its product with its transpose gives the
        per-step state-noise covariance.
    A_d, B_d : ndarray, shapes (J_m, K, N_x, N_x), (J_m, K, N_x, L + M)
        State transition and input/decision maps.
    A_aug : ndarray, shape (1 or J_m, K, N_p, N_p)
        Continuous-time matrix of persistent frontend, DAC, and source states;
        excludes restarting pulse states and pulse memory.
    diagnostics : dict
        ``stiffness`` (ratio of nonzero decay rates), ``norm`` of ``A_aug T``,
        ``radius`` (largest ``|lambda(A_d)|``), and, for continuous time,
        ``selfcheck`` (one exponential step versus two half steps).

    >>> af, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
    >>> dz = af.discretize()
    >>> dz.ABCD.shape, dz.A_d.shape, dz.B_d.shape
    ((1, 1, 6, 6), (1, 1, 3, 3), (1, 1, 3, 4))
    >>> af.input = Sinusoid(0.5, 1e5)  # the tone adds two source states
    >>> af.discretize().ABCD.shape
    (1, 1, 8, 8)

    Returns
    -------
    Discretization
        One-step maps of the supplied converter.
    """

    def __init__(self, af: Frontend):
        """Build one-step state and comparator maps for ``af``.

        Parameters
        ----------
        af : Frontend
            Converter whose continuous or discrete matrices are clocked.

        Returns
        -------
        None
            Stores the maps and their state layout on this instance.
        """
        self.af = af
        self.dt, self.N, self.M, self.L, self.K = af.dt, af.N, af.M, af.L, af.K
        N, M, L, K, dt = self.N, self.M, self.L, self.K, self.dt
        # Discrete-time maps use only the initial value of each pulse.
        if af.discrete:
            if any(w.window is not None or w.N != 1 for w in af.dac):
                raise ValueError("a discrete-time model has no DAC pulse shape")
            if any(
                w.window is not None
                or w.filter is not None
                or w.shape.N != 1
                or np.any(w.shape.A)
                for w in af.input
                if isinstance(w, Samples)
            ):
                raise ValueError(
                    "a discrete-time model has no sampled-input pulse shape"
                )

        # 1. Cascade the frontend matrices, DACs, sampled pulses, and sources.
        # State order: [x, DAC states, sample states, source states, offset].
        # Keep D_M separate: it acts on the held decision.
        model = StateSpace(*(m[:, np.arange(K) % m.shape[1]] for m in af.abcd))
        D_M = model.D[..., L:]
        full = StateSpace(
            model.A, model.B, model.C, _cat(-1, model.D[..., :L], 0 * D_M)
        )
        dacs = []  # (first state, DAC) per DAC
        for m, w in enumerate(af.dac):
            dacs.append((full.N, w))
            full = cascade(w, full, L + m)
        sampled = []  # (input column, first state, Samples)
        for ell, w in enumerate(af.input):
            if isinstance(w, Samples):
                sampled.append((ell, full.N, w))
                full = cascade(w, full, ell)
        B_in = full.B[..., :L]  # direct input coupling; sources enter through A
        sources = []  # (first state, autonomous source)
        source_states = {}  # locate a swept sine in the final x0
        for ell, w in enumerate(af.input):
            if (
                isinstance(w, StateSpace)
                and not isinstance(w, Samples)
                and np.any(w.x0)
            ):
                if af.discrete:  # zero-order hold on the sampled source
                    w = StateSpace(expm(w.A * dt), w.B, w.C, w.D, w.x0)
                source_states[ell] = np.arange(full.N, full.N + w.N)
                sources.append((full.N, w))
                full = cascade(w, full, ell)
        offset_state = None
        if np.any(af.comparator_input_offset):
            offset_state = full.N
            full = StateSpace(
                _block(full.A, np.zeros((1, 1, full.N, 1)), np.zeros((1, 1, 1, 1))),
                _cat(-2, full.B, np.zeros((1, 1, 1, L + M))),
                _cat(-1, full.C, af.comparator_input_offset[None, None, :, None]),
                full.D,
            )
        B_in = _cat(-2, B_in, np.zeros((1, 1, full.N - B_in.shape[-2], L)))
        # 2. A pulse's restart states reset at each edge; all other states
        # persist across periods. A long window also needs past drive values.
        pulse_sources = [(off, w) for off, w in dacs] + [
            (off, w) for _, off, w in sampled
        ]
        persistent = np.ones(full.N, dtype=bool)
        restart_masks = []
        for off, w in pulse_sources:
            mask = np.zeros(full.N, dtype=bool)
            mask[off : off + w.N] = w.restart
            persistent[mask] = False
            restart_masks.append(mask)
        self.N_z = sum(w.N for _, w in sources) + int(offset_state is not None)
        self.N_aux = int(persistent.sum()) - N - self.N_z
        spans = [w.window or (0.0, 1.0) for _, w in pulse_sources]
        # A pulse ending past T needs its earlier decision or sample retained.
        lags = [
            0 if af.discrete else max(0, int(np.ceil(d + wd - 1e-12)) - 1)
            for d, wd in spans
        ]
        self.r = max(lags[:M], default=0)
        memory = [
            (q, lag) for q, count in enumerate(lags) for lag in range(1, count + 1)
        ]
        N_P = int(persistent.sum())
        memory_index = {key: N_P + j for j, key in enumerate(memory)}
        self.N_memory = len(memory)
        self.N_x = N_x = N_P + len(memory)
        self.J_m = J_m = np.broadcast_shapes(
            *(matrix.shape[:1] for matrix in full.abcd)
        )[0]
        A_PP = full.A[..., persistent, :][..., :, persistent]
        self.A_aug, self.B_aug_u = A_PP, B_in[..., persistent, :]
        self.flicker_nodes = tuple(af._flicker)
        self.B_flicker = None
        if self.flicker_nodes:
            if af.state_covariance is None or any(
                np.any(af.state_covariance[..., node, node] <= 0)
                for node in self.flicker_nodes
            ):
                raise ValueError("flicker needs positive state noise intensity")
            if any(
                not np.allclose(
                    af.state_covariance[..., node, node],
                    af.state_covariance[:, :1, node, node],
                    rtol=1e-12,
                    atol=0,
                )
                for node in self.flicker_nodes
            ):
                raise ValueError("flicker needs phase-invariant state noise intensity")
            drive = np.eye(N_P)[:, self.flicker_nodes]
            drive = np.broadcast_to(drive, A_PP.shape[:2] + drive.shape)
            self.B_flicker = np.zeros((J_m, K, N_x, len(self.flicker_nodes)))
            self.B_flicker[..., :N_P, :] = _zoh(A_PP, drive, dt)[1]

        # 3. Start frontend and pulse memory at zero. Autonomous sources start
        # from their own x0; a constant comparator offset is a state fixed at 1.
        J_x = max([1] + [w.x0.shape[0] for _, w in sources])
        x0 = np.zeros((J_x, full.N))
        for off, w in sources:
            x0[:, off : off + w.N] = w.x0[np.arange(J_x) % w.x0.shape[0]]
        if offset_state is not None:
            x0[:, offset_state] = 1.0
        self.x0 = np.concatenate(
            (x0[:, persistent], np.zeros((J_x, len(memory)))), axis=1
        )

        # 4. A zero C and D comparator row means no decision at that phase.
        # Its output row must copy the previous decision instead.
        quiet = (np.abs(model.C).sum(-1) + np.abs(model.D).sum(-1) == 0.0)[..., None]
        self.C = np.zeros((J_m, K, M, N_x))
        self.C[..., :N_P] = np.where(quiet, 0.0, full.C[..., persistent])
        self.D_L = np.broadcast_to(
            np.where(quiet, 0.0, full.D[..., :L]), (J_m, K, M, L)
        ).copy()
        # A sampled pulse at the edge uses the current sample or its memory.
        for q, (ell, _, w) in enumerate(sampled, start=M):
            delay, width = spans[q]
            A_R = full.A[:1, :, restart_masks[q]][..., restart_masks[q]]
            x0_R = w.x0[0][w.restart]
            C_R = np.where(quiet, 0.0, full.C[..., restart_masks[q]])
            for lag in range(lags[q] + 1):
                if not delay <= lag < delay + width:
                    continue
                edge = C_R @ (expm(A_R * ((lag - delay) * dt)) @ x0_R[..., None])
                if lag == 0:
                    self.D_L[..., ell] += edge[..., 0]
                else:
                    self.C[..., memory_index[q, lag]] += edge[..., 0]
        D_v = np.where(quiet, 0.0, D_M) + np.eye(M) * quiet

        # 5. Integrate every active pulse segment. Column q is the decision
        # (or sample) that started pulse q; lagged columns use pulse memory.
        pulses = [  # coupling into persistent states, restart dynamics and state
            (
                full.A[..., persistent, :][..., :, mask],
                full.A[:1, :, mask][..., :, mask],
                w.x0[0][w.restart],
            )
            for mask, (_, w) in zip(restart_masks, pulse_sources)
        ]
        x_rows = self._state_rows(A_PP, pulses, spans, lags, memory_index)
        self.B_u = np.zeros((J_m, K, N_x, L))
        for k in range(K):
            self.B_u[:, k, :N_P] = (
                self.B_aug_u[:, k]
                if af.discrete
                else _zoh(A_PP[:, k], self.B_aug_u[:, k], dt)[1]
            )
        for q, (ell, _, _) in enumerate(sampled, start=M):
            self.B_u[..., ell] += x_rows[..., N_x + q]
        self.ABCD = np.zeros((J_m, K, N_x + M, N_x + M))
        self.ABCD[..., :N_x, :] = x_rows[..., : N_x + M]
        self.ABCD[..., N_x:, :] = self.C @ x_rows[..., : N_x + M]
        self.ABCD[..., N_x:, N_x:] += D_v
        # 6. State noise drives only the N frontend states. Its factor converts
        # independent normal draws into the exact per-step covariance.
        self.noise_factor = None
        if af.state_covariance is not None:
            Q = af.state_covariance
            J_q = np.broadcast_shapes((model.A.shape[0],), (Q.shape[0],))[0]
            self.noise_factor = np.zeros((J_q, K, N_x, N))
            for k in range(K):
                for j in range(J_q):
                    q = Q[j % Q.shape[0], k % Q.shape[1]]
                    cov = (
                        q
                        if af.discrete
                        else discrete_process_noise_cov(
                            model.A[j % model.A.shape[0], k], q, dt
                        )
                    )
                    self.noise_factor[j, k, :N] = positive_semidefinite(cov)
        # 7. Put frontend states first, then decision-driven states shared with
        # the digital twin, then source and sampled-input states.
        shared = np.zeros(full.N, dtype=bool)
        for off, w in dacs:
            shared[off : off + w.N] = True
        shared = np.concatenate(
            (shared[persistent], np.array([q < M for q, _ in memory], dtype=bool))
        )
        shared[:N] = False
        perm = np.concatenate(
            (np.arange(N), np.flatnonzero(shared), N + np.flatnonzero(~shared[N:]))
        )
        self.N_s = int(shared.sum())
        cols = np.r_[perm, N_x + np.arange(M)]  # the y rows keep their place
        self.ABCD = np.ascontiguousarray(self.ABCD[..., cols[:, None], cols])
        self.C = self.C[..., perm]
        self.B_u = self.B_u[..., perm, :]
        if self.B_flicker is not None:
            self.B_flicker = self.B_flicker[..., perm, :]
        self.x0 = self.x0[..., perm]
        # Source rows first lose restarting states, then follow the final order.
        persistent_rows = np.flatnonzero(persistent)
        self._source_state_indices = {}
        for ell, rows in source_states.items():
            compact = np.flatnonzero(np.isin(persistent_rows, rows))
            self._source_state_indices[ell] = np.flatnonzero(np.isin(perm, compact))
        if self.noise_factor is not None:
            self.noise_factor = self.noise_factor[..., perm, :]
        self.diagnostics = self._diagnose()

    def _repeat_sine_batch(
        self,
        af: Frontend,
        members: np.ndarray,
        amplitudes: np.ndarray,
        input_index: int,
    ) -> "Discretization":
        """Repeat member maps and scale only the selected sine's initial state.

        Parameters
        ----------
        af : Frontend
            Expanded frontend used by the simulation.
        members : ndarray of int, shape (J_exp,)
            Original circuit member for each sweep column.
        amplitudes : ndarray of float, shape (J_exp,)
            Linear tone amplitude for each column.
        input_index : int
            Input replaced by the sine source.

        Returns
        -------
        Discretization
            Shallow copy with selected step maps and scaled source states.
        """
        new = copy(self)
        new.af = af
        for name in (
            "ABCD",
            "C",
            "D_L",
            "B_u",
            "A_aug",
            "B_aug_u",
            "noise_factor",
            "B_flicker",
        ):
            value = getattr(self, name)
            if value is not None and value.shape[0] != 1:
                setattr(new, name, value[members % value.shape[0]])
        new.x0 = self.x0[members % self.x0.shape[0]].copy()
        tone_states = self._source_state_indices[input_index]
        if tone_states.size != 2:
            raise ValueError("the sine source needs two persistent states")
        new.x0[:, tone_states] *= amplitudes[:, None]
        new.J_m = new.ABCD.shape[0]
        return new

    def _state_rows(
        self,
        A_PP: np.ndarray,
        pulses: list,
        spans: list[tuple[float, float]],
        lags: list[int],
        memory_index: dict[tuple[int, int], int],
    ) -> np.ndarray:
        """Integrate each pulse's contribution to the next state.

        ``pulses[q] = (coupling, A_R, x0_R)`` and ``spans[q] = (d, w)``.
        For lag ``j``, integrate
        ``exp(A_PP*(T-t)) @ coupling @ exp(A_R*(t+j*T-d*T)) @ x0_R`` over
        ``[0,T]`` intersected with ``[(d-j)T,(d+w-j)T]``. Here ``T=dt`` and
        the result goes in that lag's drive column.
        """
        af, N_x, dt = self.af, self.N_x, self.dt
        J_m, K, N_P = A_PP.shape[:3]
        x_rows = np.zeros((J_m, K, N_x, N_x + len(pulses)))
        if af.discrete:
            x_rows[..., :N_P, :N_P] = A_PP
            for q, (coupling, _, x0_R) in enumerate(pulses):
                x_rows[..., :N_P, N_x + q] = coupling @ x0_R
            return x_rows
        x_rows[..., :N_P, :N_P] = expm(A_PP * dt)
        for (q, lag), row in memory_index.items():
            col = N_x + q if lag == 1 else memory_index[q, lag - 1]
            x_rows[..., row, col] = 1.0
        for q, (coupling, A_R, x0_R) in enumerate(pulses):
            delay, width = spans[q]
            t_start, t_stop = delay * dt, (delay + width) * dt
            for j in range(lags[q] + 1):
                on, off = max(0.0, t_start - j * dt), min(dt, t_stop - j * dt)
                if off <= on + 1e-15 * dt:
                    continue
                x_on = expm(A_R * (on + j * dt - t_start)) @ x0_R
                E = expm(_block(A_PP, coupling, A_R) * (off - on))
                column = expm(A_PP * (dt - off)) @ E[..., :N_P, N_P:] @ x_on[..., None]
                c = N_x + q if j == 0 else memory_index[q, j]
                x_rows[..., :N_P, c] += column[..., 0]
        return x_rows

    @property
    def A_d(self) -> np.ndarray:
        """One-step state transition matrices.

        Returns
        -------
        ndarray of float, shape (J_m, K, N_x, N_x)
            Transition map for each frontend and cyclic phase.
        """
        return self.ABCD[:, :, : self.N_x, : self.N_x]

    @property
    def B_d(self) -> np.ndarray:
        """One-step input and control maps.

        Returns
        -------
        ndarray of float, shape (J_m, K, N_x, L + M)
            Input columns followed by decision columns.
        """
        return np.concatenate(
            (self.B_u, self.ABCD[:, :, : self.N_x, self.N_x :]), axis=3
        )

    def _diagnose(self) -> dict:
        """Report matrix stiffness, transition radius, and exponential consistency."""
        A = self.A_aug[0, 0]
        lam = np.linalg.eigvals(A)
        re = np.abs(lam.real[np.abs(lam.real) > 1e-12 * max(np.abs(lam).max(), 1e-300)])
        finite = bool(np.all(np.isfinite(self.ABCD)))
        d = dict(
            stiffness=float(re.max() / re.min()) if re.size > 1 else 1.0,
            norm=float(np.linalg.norm(A * self.dt, 2)),
            radius=(
                float(np.abs(np.linalg.eigvals(self.A_d[0, 0])).max())
                if finite
                else np.inf
            ),
        )
        if not self.af.discrete:
            with np.errstate(all="ignore"):
                half, full = expm(A * self.dt / 2), expm(A * self.dt)
                d["selfcheck"] = (
                    float(np.linalg.norm(half @ half - full) / np.linalg.norm(full))
                    if finite
                    else np.inf
                )
            if not d["selfcheck"] < 1e-10:
                warnings.warn(
                    "discretisation matrix-exponential self-check failed: "
                    "relative Frobenius difference between two half steps "
                    f"and one full step is {d['selfcheck']:.1e} (limit 1e-10); "
                    f"||A_aug dt||_2={d['norm']:.1e}, "
                    f"stiffness ratio={d['stiffness']:.1e}, "
                    f"one-step matrices finite={finite}. "
                    "The discretised result may be unreliable; inspect the "
                    "model scaling and clock period.",
                    stacklevel=3,
                )
        return d

    def precompute_excitation(
        self, excitation: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Fill ``excitation`` with ``[e_z; e_y]`` for every step and member.

        For ``k = i % K`` and matrix variant ``j_m = 0`` or ``j``:
        ``e_z[i,j] = B_u[j_m,k] u[i-1,j] + F[j_f,k] w[i-1,j] + f[i,j]``;
        ``e_y[i,j] = C[j_m,k] e_z[i,j] + D_L[j_m,k] u[i,j] + G n[i,j]``.
        ``u`` contains only sampled input channels (L); ``w`` (N) and ``n``
        (M) are standard-normal draws. ``F`` (N_x, N) is the state-noise
        factor, with ``j_f = 0`` if shared or ``j`` if per member. ``f`` (N_x)
        is filtered flicker noise; ``G`` (M, M) factors the comparator-noise
        covariance. Autonomous sources are states in ``ABCD``.
        Absent terms are zero, and ``e_z[0] = 0``.

        Parameters
        ----------
        excitation : ndarray of float, shape (size, N_x + M, J)
            Output buffer for sampled inputs and noise; ``size`` is the number
            of clock steps and ``J`` the number of parallel batch members.
        rng : numpy.random.Generator
            Random generator for noise samples.

        Returns
        -------
        None
            Fills ``excitation`` in place; raises ``ValueError`` when there is
            no sampled input or noise to precompute.
        """
        af, N, M, N_x, K = self.af, self.N, self.M, self.N_x, self.K
        sampled = [(ell, w) for ell, w in enumerate(af.input) if isinstance(w, Samples)]
        if (
            not sampled
            and af.state_covariance is None
            and af.output_covariance is None
            and not af._flicker
        ):
            raise ValueError("no sampled input or noise to precompute")
        if (
            excitation.ndim != 3
            or excitation.shape[0] < 1
            or excitation.shape[1] != N_x + M
            or excitation.shape[2] < 1
        ):
            raise ValueError(
                f"excitation must have shape (size, {N_x + M}, J) with size, J > 0"
            )
        size, _, J = excitation.shape
        # [:, :N_x, :] holds e_z, added to the state z[i]; [:, N_x:, :]
        # holds e_y, added to the comparator input y[i]. Both start at zero.
        excitation.fill(0.0)

        # Write F w[i-1] first: out= overwrites, and only frontend states
        # receive white noise. Subsequent sources add to this same buffer.
        if self.noise_factor is not None:
            noise = rng.normal(size=(size - 1, N, J))  # drawn in step order
            for k in range(K):
                rows = slice(k or K, size, K)
                previous = slice(rows.start - 1, size - 1, K)
                F = np.broadcast_to(self.noise_factor[:, k, :N], (J, N, N))
                np.einsum("jab,ibj->iaj", F, noise[previous], out=excitation[rows, :N])
            del noise

        # Samples fill e_z with B_u u[i-1] for i>0 and e_y with D_L u[i]
        # at every edge, including i=0.
        if sampled:
            columns, waves = zip(*sampled)
            U = np.stack(
                [np.broadcast_to(w.held(size, log=False), (size, J)) for w in waves],
                axis=1,
            )  # (size, sampled inputs, J)
            B = np.broadcast_to(
                np.take(self.B_u, columns, axis=-1), (J, K, N_x, len(columns))
            )
            D = np.broadcast_to(
                np.take(self.D_L, columns, axis=-1), (J, K, M, len(columns))
            )
            for k in range(K):
                rows = slice(k or K, size, K)  # skip i=0, where e_z[0] = 0
                previous = slice(rows.start - 1, size - 1, K)
                excitation[rows, :N_x] += np.einsum(
                    "jnl,ilj->inj", B[:, k], U[previous]
                )
                excitation[k::K, N_x:] += np.einsum("jml,ilj->imj", D[:, k], U[k::K])
            del U, B, D
        # Flicker adds a filtered record to e_z for i>0, one per state/member.
        for p, node in enumerate(self.flicker_nodes):
            f_corner, alpha = af._flicker[node]
            taps = flicker_taps(size - 1, alpha)
            for j in range(J):
                colored = flicker_samples(
                    taps,
                    af.state_covariance[
                        j % af.state_covariance.shape[0], 0, node, node
                    ],
                    f_corner,
                    af.dt,
                    alpha,
                    rng,
                )
                for k in range(K):
                    drive = self.B_flicker[j % self.J_m, k, :, p]
                    rows = slice(k or K, size, K)
                    previous = slice(rows.start - 1, size - 1, K)
                    excitation[rows, :N_x, j] += colored[previous, None] * drive
        # After e_z is complete, add its C e_z contribution to e_y.
        for k in range(K):
            C = np.broadcast_to(self.C[:, k], (J, M, N_x))
            diagonal = np.diagonal(C, axis1=-2, axis2=-1)
            if M <= N_x and np.array_equal(C, np.eye(M, N_x) * diagonal[..., None]):
                # C = [diag(c) | 0]: scale the corresponding states directly.
                excitation[k::K, N_x:] += excitation[k::K, :M] * diagonal.T
            else:
                excitation[k::K, N_x:] += np.einsum(
                    "jab,ibj->iaj", C, excitation[k::K, :N_x]
                )
        # Comparator noise G n[i] completes e_y at every edge.
        if af.output_covariance is not None:
            excitation[:, N_x:] += np.einsum(
                "mk,ikj->imj",
                positive_semidefinite(af.output_covariance),
                rng.normal(size=(size, M, J)),
            )

    def __str__(self) -> str:
        """Summarize state dimensions and numerical diagnostics."""
        d = self.diagnostics
        s = f"Discretization(J_m={self.J_m}, K={self.K}, N_x={self.N_x} [N={self.N}, aux={self.N_aux}, memory={self.N_memory}, sources={self.N_z}, shared={self.N_s}], "
        s += f"stiffness={d['stiffness']:.2e}, |A dt|={d['norm']:.2e}, radius={d['radius']:.6f}"
        return s + (f", selfcheck={d['selfcheck']:.1e})" if "selfcheck" in d else ")")
