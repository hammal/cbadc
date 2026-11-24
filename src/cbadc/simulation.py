"""Clocked frontend simulation with NumPy and Numba recursion kernels."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

import numba
import numpy as np
from numba import njit as _njit
from numba import prange as _prange
from numpy.typing import ArrayLike

from .frontend import Discretization, Frontend, Samples

_W = 16  # batch members one Numba worker advances together
# Measured faster than tiny per-member BLAS calls on Apple Silicon.
_PACKED_PER_FRONTEND_MAX = 64


def _fmt_bytes(b: float) -> str:
    """Format a byte count with decimal units.

    Parameters
    ----------
    b : float
        Nonnegative byte count.

    Returns
    -------
    str
        Byte count with a decimal unit suffix.
    """
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if b < 1000 or unit == "TB":
            return f"{b:.0f} {unit}" if unit == "B" else f"{b:.1f} {unit}"
        b /= 1000
    return f"{b:.1f} TB"


def _fmt_time(t: float) -> str:
    """Format an elapsed time estimate.

    Parameters
    ----------
    t : float
        Nonnegative duration in seconds.

    Returns
    -------
    str
        Human-readable runtime estimate.
    """
    if t < 1e-3:
        return "<1 ms"
    if t < 1.0:
        return f"{1e3 * t:.0f} ms"
    if t < 120.0:
        return f"{t:.1f} s"
    return f"{t / 60:.1f} min"


def _announce(
    af: Frontend,
    size: int,
    N_xv: int,
    path: str,
    memory: dict,
    expected: float,
    excitation_scratch: int = 0,
):
    """Print simulation dimensions, memory, and estimated runtime.

    Parameters
    ----------
    af : Frontend
        Converter being simulated.
    size : int
        Number of clock steps.
    N_xv : int
        Number of augmented state and comparator rows.
    path : str
        Chosen simulation implementation.
    memory : dict of str to float
        Byte counts for retained records and working excitation.
    expected : float
        Estimated runtime in seconds.
    excitation_scratch : int, optional
        Temporary bytes allocated while building excitation; defaults to zero.

    Returns
    -------
    None
        Writes one summary line to stderr.
    """
    total = sum(memory.values())
    parts = [f"{k} {_fmt_bytes(b)}" for k, b in memory.items() if b > 0]
    peak = max(total, memory.get("excitation", 0) + excitation_scratch)
    peak_note = f" | excitation peak ~{_fmt_bytes(peak)}" if excitation_scratch else ""
    print(
        f"simulate: J={af.J} K={af.K} L={af.L} M={af.M} N={af.N} size={size} "
        f"(N_x+M={N_xv}) | {path} | ~{_fmt_bytes(total)}: "
        + ", ".join(parts)
        + peak_note
        + f" | ~{_fmt_time(expected)}",
        file=sys.stderr,
    )


def _input_record(af: Frontend, size: int) -> np.ndarray:
    """``u`` ``(size, L, J)``: sample drive values or source trajectories.

    Columns shared by the batch are a broadcast view, so a Monte Carlo over
    frontend matrix variants do not pay for ``J`` copies of one tone.

    Parameters
    ----------
    af : Frontend
        Converter whose input sources are evaluated.
    size : int
        Number of clock steps.

    Returns
    -------
    ndarray of float, shape (size, L, J)
        Clocked drive values for sampled inputs, source outputs otherwise.
    """
    L, J, dt = af.L, af.J, af.dt
    J_u = max([1] + [w.J for w in af.input])
    u = np.zeros((size, L, J_u))
    for ell, w in enumerate(af.input):
        if isinstance(w, Samples):
            u[:, ell] = w.held(size)
        else:
            X = w.trajectory(dt, size)
            C = np.broadcast_to(w.C[:, 0, 0], (w.J, w.N))
            u[:, ell] = np.einsum("jk,ijk->ij", C, X)
    return np.broadcast_to(u, (size, L, J)) if J_u != J else u


def _pack_batch_matrices(step_matrix: np.ndarray, W: int) -> np.ndarray:
    """Pack ``step_matrix[bW + j, k, r, c]`` as ``[b, k, c, r, j]``.

    ``b`` selects a group of ``W`` batch members, ``j`` a lane within it,
    ``r`` an output row, and ``c`` an input column. Keeping ``j`` last lets
    Numba vectorize across members. Shared matrices are repeated across lanes;
    the final group is padded if it has fewer than ``W`` members.

    Parameters
    ----------
    step_matrix : ndarray of float, shape (1 or J, K, rows, columns)
        Clocked frontend or twin maps.
    W : int
        Positive batch block width.

    Returns
    -------
    ndarray of float, shape (blocks, K, columns, rows, W)
        Packed matrices for batch blocks.
    """
    J_m, K, rows, columns = step_matrix.shape
    full = (
        np.broadcast_to(step_matrix, (max(J_m, W), K, rows, columns))
        if J_m == 1
        else step_matrix
    )
    J_pad = -(-full.shape[0] // W) * W
    padded = (
        full
        if J_pad == full.shape[0]
        else np.concatenate(
            [full, np.repeat(full[-1:], J_pad - full.shape[0], axis=0)], axis=0
        )
    )
    return np.ascontiguousarray(
        padded.reshape(J_pad // W, W, K, rows, columns).transpose(0, 2, 4, 3, 1)
    )


def simulate(
    af: Frontend,
    size: int,
    x0: Optional[ArrayLike],
    record: str | Sequence[str],
    engine: str,
    threads: Optional[int],
    verbose: bool,
    seed: Optional[int],
    *,
    discretization: Optional[Discretization] = None,
) -> dict[str, np.ndarray]:
    """Advance the augmented state, read the comparators, then quantize.

    For one batch member and phase, let ``A, B, C, D`` be the *discrete
    augmented* state-update and readout matrices, not the continuous-time
    frontend matrices. ``C`` is ``dz.C``; ``D`` is direct previous-decision
    feedthrough. Their equations are

    ``z_i = A z_{i-1} + B v_{i-1} + e_z``
    ``y_i = C z_i + D v_{i-1} + d_i``,  ``v_i = Q(y_i)``.

    Thus ``step_matrix = [[A, B], [C A, C B + D]]`` and
    ``[z_i; y_i] = step_matrix[j_m,k] [z_{i-1}; v_{i-1}] + [e_z; C e_z + d_i]``.
    ``C`` reads the *new* state. ``e_z`` is sampled-input and state-noise
    excitation; ``d_i`` is direct sampled input and comparator noise. The
    discretization exposes this composed matrix as ``dz.ABCD``. At ``i = 0``,
    quantize the initialized comparator input without applying the step matrix.

    Here ``k = i % K`` and ``j_m`` is 0 for a shared matrix or ``j`` for batch
    member ``j``. The twin similarly uses
    ``step_matrix_hat = [[A_hat, B_hat], [C_hat A_hat, C_hat B_hat + D_hat]]``;
    here ``C_hat`` includes zero columns for twin DAC memory. The twin state
    is driven by the recorded decisions; ``D_hat`` reads the held decisions
    directly.

    Dimensions (counts except ``dt``):

    - ``J``: parallel batch members; ``K``: cyclic clock phases.
    - ``L``: external input channels; ``M``: comparator inputs and decisions.
    - ``N``: frontend states; ``N_x``: all augmented frontend states.
    - ``N_s``: decision-driven DAC states and pulse memory.
    - ``N_hat = N + N_s``: twin state ``z_hat = [x_hat; DAC memory]``.
    - ``J_m``: matrix variants, either 1 or J; ``N_xv = N_x + M``: map size.
    - ``J_hat``: twin matrix variants, either 1 or J.
    - ``M_hat``: digital-twin readouts; ``dt``: step period in seconds.

    ``step_matrix`` has shape ``(J_m, K, N_xv, N_xv)``; its additive term
    ``[e_z; C e_z + d_i]`` has shape ``(size, N_xv, J)`` when present.
    ``C`` has shape ``(J_m, K, M, N_x)``;
    ``step_matrix_hat`` has shape ``(J_hat, K, N_hat + M_hat, N_hat + M)``.
    Records use ``(size, channels, J)``.

    Execution order:

    1. Discretize once in Python/SciPy, then precompute the additive term with NumPy;
       autonomous sources are already states in ``z``. Initialize ``z_0``
       and ``y_0``.
    2. At each later step, apply ``step_matrix`` and replay the same decisions in
       ``[z_hat_i; y_hat_i] = step_matrix_hat[j_hat,k] [z_hat_{i-1}; v_{i-1}]``.
       The twin starts from zero and has no quantizer of its own.
    3. Limit the first N states by slew and swing. With ``delta_x`` equal to
       bounded minus linear state, correct ``y_i`` by
       ``C[j_m,k,:,:N] @ delta_x``, then quantize and record the requested
       signals. The new ``v_i`` drives the next step.

    With ``engine="numpy"``, batched ``@`` uses BLAS (optimized compiled
    matrix multiplication) for the step and twin maps. With ``engine="numba"``,
    compiled kernels advance blocks of 16 batch
    members: explicit sums for maps up to 24 rows, also up to 64 rows when
    ``J_m = J``; larger maps call BLAS per block if shared, per member
    otherwise.
    Limits, quantization, and recording remain inside the recursion kernel.

    Parameters
    ----------
    af : Frontend
        Converter to simulate.
    size : int
        Positive number of clock steps.
    x0 : array_like of float, shape (N,), (N, 1), or (N, J), optional
        Initial design states for the ``J`` parallel batch members.
    record : str or sequence of str
        Extra result records, or ``"all"``.
    engine : {"numba", "numpy"}
        Compiled or reference simulation loop.
    threads : int, optional
        Numba worker count.
    verbose : bool
        Print the simulation estimate to stderr.
    seed : int or None
        Seed for this run's noise draws; ``None`` draws a fresh seed.
    discretization : Discretization, optional
        Prepared step maps for an internal batch sweep; defaults to ``None``
        (discretize ``af`` here).

    Returns
    -------
    dict[str, ndarray]
        Simulation records; see ``Frontend.simulate`` for keys and shapes.
    """
    if engine not in ("numba", "numpy"):
        raise ValueError(f"engine must be 'numba' or 'numpy', not {engine!r}")
    available = frozenset(("v", "u", "y_hat", "t", "x", "y", "x_hat"))
    records = (
        available
        if record == "all"
        else frozenset((record,) if isinstance(record, str) else record)
    )
    unknown = records - available
    if unknown:
        raise ValueError(f"unknown simulation records: {', '.join(sorted(unknown))}")
    record_x, record_y, record_x_hat = (key in records for key in ("x", "y", "x_hat"))
    M_hat = 1 if af.C_hat is None else af.C_hat.shape[-2]
    dz = af.discretize() if discretization is None else discretization
    step_matrix = dz.ABCD
    N, M, L, N_x, K, J, dt, J_m = af.N, af.M, af.L, dz.N_x, dz.K, af.J, af.dt, dz.J_m
    N_xv = N_x + M
    if J_m not in (1, J):
        raise ValueError(
            f"the matrices vary over {J_m} batch members but the batch has {J}"
        )
    packed = N_xv <= 24 or (J_m > 1 and N_xv <= _PACKED_PER_FRONTEND_MAX)
    # Float levels support arithmetic; nonbinary decisions may be fractional.
    levels = af.levels.astype(float)
    binary = bool(np.all(af.levels == 2))
    # Skip the state-limit and slew-limit work when neither limit is finite.
    clip = bool(
        np.isfinite(af.state_min).any()
        or np.isfinite(af.state_max).any()
        or np.isfinite(af.slew_rate).any()
    )
    # Samples and noise need time-indexed additive terms; autonomous
    # sources are already included in the discretized state update.
    has_excitation = (
        any(isinstance(w, Samples) for w in af.input)
        or af.state_covariance is not None
        or af.output_covariance is not None
        or bool(af._flicker)
    )
    threads_before = numba.get_num_threads()
    if threads is not None:
        numba.set_num_threads(threads)
    try:
        # -- the plan: requested records and working excitation
        v_dtype = np.int8 if binary else np.float64
        J_u = max([1] + [w.J for w in af.input])
        memory = {
            ("v" if "v" in records else "v workspace"): size
            * M
            * J
            * np.dtype(v_dtype).itemsize,
            "y": size * M * J * 8 if record_y else 0,
            "u": size * L * J_u * 8 if "u" in records else 0,
            "t": size * 8 if "t" in records else 0,
            "excitation": size * N_xv * J * 8 if has_excitation else 0,
            "x": size * N * J * 8 if record_x else 0,
            "x_hat": size * N * J * 8 if record_x_hat else 0,
            ("y_hat" if "y_hat" in records else "y_hat workspace"): size
            * M_hat
            * J
            * 8,
        }
        # rule-of-thumb costs of one 2021 server core: the exponentials of the
        # discretisation per batch member and phase, first touch of the record, and
        # the recursion's multiply-adds
        setup = J_m * K * 17e-6 * (N_x - dz.N_z) + 0.15e-9 * sum(
            b for b in memory.values() if b > 0
        )
        macs = size * J * N_xv**2
        n_threads = min(numba.get_num_threads(), -(-J // _W))
        if engine == "numpy":
            expected, path = setup + size * 4e-6 + macs * 0.5e-9, "numpy"
        elif not packed:
            expected, path = setup + size * 1.2e-6 + macs * 0.1e-9, "numba (BLAS)"
        elif n_threads > 1:
            expected = setup + macs * 0.2e-9 / (0.4 * n_threads)
            path = f"numba, {n_threads} threads"
        else:
            expected, path = setup + size * 60e-9 + macs * 0.3e-9, "numba"
        if verbose:
            noise_scratch = (
                (size - 1) * N * J * 8 if dz.noise_factor is not None else 0
            ) + (2 * size * M * J * 8 if af.output_covariance is not None else 0)
            sampled_count = sum(isinstance(w, Samples) for w in af.input)
            sample_scratch = (
                8 * J * (size * sampled_count + ((size + K - 1) // K) * max(N_x, M))
                if sampled_count
                else 0
            )
            _announce(
                af,
                size,
                N_xv,
                path,
                memory,
                expected,
                max(noise_scratch, sample_scratch),
            )

        # -- the start: the discretisation's x0, the design states from the argument
        # within the swing, and what the comparator sees at edge 0 (quantised by the loop)
        # e_z and e_y occupy the state and comparator rows; an empty first
        # axis tells the recursion that no sampled input or noise is present.
        excitation = np.zeros((size if has_excitation else 0, N_xv, J))
        if has_excitation:
            dz.precompute_excitation(excitation, np.random.default_rng(seed))
        xv0 = np.zeros((N_xv, J))
        xv0[:N_x] = dz.x0[np.arange(J) % dz.x0.shape[0]].T
        if x0 is not None:
            xv0[:N] = np.broadcast_to(
                np.asarray(x0, dtype=float).reshape(N, -1), (N, J)
            )
        np.clip(xv0[:N], af.state_min[:, None], af.state_max[:, None], out=xv0[:N])
        xv0[N_x:] = np.einsum("jmn,nj->mj", dz.C[:, 0], xv0[:N_x])
        if has_excitation:
            xv0[N_x:] += excitation[0, N_x:]
        v = np.zeros((size, M, J), v_dtype)
        # Each step's additions are consumed once. Reuse its state and comparator
        # slots for the completed records, then copy out only requested rows.
        x = (
            excitation[:, :N]
            if record_x and has_excitation
            else np.zeros((size if record_x else 0, N, J))
        )
        y = (
            excitation[:, N_x:]
            if record_y and has_excitation
            else np.zeros((size if record_y else 0, M, J))
        )
        x_hat = np.zeros((size if record_x_hat else 0, N, J))
        y_hat = np.zeros((size, M_hat, J))
        C_hat = af.C_hat if af.C_hat is not None else -np.eye(N)[None, None, -1:]
        D_hat = af.D_hat if af.D_hat is not None else np.zeros((1, 1, M_hat, M))
        C_hat = np.ascontiguousarray(C_hat[:, np.arange(K) % C_hat.shape[1]])
        D_hat = np.ascontiguousarray(D_hat[:, np.arange(K) % D_hat.shape[1]])
        if C_hat.shape[0] not in (1, J):
            raise ValueError(f"C_hat has {C_hat.shape[0]} batch members, the batch {J}")
        if D_hat.shape[0] not in (1, J):
            raise ValueError(f"D_hat has {D_hat.shape[0]} batch members, the batch {J}")
        if D_hat.shape[-2:] != (M_hat, M):
            raise ValueError(f"D_hat must be ({M_hat}, {M})")
        # Keep the frontend and decision-driven DAC states; exclude input states.
        N_hat = N + dz.N_s
        twin_columns = np.r_[np.arange(N_hat), N_x + np.arange(M)]
        state_hat = step_matrix[..., :N_hat, twin_columns]
        J_hat = max(dz.J_m, C_hat.shape[0], D_hat.shape[0])
        readout_hat = np.broadcast_to(
            C_hat @ state_hat[..., :N, :], (J_hat, K, M_hat, N_hat + M)
        ).copy()
        readout_hat[..., N_hat:] += D_hat
        step_matrix_hat = np.ascontiguousarray(
            np.concatenate(
                (
                    np.broadcast_to(state_hat, (J_hat, K, N_hat, N_hat + M)),
                    readout_hat,
                ),
                axis=2,
            )
        )
        x_min, x_max = af.state_min, af.state_max
        slew = np.broadcast_to(dt * af.slew_rate, (K, N))
        args = (
            step_matrix,
            step_matrix_hat,
            excitation,
            dz.C,
            xv0,
            v,
            x,
            x_hat,
            y,
            y_hat,
            x_min,
            x_max,
            slew,
            levels,
            clip,
        )
        if engine == "numpy":
            _numpy_loop(*args)
        else:
            # One worker advances up to _W members of J. Small matrices use
            # packed batch lanes; large ones call np.dot and use only the
            # dummy arrays' last-axis width.
            packed_step_matrix = (
                _pack_batch_matrices(step_matrix, _W)
                if packed
                else np.zeros((1, K, 1, 1, _W))
            )
            packed_step_matrix_hat = (
                _pack_batch_matrices(step_matrix_hat, _W)
                if packed
                else np.zeros((1, K, 1, 1, _W))
            )
            _linear_loop(
                *args[:2], packed_step_matrix, packed_step_matrix_hat, *args[2:]
            )
    finally:
        numba.set_num_threads(threads_before)
    out = {}
    if "v" in records:
        out["v"] = v
    if "u" in records:
        out["u"] = _input_record(af, size)
    if "y_hat" in records:
        out["y_hat"] = y_hat
    if "t" in records:
        out["t"] = np.arange(size) * dt
    if record_x:
        out["x"] = x.copy() if has_excitation else x
    if record_y:
        out["y"] = y.copy() if has_excitation else y
    if record_x_hat:
        out["x_hat"] = x_hat
    return out


def _numpy_loop(
    step_matrix,
    step_matrix_hat,
    excitation,
    C,
    xv0,
    v,
    x,
    x_hat,
    y,
    y_hat,
    x_min,
    x_max,
    slew,
    levels,
    clip,
):
    """Reference simulation loop with batches ``(J, N_x + M, 1)``.

    Parameters
    ----------
    step_matrix : ndarray, shape (J_m, K, N_x + M, N_x + M)
        Step maps.
    step_matrix_hat : ndarray, shape (1 or J, K, N_hat + M_hat, N_hat + M)
        ``[z_hat[i]; y_hat[i]]`` from ``[z_hat[i-1]; v[i-1]]``.
    excitation : ndarray, shape (size, N_x + M, J)
        Precomputed sampled-input and noise terms, or an empty array.
    C : ndarray, shape (J_m, K, M, N_x)
        Corrects the comparator input after state clipping.
    xv0 : ndarray, shape (N_x + M, J)
        Initial augmented state and previous decisions.
    v : ndarray, shape (size, M, J)
        Decision output buffer.
    x : ndarray, shape (size, N, J)
        Optional design-state output buffer.
    x_hat : ndarray, shape (size or 0, N, J)
        Optional twin state record.
    y : ndarray, shape (size or 0, M, J)
        Optional comparator-input output buffer.
    y_hat : ndarray, shape (size, M_hat, J)
        Twin readout record.
    x_min, x_max : ndarray, shape (N,)
        State lower and upper bounds.
    slew : ndarray, shape (K, N)
        Per-phase state slew limits.
    levels : ndarray of float, shape (M,)
        Quantizer levels for each comparator.
    clip : bool
        Apply state limits and slew limits.

    Returns
    -------
    None
        Updates the supplied simulation record arrays in place.
    """
    size, M, J = v.shape
    N, N_x, K = x_min.shape[0], xv0.shape[0] - M, step_matrix.shape[1]
    scale, odd = (levels - 1.0)[:, None], (levels % 2.0)[:, None]
    N_hat = step_matrix_hat.shape[3] - M
    M_hat = step_matrix_hat.shape[2] - N_hat
    has_excitation = excitation.shape[0] > 0
    binary = bool(np.all(levels == 2))
    negative = np.empty((M, J), dtype=bool) if binary else None
    previous = np.ascontiguousarray(xv0.T)[:, :, None]  # initialized [z[0]; y[0]]
    current = np.empty_like(previous)
    np.copyto(current, previous)
    z_hatv = np.zeros((J, N_hat + M, 1))  # [z_hat; v], twin starts from rest
    zy_hat = np.zeros((J, N_hat + M_hat, 1))

    for i in range(size):
        k = i % K
        if i:
            np.matmul(step_matrix[:, k], previous, out=current)
            np.matmul(step_matrix_hat[:, k], z_hatv, out=zy_hat)
            if has_excitation:
                current += excitation[i].T[:, :, None]

        if clip:
            linear_state, prior_state = current[:, :N, 0], previous[:, :N, 0]
            bounded_state = np.clip(
                prior_state + np.clip(linear_state - prior_state, -slew[k], slew[k]),
                x_min,
                x_max,
            )
            # The comparator must see the bounded state, not the linear one.
            current[:, N_x:] += (
                C[:, k, :, :N] @ (bounded_state - linear_state)[:, :, None]
            )
            current[:, :N, 0] = bounded_state

        comparator_input = current[:, N_x:, 0].T  # (M, J)
        if y.shape[0]:
            y[i] = comparator_input
        if binary:
            np.less(comparator_input, 0, out=negative)
            v[i].fill(1)
            np.copyto(v[i], -1, where=negative)
        else:
            # Quantize to odd/even codes, then normalize to [-1, 1].
            code = 2.0 * np.floor((scale * comparator_input + odd) / 2.0) + 1.0 - odd
            v[i] = np.where(
                scale == 1.0,
                np.where(comparator_input < 0, -1.0, 1.0),
                np.clip(code, -scale, scale) / scale,
            )
        if x.shape[0]:
            x[i] = current[:, :N, 0].T
        if x_hat.shape[0]:
            x_hat[i] = zy_hat[:, :N, 0].T
        y_hat[i] = zy_hat[:, N_hat:, 0].T

        current[:, N_x:, 0] = v[i].T  # [z[i]; v[i]] becomes the next input
        previous, current = current, previous
        z_hatv[:, :N_hat] = zy_hat[:, :N_hat]
        z_hatv[:, N_hat:, 0] = v[i].T


@_njit(cache=True, fastmath=False)
def _block_loop(
    b,
    step_matrix,
    step_matrix_hat,
    packed_step_matrix,
    packed_step_matrix_hat,
    excitation,
    C,
    xv0,
    v,
    x,
    x_hat,
    y,
    y_hat,
    x_min,
    x_max,
    slew,
    levels,
    clip,
):
    """Run the :func:`_numpy_loop` recursion for up to ``W`` batch members.

    Only the matrix products differ: small maps use explicit sums over batch
    lanes, while large maps use ``np.dot``. The twin has no input or quantizer.

    Parameters
    ----------
    b : int
        Zero-based block of batch members to advance.
    step_matrix : ndarray, shape (J_m, K, N_x + M, N_x + M)
        Step maps.
    step_matrix_hat : ndarray, shape (1 or J, K, N_hat + M_hat, N_hat + M)
        Twin state and readout map.
    packed_step_matrix : ndarray, shape (blocks, K, N_x + M, N_x + M, W)
        Packed step maps, or a width-only dummy when using ``np.dot``.
    packed_step_matrix_hat : ndarray, shape (blocks, K, N_hat + M, N_hat + M_hat, W)
        Packed twin maps, or a width-only dummy when using ``np.dot``.
    excitation : ndarray, shape (size, N_x + M, J)
        Precomputed sampled-input and noise terms, or an empty array.
    C : ndarray, shape (J_m, K, M, N_x)
        Corrects the comparator input after state clipping.
    xv0 : ndarray, shape (N_x + M, J)
        Initial augmented state.
    v : ndarray, shape (size, M, J)
        Decision output buffer.
    x : ndarray, shape (size, N, J)
        Design-state output buffer.
    x_hat : ndarray, shape (size or 0, N, J)
        Optional twin state record.
    y : ndarray, shape (size or 0, M, J)
        Comparator-input output buffer.
    y_hat : ndarray, shape (size, M_hat, J)
        Twin readout record.
    x_min, x_max : ndarray, shape (N,)
        State bounds.
    slew : ndarray, shape (K, N)
        Per-phase state slew limits.
    levels : ndarray of float, shape (M,)
        Quantizer levels for each comparator.
    clip : bool
        Apply state bounds and slew limits.

    Returns
    -------
    None
        Updates one block of supplied simulation records in place.
    """
    size, M, J = v.shape
    N_xv = xv0.shape[0]
    N_x, N = N_xv - M, x_min.shape[0]
    J_m, K = step_matrix.shape[:2]
    W = packed_step_matrix.shape[4]
    has_exc = excitation.shape[0] > 0
    record_x, record_y = x.shape[0] > 0, y.shape[0] > 0
    N_hat = step_matrix_hat.shape[3] - M
    M_hat = step_matrix_hat.shape[2] - N_hat
    record_x_hat = x_hat.shape[0] > 0
    shared_hat = step_matrix_hat.shape[0] == 1
    packed = N_xv <= 24 or (J_m > 1 and N_xv <= _PACKED_PER_FRONTEND_MAX)
    # Lane j in this block is global batch member batch_start + j.
    # Shared matrices use batch/block index 0; varying matrices use their own.
    batch_start = b * W
    lanes = min(W, J - batch_start)
    shared = J_m == 1
    matrix_block, matrix_batch = b * (not shared), batch_start * (not shared)
    # Allocate once per block; reuse these buffers at every sample edge.
    # As in _numpy_loop: previous=[z; v], current=[z; y], then swap.
    previous, current = np.empty((N_xv, lanes)), np.empty((N_xv, lanes))
    z_hatv = np.zeros((N_hat + M, lanes))
    zy_hat = np.zeros((N_hat + M_hat, lanes))
    if not packed and not shared:
        input_column = np.empty((N_xv, 1))
        output_column = np.empty((N_xv, 1))
    if not packed and not shared_hat:
        twin_input_column = np.empty((N_hat + M, 1))
        twin_output_column = np.empty((N_hat + M_hat, 1))
    for r in range(N_xv):
        for j in range(lanes):
            previous[r, j] = xv0[r, batch_start + j]
            current[r, j] = previous[r, j]
    for i in range(size):
        k = i % K
        # 1. Form [z_i; y_i]. Edge 0 uses the supplied initial state/readout.
        if i:
            if packed and lanes == 1:
                # One lane: sum the small matrix directly, without padded lanes.
                for r in range(N_xv):
                    acc = 0.0
                    for c in range(N_xv):
                        acc += step_matrix[matrix_batch, k, r, c] * previous[c, 0]
                    current[r, 0] = acc + (
                        excitation[i, r, batch_start] if has_exc else 0.0
                    )
            elif packed:
                # Start with excitation (or zero), then add the matrix product.
                for r in range(N_xv):
                    for j in range(lanes):
                        current[r, j] = (
                            excitation[i, r, batch_start + j] if has_exc else 0.0
                        )
                for c in range(N_xv):
                    for r in range(N_xv):
                        for j in range(lanes):
                            current[r, j] += (
                                packed_step_matrix[matrix_block, k, c, r, j]
                                * previous[c, j]
                            )
            else:
                # Large matrix: one np.dot per block if shared, per lane otherwise.
                if shared:
                    np.dot(step_matrix[0, k], previous, current)
                else:
                    for j in range(lanes):
                        for r in range(N_xv):
                            input_column[r, 0] = previous[r, j]
                        np.dot(
                            step_matrix[batch_start + j, k], input_column, output_column
                        )
                        for r in range(N_xv):
                            current[r, j] = output_column[r, 0]
                if has_exc:
                    for r in range(N_xv):
                        for j in range(lanes):
                            current[r, j] += excitation[i, r, batch_start + j]
        # 2. Bound x_i; C times the state change corrects the comparator input.
        if clip:
            for r in range(N):
                for j in range(lanes):
                    x_prev = previous[r, j]
                    step = min(max(current[r, j] - x_prev, -slew[k, r]), slew[k, r])
                    x_clip = min(max(x_prev + step, x_min[r]), x_max[r])
                    if x_clip != current[r, j]:
                        for m in range(M):
                            current[N_x + m, j] += C[
                                (matrix_batch + j) * (not shared), k, m, r
                            ] * (x_clip - current[r, j])
                        current[r, j] = x_clip
        # 3. Record y_i before replacing its rows with decisions v_i.
        for m in range(M):
            scale, odd = levels[m] - 1.0, levels[m] % 2.0
            for j in range(lanes):
                y_mj = current[N_x + m, j]
                if record_y:
                    y[i, m, batch_start + j] = y_mj
                if scale == 1.0:
                    current[N_x + m, j] = -1.0 if y_mj < 0 else 1.0
                else:
                    q = 2.0 * np.floor((scale * y_mj + odd) / 2.0) + 1.0 - odd
                    current[N_x + m, j] = min(max(q, -scale), scale) / scale
                v[i, m, batch_start + j] = current[N_x + m, j]
        if record_x:
            for r in range(N):
                for j in range(lanes):
                    x[i, r, batch_start + j] = current[r, j]

        # 4. Replay v_{i-1} in the twin, using the same product choices.
        if i > 0:
            if packed and lanes == 1:
                j_hat = batch_start * (not shared_hat)
                for r in range(N_hat + M_hat):
                    acc = 0.0
                    for c in range(N_hat + M):
                        acc += step_matrix_hat[j_hat, k, r, c] * z_hatv[c, 0]
                    zy_hat[r, 0] = acc
            elif packed:
                b_hat = b * (not shared_hat)
                for r in range(N_hat + M_hat):
                    for j in range(lanes):
                        zy_hat[r, j] = 0.0
                for c in range(N_hat + M):
                    for r in range(N_hat + M_hat):
                        for j in range(lanes):
                            zy_hat[r, j] += (
                                packed_step_matrix_hat[b_hat, k, c, r, j] * z_hatv[c, j]
                            )
            elif shared_hat:
                np.dot(step_matrix_hat[0, k], z_hatv, zy_hat)
            else:
                for j in range(lanes):
                    for r in range(N_hat + M):
                        twin_input_column[r, 0] = z_hatv[r, j]
                    np.dot(
                        step_matrix_hat[batch_start + j, k],
                        twin_input_column,
                        twin_output_column,
                    )
                    for r in range(N_hat + M_hat):
                        zy_hat[r, j] = twin_output_column[r, 0]
        # 5. Record the twin, then carry [z_hat_i; v_i] to the next step.
        if record_x_hat:
            for r in range(N):
                for j in range(lanes):
                    x_hat[i, r, batch_start + j] = zy_hat[r, j]
        for m in range(M_hat):
            for j in range(lanes):
                y_hat[i, m, batch_start + j] = zy_hat[N_hat + m, j]
        for r in range(N_hat):
            for j in range(lanes):
                z_hatv[r, j] = zy_hat[r, j]
        for m in range(M):
            for j in range(lanes):
                z_hatv[N_hat + m, j] = current[N_x + m, j]
        previous, current = current, previous  # [z_i; v_i] becomes previous


@_njit(parallel=True, cache=True, fastmath=False)
def _linear_loop(
    step_matrix,
    step_matrix_hat,
    packed_step_matrix,
    packed_step_matrix_hat,
    excitation,
    C,
    xv0,
    v,
    x,
    x_hat,
    y,
    y_hat,
    x_min,
    x_max,
    slew,
    levels,
    clip,
):
    """The numpy loop above, blocks of ``W`` batch members in parallel.

    Small models (``N_x + M <= 24``) and per-frontend models up to 64 rows
    multiply with explicit loops. One batch member sums over ``c`` innermost; a
    block sums over ``j`` innermost from ``packed_step_matrix`` so its batch
    lanes vectorise.
    Larger models with shared matrices call ``np.dot`` once per block. Larger
    per-frontend models call it once per batch member. Different
    multiplication orders can differ slightly in floating-point results.
    ``step_matrix[j, k]`` is one matrix per batch member when its first axis is ``J``.
    Absent features are empty arrays; ``x`` and ``y`` are recorded when present.

    Parameters
    ----------
    step_matrix : ndarray, shape (J_m, K, N_x + M, N_x + M)
        Step maps.
    step_matrix_hat : ndarray, shape (1 or J, K, N_hat + M_hat, N_hat + M)
        Twin state and readout map.
    packed_step_matrix : ndarray, shape (blocks, K, N_x + M, N_x + M, W)
        Packed step maps, or a width-only dummy when using ``np.dot``.
    packed_step_matrix_hat : ndarray, shape (blocks, K, N_hat + M, N_hat + M_hat, W)
        Packed twin maps, or a width-only dummy when using ``np.dot``.
    excitation : ndarray, shape (size, N_x + M, J)
        Precomputed sampled-input and noise terms, or an empty array.
    C : ndarray, shape (J_m, K, M, N_x)
        Corrects the comparator input after state clipping.
    xv0 : ndarray, shape (N_x + M, J)
        Initial augmented state.
    v : ndarray, shape (size, M, J)
        Decision output buffer.
    x : ndarray, shape (size, N, J)
        Design-state output buffer.
    x_hat : ndarray, shape (size or 0, N, J)
        Optional twin state record.
    y : ndarray, shape (size or 0, M, J)
        Comparator-input output buffer.
    y_hat : ndarray, shape (size, M_hat, J)
        Twin readout record.
    x_min, x_max : ndarray, shape (N,)
        State bounds.
    slew : ndarray, shape (K, N)
        Per-phase state slew limits.
    levels : ndarray of float, shape (M,)
        Quantizer levels for each comparator.
    clip : bool
        Apply state bounds and slew limits.

    Returns
    -------
    None
        Updates the supplied simulation record arrays in place.
    """
    block_count = (
        v.shape[2] + packed_step_matrix.shape[4] - 1
    ) // packed_step_matrix.shape[4]
    if block_count == 1:  # one block: the serial code path
        _block_loop(
            0,
            step_matrix,
            step_matrix_hat,
            packed_step_matrix,
            packed_step_matrix_hat,
            excitation,
            C,
            xv0,
            v,
            x,
            x_hat,
            y,
            y_hat,
            x_min,
            x_max,
            slew,
            levels,
            clip,
        )
    else:
        for b in _prange(block_count):
            _block_loop(
                b,
                step_matrix,
                step_matrix_hat,
                packed_step_matrix,
                packed_step_matrix_hat,
                excitation,
                C,
                xv0,
                v,
                x,
                x_hat,
                y,
                y_hat,
                x_min,
                x_max,
                slew,
                levels,
                clip,
            )
