"""Model-based estimators: state-space models driven by the decisions, and the Wiener filter.

Everything here is one recursion, ``m[i] = A[k] m[i-1] + B[k] v[i-1]`` read
out as ``W[k] m[i]``, run by one kernel: :class:`StateSpaceFilter` with the
simulation's own ``A_d``, ``B_d`` is the digital twin (output ``G u``), with
the leaked matrices the causal half of the :class:`WienerFilter`, which adds
the backward message and folds the inverse of ``G`` and its roll-off into
the leak.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numba import get_num_threads as _get_num_threads
from numba import njit as _njit
from numba import prange as _prange
from scipy.linalg import expm, solve_continuous_are, solve_discrete_are
from scipy.signal import fftconvolve

from .frontend import DAC, Constant, Discretization, Frontend, StateSpace
from .simulation import _W

logger = logging.getLogger(__name__)


def _care(A: np.ndarray, Q: np.ndarray, eta2: float) -> np.ndarray:
    """Stabilising continuous Riccati solution, scaled for conditioning.

    With ``w = max(abs(eig(A)))`` (or 1) and ``V = w * eta2 * X``, solve
    ``(A/w).T X + X (A/w) - X X + Q/(w*w*eta2) = 0``.
    Unit measurement covariance avoids ill-conditioning at large ``eta2``.

    Parameters
    ----------
    A : ndarray of float, shape (N, N)
        Continuous-time state matrix.
    Q : ndarray of float, shape (N, N)
        Symmetric state-noise intensity.
    eta2 : float
        Positive regularization variance.

    Returns
    -------
    ndarray of float, shape (N, N)
        Stabilizing continuous-time Riccati solution.
    """
    N = A.shape[0]
    w = float(np.max(np.abs(np.linalg.eigvals(A)))) or 1.0
    scale = w * eta2
    Q_scaled = Q / (w * w * eta2)
    for balanced in (True, False):
        try:
            return scale * solve_continuous_are(
                A / w, np.eye(N), Q_scaled, np.eye(N), balanced=balanced
            )
        except Exception:  # noqa: BLE001 - scipy raises several types on ill-conditioning
            continue
    Z = np.block(
        [[A / w, -np.eye(N)], [-Q_scaled, -A.T / w]]
    )  # Hamiltonian stable subspace
    lam, V = np.linalg.eig(Z)
    V = V[:, lam.argsort()]
    X = np.real(V[N:, :N] @ np.linalg.pinv(V[:N, :N]))
    return scale * 0.5 * (X + X.T)


def _dare(A_d: np.ndarray, Q: np.ndarray, eta2: float) -> np.ndarray:
    """Stationary discrete Riccati solution, normalized by ``||Q||``.

    Parameters
    ----------
    A_d : ndarray of float, shape (N, N)
        One-step state transition matrix.
    Q : ndarray of float, shape (N, N)
        Symmetric per-step state-noise covariance.
    eta2 : float
        Positive regularization variance.

    Returns
    -------
    ndarray of float, shape (N, N)
        Stationary discrete-time Riccati solution.
    """
    N = A_d.shape[0]
    s = float(np.linalg.norm(Q)) or 1.0
    return s * solve_discrete_are(A_d.T, np.eye(N), Q / s, (eta2 / s) * np.eye(N))


def _reversed(w: DAC) -> DAC:
    """The DAC pulse run backwards in time within one period.

    Parameters
    ----------
    w : DAC
        Pulse with a window contained in one clock period.

    Returns
    -------
    DAC
        Equivalent pulse reversed within one clock period.
    """
    if w.filter is not None:
        raise NotImplementedError(
            "the Wiener filter needs pulse-shaped DACs without filters"
        )
    d, wd = w.window or (0.0, 1.0)
    if d + wd > 1.0 + 1e-12:
        raise NotImplementedError(
            "the Wiener filter needs DAC pulses within one clock period"
        )
    x_end = np.stack(
        [
            expm(a * wd) @ x
            for a, x in zip(
                np.broadcast_to(w.A[:, 0], (w.J, w.N, w.N)),
                np.broadcast_to(w.x0, (w.J, w.N)),
            )
        ]
    )
    return DAC(StateSpace(-w.A, w.B, w.C, w.D, x_end), window=(1.0 - d - wd, wd))


def _cyclic_batch(a: np.ndarray) -> np.ndarray:
    """Normalize a matrix to ``(J, K, rows, cols)`` form before broadcasting.

    Parameters
    ----------
    a : array_like of float, shape (rows, cols), (K, rows, cols), or (J, K, rows, cols)
        State, input, or readout matrix.

    Returns
    -------
    ndarray of float, shape (J, K, rows, cols)
        Matrix with explicit batch and phase axes; float conversion may copy.
    """
    a = np.asarray(a, dtype=float)
    a = a.reshape((1,) * (4 - a.ndim) + a.shape) if a.ndim < 4 else a
    return a


def _message(A, B, W, v, out, forward: bool) -> None:
    """``out[i] += W[k] m[i]`` for ``m[i] = A[k] m[i-1] + B[k] v[i-1]`` (forward, ``m[0] = 0``)
    or ``m[i] = A[k] m[i+1] + B[k] v[i]`` (backward, ``m[size] = 0``), ``k = i % K``; one
    matrix per column when the first axis is ``J``. Cyclic and batch axes broadcast.

    Parameters
    ----------
    A : ndarray, shape (J, K, N, N)
        State transition matrices.
    B : ndarray, shape (J, K, N, M)
        Decision-to-state matrices.
    W : ndarray, shape (J, K, L, N)
        State readout matrices; leading axes may broadcast.
    v : ndarray, shape (size, M, J)
        Decision samples.
    out : ndarray, shape (size, L, J)
        Output accumulation buffer, modified in place.
    forward : bool
        Select forward or backward recursion.

    Returns
    -------
    None
        Adds the recursive message output to ``out`` in place.
    """
    A, B, W = (_cyclic_batch(a) for a in (A, B, W))
    K = max(a.shape[1] for a in (A, B, W))
    J_m = max(a.shape[0] for a in (A, B, W))
    A, B, W = (
        np.ascontiguousarray(np.broadcast_to(a, (J_m, K) + a.shape[2:]))
        for a in (A, B, W)
    )
    # Small per-member models benefit from contiguous state vectors. Shared
    # models keep SIMD across J unless every member can have its own worker.
    J = v.shape[2]
    small = 1 < J <= _W and (J_m > 1 or J <= _get_num_threads())
    loop = _message_small_batch if small else _message_loop
    loop(A, B, W, v, out, forward)


@_njit(parallel=True, cache=True)
def _message_loop(A, B, W, v, out, forward):
    """The recursion above in blocks of ``_W`` columns, the column sums sweeping whole rows.

    One column sums over the contiguous last axis instead, as the simulation kernel does.

    Parameters
    ----------
    A : ndarray of float, shape (J_m, K, N, N)
        State transition matrices.
    B : ndarray of float, shape (J_m, K, N, M)
        Decision-to-state matrices.
    W : ndarray of float, shape (J_m, K, L, N)
        State readout matrices.
    v : ndarray, shape (size, M, J)
        Decision samples.
    out : ndarray, shape (size, L, J)
        Output accumulation buffer.
    forward : bool
        Select forward or backward recursion.

    Returns
    -------
    None
        Updates the supplied output buffer in place.
    """
    size, M, J = v.shape
    J_m, K, N, _ = A.shape
    L = W.shape[2]
    if J == 1:
        m = np.zeros(N)
        mn = np.empty(N)
        for step in range(size):
            i = step if forward else size - 1 - step
            if forward and i == 0:
                continue
            k = i % K
            s = i - 1 if forward else i
            for r in range(N):
                acc = 0.0
                for c in range(N):
                    acc += A[0, k, r, c] * m[c]
                for c in range(M):
                    acc += B[0, k, r, c] * v[s, c, 0]
                mn[r] = acc
            for r in range(N):
                m[r] = mn[r]
            for ell in range(L):
                acc = 0.0
                for r in range(N):
                    acc += W[0, k, ell, r] * m[r]
                out[i, ell, 0] += acc
        return
    n_blk = (J + _W - 1) // _W
    for b in _prange(n_blk):
        j0 = b * _W
        Wd = min(_W, J - j0)
        m = np.zeros((N, Wd))
        mn = np.empty((N, Wd))
        for step in range(size):
            i = step if forward else size - 1 - step
            if forward and i == 0:  # m[0] = 0 contributes nothing
                continue
            k = i % K
            s = i - 1 if forward else i  # the decision the step reads
            for r in range(N):
                for j in range(Wd):
                    mn[r, j] = 0.0
            for c in range(N):
                for r in range(N):
                    for j in range(Wd):
                        jm = j0 + j if J_m > 1 else 0
                        mn[r, j] += A[jm, k, r, c] * m[c, j]
            for c in range(M):
                for r in range(N):
                    for j in range(Wd):
                        jm = j0 + j if J_m > 1 else 0
                        mn[r, j] += B[jm, k, r, c] * v[s, c, j0 + j]
            for r in range(N):
                for j in range(Wd):
                    m[r, j] = mn[r, j]
            for ell in range(L):
                for j in range(Wd):
                    jm = j0 + j if J_m > 1 else 0
                    acc = 0.0
                    for r in range(N):
                        acc += W[jm, k, ell, r] * m[r, j]
                    out[i, ell, j0 + j] += acc


@_njit(parallel=True, cache=True)
def _message_small_batch(A, B, W, v, out, forward):
    """Run independent batch members in parallel for ``1 < J <= _W``.

    Parameters
    ----------
    A, B, W : ndarray of float
        Matrices with shapes (J_m, K, N, N), (J_m, K, N, M), and
        (J_m, K, L, N), respectively; J_m is 1 or J.
    v : ndarray, shape (size, M, J)
        Comparator decisions.
    out : ndarray of float, shape (size, L, J)
        Output accumulation buffer.
    forward : bool
        Select the forward or backward recursion defined in _message.

    Returns
    -------
    None
        Add W m to out in place. State buffers are reused at every step.
    """
    size, M, J = v.shape
    J_m, K, N, _ = A.shape
    L = W.shape[2]
    for j in _prange(J):
        jm = np.int64(j) if J_m > 1 else 0
        m = np.zeros(N)
        mn = np.empty(N)
        for step in range(size):
            i = step if forward else size - 1 - step
            if forward and i == 0:
                continue
            k = i % K
            s = i - 1 if forward else i
            for r in range(N):
                acc = 0.0
                for c in range(N):
                    acc += A[jm, k, r, c] * m[c]
                for c in range(M):
                    acc += B[jm, k, r, c] * v[s, c, j]
                mn[r] = acc
            for ell in range(L):
                acc = 0.0
                for r in range(N):
                    acc += W[jm, k, ell, r] * mn[r]
                out[i, ell, j] += acc
            m, mn = mn, m


class WienerFilter:
    """The analytical estimator ``u_hat = W (m_b - m_f)`` of a :class:`Frontend`.

    ``m_f`` and ``m_b`` are the forward and backward state messages of the
    control signals through the loop filter leaked by ``V / eta2``, where
    ``V_f``, ``V_b`` solve the Riccati equations of the input statistics and
    ``eta2`` trades bandwidth against noise suppression. Attributes ``W``,
    ``A_f``, ``B_f``, ``A_b``, ``B_b`` are the readout and the two one-step maps.

    Parameters
    ----------
    frontend : Frontend
        Converter whose single-cycle matrices and DAC pulses define the filter.
    eta2 : float
        Positive regularization variance in squared input units.

    >>> af, OSR = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
    >>> af.dac = DAC(Constant(1.0))  # this estimator uses immediate NRZ
    >>> wf = af.wiener_filter(OSR=OSR)
    >>> wf.W.shape, wf.taps().shape[1:]
    ((1, 3), (3, 1))
    >>> v = af.simulate(1 << 10)["v"]
    >>> bool(np.allclose(wf(v), wf(v, method="state-space"), atol=1e-12))
    True

    Returns
    -------
    WienerFilter
        Initialized analytical estimator.
    """

    def __init__(self, frontend: Frontend, eta2: float):
        self.frontend = frontend
        self._taps = None
        self.eta2 = eta2

    @property
    def eta2(self) -> float:
        """Current Wiener regularization variance.

        Returns
        -------
        float
            Positive variance in squared input units.
        """
        return self._eta2

    @eta2.setter
    def eta2(self, eta2: float) -> None:
        """Recompute filter maps for a new regularization variance.

        Parameters
        ----------
        eta2 : float
            Positive regularization variance in squared input units.

        Returns
        -------
        None
            Updates the regularization variance and recomputes the filter maps.
        """
        if not eta2 > 0:
            raise ValueError("eta2 must be positive")
        self._eta2, self._taps = float(eta2), None
        af, L, N = self.frontend, self.frontend.L, self.frontend.N
        if af.K != 1:
            raise NotImplementedError("the Wiener filter needs a non-cyclic model")
        A_all, B_all = af.A[:, 0], af.B[:, 0]  # (J_m, N, N), (J_m, N, L + M)
        J_m = max(A_all.shape[0], B_all.shape[0])
        W, A_f, B_f, A_b, B_b = [], [], [], [], []
        for j in range(J_m):  # one Riccati pair per frontend matrix set
            A = A_all[j % len(A_all)]
            B_u, B_v = B_all[j % len(B_all), :, :L], B_all[j % len(B_all), :, L:]
            Q = B_u @ B_u.T
            if af.discrete:
                A_inv = np.linalg.inv(A)
                V_f, V_b = _dare(A, Q, eta2), _dare(A_inv, Q, eta2)
                K_f = np.linalg.solve(np.eye(N) + V_f / eta2, np.eye(N))
                K_b = np.linalg.solve(np.eye(N) + V_b / eta2, np.eye(N))
                maps = (A @ K_f, B_v, A_inv @ K_b, -A_inv @ B_v)
            else:
                V_f, V_b = _care(A.T, Q, eta2), _care(-A.T, Q, eta2)
                maps = (
                    A - V_f / eta2,
                    B_v,
                    -A - V_b / eta2,
                    -B_v,
                )  # leaked, continuous
            W.append(np.linalg.solve(V_f + V_b, B_u).T)
            for store, value in zip((A_f, B_f, A_b, B_b), maps):
                store.append(value)
        W, A_f, B_f, A_b, B_b = (np.array(a) for a in (W, A_f, B_f, A_b, B_b))
        if not af.discrete:  # one-step maps of the leaked filters, all columns at once
            A_f, B_f = self._leaked(A_f, B_f, af.dac)
            A_b, B_b = self._leaked(A_b, B_b, [_reversed(w) for w in af.dac])
        # 2-D for one frontend, (J, 1, ., .) for a Monte Carlo
        self.W, self.A_f, self.B_f, self.A_b, self.B_b = (
            a[0] if J_m == 1 else a[:, None] for a in (W, A_f, B_f, A_b, B_b)
        )

    def _leaked(
        self, A: np.ndarray, B_v: np.ndarray, dac: list[DAC]
    ) -> tuple[np.ndarray, np.ndarray]:
        """One-step maps of leaked loop filters driven by DAC pulses.

        Parameters
        ----------
        A : ndarray of float, shape (J, N, N)
            Continuous-time leaked state matrices.
        B_v : ndarray of float, shape (J, N, M)
            Decision-to-state matrices.
        dac : list of DAC, length M
            Actual pulse shape for each control channel.

        Returns
        -------
        A_d : ndarray of float, shape (J, N, N)
            Leaked one-step state matrices.
        B_d : ndarray of float, shape (J, N, M)
            Leaked decision-to-state maps.
        """
        af = self.frontend
        B, C = af.B[0, 0], af.C[0, 0]
        B_full = np.concatenate(
            (np.broadcast_to(B[:, : af.L], B_v.shape[:1] + B[:, : af.L].shape), B_v),
            axis=2,
        )
        leaked = Frontend(A[:, None], B_full[:, None], C, dt=af.dt, dac=dac)
        dz = Discretization(leaked)
        if dz.N_x != af.N:
            raise NotImplementedError(
                "the Wiener filter needs DAC pulses within one clock period"
            )
        return dz.ABCD[:, 0, : af.N, : af.N], dz.ABCD[:, 0, : af.N, af.N :]

    @property
    def J_m(self) -> int:
        """Frontend matrix sets the filter was built for: 1 or Monte Carlo columns.

        Returns
        -------
        int
            Number of distinct frontend columns in the estimator.
        """
        return 1 if self.W.ndim == 2 else self.W.shape[0]

    @property
    def L(self) -> int:
        """Number of estimated input channels.

        Returns
        -------
        int
            Input-channel count ``L``.
        """
        return self.frontend.L

    @property
    def M(self) -> int:
        """Number of comparator decision channels.

        Returns
        -------
        int
            Decision-channel count ``M``.
        """
        return self.frontend.M

    @property
    def N(self) -> int:
        """Number of filter states.

        Returns
        -------
        int
            State dimension ``N``.
        """
        return self.frontend.N

    # -- evaluation -----------------------------------------------------------
    def __call__(self, v: np.ndarray, method: str = "fir") -> np.ndarray:
        """Estimate ``u`` from control signals ``v`` of shape ``(size, M, J)``.

        ``"fir"`` convolves with the truncated taps (one FFT), ``"state-space"``
        runs the two recursions; they agree to machine precision once the taps
        have decayed. A filter built for Monte Carlo frontends has one recursion
        per column and no FIR.

        Parameters
        ----------
        v : ndarray, shape (size, M, J)
            Clocked comparator decisions, one sequence per batch column.
        method : {"fir", "state-space"}, optional
            FIR convolution or forward and backward state recursions.
            Defaults to ``"fir"``.

        Returns
        -------
        ndarray of float, shape (size, L, J)
            Estimated input sequence.
        """
        if method == "fir":
            if self.J_m > 1:
                raise NotImplementedError(
                    "one frontend per column: use method='state-space'"
                )
            h = self.taps()
            c = (h.shape[0] - 1) // 2
            out = np.zeros((v.shape[0], self.L, v.shape[2]))
            for m in range(self.M):
                for ell in range(self.L):
                    out[:, ell] += fftconvolve(v[:, m], h[:, m, ell][:, None], axes=0)[
                        c : c + v.shape[0]
                    ]
            return out
        if method != "state-space":
            raise ValueError("method must be 'fir' or 'state-space'")
        out = np.zeros((v.shape[0], self.L, v.shape[2]))
        _message(
            self.A_b, self.B_b, self.W, v, out, False
        )  # + W m_b, m_b[i] knows v[i:]
        _message(
            self.A_f, self.B_f, -self.W, v, out, True
        )  # - W m_f, m_f[i] knows v[:i]
        return out

    def taps(self, K: Optional[int] = None, tol: float = 1e-12) -> np.ndarray:
        """FIR taps ``(K, M, L)`` about lag zero; ``K`` grows until the edge taps fall below ``tol``.

        Lag ``+j`` (future) is ``W A_b^j B_b`` at index ``c - j`` and lag
        ``-1-j`` (past) is ``-W A_f^j B_f`` at index ``c + 1 + j``, ``c = (K-1)//2``.

        Parameters
        ----------
        K : int, optional
            Number of taps, at least two; chosen automatically when omitted.
            Defaults to ``None``.
        tol : float, optional
            Relative endpoint energy threshold for automatic tap length.
            Defaults to ``1e-12``.

        Returns
        -------
        ndarray of float, shape (K, M, L)
            Centered FIR impulse-response coefficients.
        """
        if self.J_m > 1:
            raise NotImplementedError(
                "one frontend per column: use method='state-space'"
            )
        if K is None:
            if self._taps is not None:
                return self._taps
            K = 1 << 6
            while K < 1 << 16:
                h = self.taps(K)
                energy = np.linalg.norm(h.reshape(K, -1), axis=1)
                if (
                    energy[K // 2] <= 0.0
                    or max(energy[0], energy[-1]) / energy[K // 2] < tol
                ):
                    break
                K <<= 1
            else:
                logger.warning(
                    "Wiener FIR taps have not decayed to %.0e by K=%d", tol, K
                )
            self._taps = h
            return h
        if K < 2:
            raise ValueError("K must be at least 2")
        taps = np.zeros((K, self.M, self.L))
        centre = (K - 1) // 2
        column = self.B_b
        for j in range(centre + 1):
            taps[centre - j] = (self.W @ column).T
            column = self.A_b @ column
        column = self.B_f
        for j in range(K - centre - 1):
            taps[centre + 1 + j] = (-self.W @ column).T
            column = self.A_f @ column
        return taps

    # -- frequency domain -----------------------------------------------------
    def G(self, jw: np.ndarray) -> np.ndarray:
        """Open-loop state response ``(jw I - A)^-1 B``, shape ``(size, N, L + M)``.

        Parameters
        ----------
        jw : ndarray of complex, shape (size,)
            One-dimensional angular-frequency vector ``1j * omega`` in rad/s.

        Returns
        -------
        ndarray of complex, shape (size, N, L + M)
            Open-loop state response at each angular frequency.
        """
        return self.frontend.transfer_function(jw, state_output=True)[1]

    def ntf(self, jw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Noise transfer function ``G^H (G G^H + eta2 I)^-1``, shape ``(size, L, N)``.

        Parameters
        ----------
        jw : ndarray of complex, shape (size,)
            One-dimensional angular-frequency vector ``1j * omega`` in rad/s.

        Returns
        -------
        jw : ndarray of complex, shape (size,)
            Input angular frequencies.
        H : ndarray of complex, shape (size, L, N)
            Noise transfer matrix at each frequency.
        """
        G = self.G(jw)[:, :, : self.L]
        GH = G.conj().transpose(0, 2, 1)
        x = np.linalg.solve(G @ GH + self.eta2 * np.eye(self.N), G)
        return jw, x.conj().transpose(0, 2, 1)

    def stf(self, jw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signal transfer function ``G^H (G G^H + eta2 I)^-1 G``, shape ``(size, L, L)``.

        Parameters
        ----------
        jw : ndarray of complex, shape (size,)
            One-dimensional angular-frequency vector ``1j * omega`` in rad/s.

        Returns
        -------
        jw : ndarray of complex, shape (size,)
            Input angular frequencies.
        H : ndarray of complex, shape (size, L, L)
            Signal transfer matrix at each frequency.
        """
        G = self.G(jw)[:, :, : self.L]
        GH = G.conj().transpose(0, 2, 1)
        return jw, GH @ np.linalg.solve(G @ GH + self.eta2 * np.eye(self.N), G)


class StateSpaceFilter:
    """A state-space model driven by the decisions: ``m[i] = A[k] m[i-1] + B[k] v[i-1]``, ``out = W[k] m``.

    Matrices follow the frontend's convention, ``(N, N)``, cyclic ``(K, N, N)``
    with ``k = i % K``, or ``(J, K, N, N)`` with one model per column; the
    output is low-pass decimated by ``DSR``. :meth:`from_model` is the causal
    half of the Wiener filter, the forward message of the leaked loop filter
    read out with ``-W``; it settles because of the leak and estimates ``u``
    itself. The digital twin replays the frontend matrices on decisions;
    :meth:`Frontend.simulate` returns its ``x_hat`` and ``y_hat``.

    Parameters
    ----------
    A : ndarray of float, shape (N, N), (K, N, N), or (J, K, N, N)
        State transition matrix.
    B : ndarray of float, shape (N, M), (K, N, M), or (J, K, N, M)
        Decision-to-state matrix.
    W : ndarray of float, shape (L, N), (K, L, N), or (J, K, L, N)
        State readout matrix.
    DSR : int, optional
        Positive output decimation ratio.
        Defaults to ``1``.

    >>> af, OSR = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
    >>> af.dac = DAC(Constant(1.0))  # this estimator uses immediate NRZ
    >>> causal = StateSpaceFilter.from_model(af, OSR=OSR, DSR=4)
    >>> causal(af.simulate(1 << 10, verbose=False)["v"]).shape
    (256, 1, 1)

    Returns
    -------
    StateSpaceFilter
        Initialized causal state-space estimator.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, W: np.ndarray, DSR: int = 1):
        self.A, self.B, self.W, self.DSR = (
            np.asarray(A, dtype=float),
            np.asarray(B, dtype=float),
            np.asarray(W, dtype=float),
            int(DSR),
        )
        if self.A.ndim < 2 or self.A.ndim != self.B.ndim or self.W.ndim < 2:
            raise ValueError("A and B need the same number of axes, at least two")

    @classmethod
    def from_model(
        cls,
        frontend: Frontend,
        eta2: Optional[float] = None,
        OSR: Optional[float] = None,
        DSR: int = 1,
    ) -> "StateSpaceFilter":
        """The causal half of the Wiener filter of ``frontend``.

        Parameters
        ----------
        frontend : Frontend
            Converter to estimate from its decision stream.
        eta2 : float, optional
            Positive regularization variance; required unless ``OSR`` is given.
            Defaults to ``None``.
        OSR : float, optional
            Oversampling ratio used to derive ``eta2`` when omitted.
            Defaults to ``None``.
        DSR : int, optional
            Positive output decimation ratio.
            Defaults to ``1``.

        Returns
        -------
        StateSpaceFilter
            Causal state-space estimator derived from the converter.
        """
        wf = frontend.wiener_filter(eta2=eta2, OSR=OSR)
        return cls(wf.A_f, wf.B_f, -wf.W, DSR)

    @property
    def N(self) -> int:
        """Number of state-space filter states.

        Returns
        -------
        int
            State dimension ``N``.
        """
        return self.A.shape[-1]

    def states(self, v: np.ndarray) -> np.ndarray:
        """``m`` ``(size, N, J)``; ``states[i]`` depends on ``v[:i]`` only.

        Parameters
        ----------
        v : ndarray, shape (size, M, J)
            Clocked comparator decisions.

        Returns
        -------
        ndarray of float, shape (size, N, J)
            Forward message state sequence.
        """
        out = np.zeros((v.shape[0], self.N, v.shape[2]))
        _message(self.A, self.B, np.eye(self.N), v, out, True)
        return out

    def pole_radii(self) -> np.ndarray:
        """``|eig(A)|`` per phase and column; the product over a cycle decides stability.

        Returns
        -------
        ndarray of float, shape (..., N)
            Per-phase and per-frontend pole magnitudes.
        """
        return np.abs(np.linalg.eigvals(self.A))

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """``W m``, low-pass decimated by ``DSR``: ``(size // DSR, L, J)``.

        Parameters
        ----------
        v : ndarray, shape (size, M, J)
            Clocked comparator decisions.

        Returns
        -------
        ndarray of float, shape (size // DSR, L, J)
            Estimated and decimated input sequence.
        """
        from .snr import decimate

        out = np.zeros((v.shape[0], self.W.shape[-2], v.shape[2]))
        _message(self.A, self.B, self.W, v, out, True)
        return decimate(out, self.DSR) if self.DSR > 1 else out
