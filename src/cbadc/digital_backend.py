"""The digital backend.

This module contains classes for digital signal processing and general
post-processing of the analog frontend output.
"""

import logging as _logging
from copy import deepcopy as _deepcopy
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _sliding_window_view
from scipy.integrate import solve_ivp as _solve_ivp
from scipy.linalg import (
    expm as _expm,
)
from scipy.linalg import (
    solve_continuous_are as _care,
)
from scipy.linalg import (
    solve_discrete_are as _dare,
)
from scipy.signal import (
    TransferFunction,
)
from scipy.signal import (
    convolve as _convolve,
)
from scipy.signal import (
    decimate as _decimate,
)
from scipy.signal import (
    firwin2 as _firwin2,
)
from scipy.signal import (
    resample as _resample,
)

from .analog_frontend import AnalogFrontend
from .analog_signal import ZeroOrderHold
from .utilities import show_status as _show_status

logger = _logging.getLogger(__name__)


class WienerFilter:
    """The analytical Wiener filter for control-bounded converters.

    The Wiener filter is a linear filter that minimizes the mean square error
    between the output of the analog frontend and the output of the filter.

    Parameters
    ----------
    analog_frontend : :py:class:`cbadc.AnalogFrontend`
        The analog frontend.
    eta2 : `np.double`
        The noise variance and bandwidth term.

    """

    def __init__(self, analog_frontend: AnalogFrontend, eta2: np.double):
        self._analog_frontend = analog_frontend
        # The eta2 setter computes the Wiener filter
        self.eta2 = eta2

    @property
    def analog_frontend(self) -> AnalogFrontend:
        """Return the analog frontend.

        Returns
        -------
        analog_frontend : :py:class:`cbadc.AnalogFrontend`
            The analog frontend.
        """
        return self._analog_frontend

    @property
    def eta2(self) -> np.double:
        """Return the noise variance.

        Returns
        -------
        eta2 : `np.double`
            The noise variance.
        """
        return self._eta2

    @property
    def L(self) -> int:
        """Return the number of references.

        Returns
        -------
        L : `int`
            The number of references.
        """
        return self._analog_frontend.L

    @property
    def M(self) -> int:
        """Return the number of control signals.

        Returns
        -------
        M : `int`
            The number of control signals.
        """
        return self._analog_frontend.M

    @property
    def N(self) -> int:
        """Return the number of states.

        Returns
        -------
        N : `int`
            The number of states.
        """
        return self._analog_frontend.N

    def _V_X_f_uggly(self, eta2, iterations: int = 1000) -> np.ndarray:
        V_X_f = np.eye(self.N, dtype=np.double) * 1e30
        for _ in range(iterations):
            V_x_prime = (
                self.analog_frontend.A[0] @ V_X_f @ self.analog_frontend.A[0].T
                + self.analog_frontend.B[0, :, : self.L]
                @ self.analog_frontend.B[0, :, : self.L].T
            )
            G_k_inv = (
                np.eye(self.N, dtype=np.double) / eta2
                + self.analog_frontend.C[0, : self.N, :]
                @ V_x_prime
                @ self.analog_frontend.C[0, : self.N, :].T
            )
            G_dd_inv = self.analog_frontend.C[0, : self.N, :].T @ np.linalg.solve(
                G_k_inv, self.analog_frontend.C[0, : self.N, :].T
            )
            V_x_f = V_x_prime - V_x_prime @ np.linalg.solve(G_dd_inv, V_x_prime)
            return V_x_f

    @eta2.setter
    def eta2(self, eta2: np.double):
        """Set the noise variance and bandwidth term.

        Parameters
        ----------
        eta2 : `float`
            The noise variance ratio between input and
            output noise.
        """
        if not isinstance(eta2, float) or eta2 <= 0:
            raise ValueError("eta2 must be a positive float")

        self._eta2 = eta2

        L = self._analog_frontend.L
        N = self._analog_frontend.N
        M = self._analog_frontend.M

        if self._analog_frontend.is_discrete_time:
            raise NotImplementedError(
                "Wiener filter for discrete-time analog frontends is not implemented yet."
            )
            # Compute modified Bryson-Frazier smoother
            A_dare: np.ndarray = self._analog_frontend.A[0].T.conjugate()
            # B_dare = np.eye(N, dtype=np.double)
            # Q_dare = (
            #     self._analog_frontend.B[0, :, :L] @ self._analog_frontend.B[0, :, :L].T
            # )
            # R_dare = self._eta2 * np.eye(N, dtype=np.double)
            # V_X_f = _dare(A_dare, B_dare, Q_dare, R_dare)
            V_X_f = self._V_X_f_uggly(self._eta2)
            G = np.linalg.inv(R_dare + V_X_f)
            F = np.eye(N, dtype=np.double) - V_X_f @ G
            self._Af = self._analog_frontend.A[0] @ F
            self._Bf = self._analog_frontend.B[0, :, L:]
            self._Ab = F.transpose() @ self._analog_frontend.A[0].transpose()
            self._Bb = -G
            self._W = -self._analog_frontend.B[0, :, :L].transpose()
        else:
            # Compute the Wiener filter
            # Algebraic Riccati equation Notation
            A_care: np.ndarray = self._analog_frontend.A[0].T
            B_care = np.eye(N, dtype=np.double)
            # Q = B B^T
            Q_care = (
                self._analog_frontend.B[0, :, :L] @ self._analog_frontend.B[0, :, :L].T
            )
            R_care = self._eta2 * np.eye(N, dtype=np.double)

            # Compute stationary covariance matrices
            V_f = _care(A_care, B_care, Q_care, R_care)
            V_b = _care(-A_care, B_care, Q_care, R_care)

            self._W = np.linalg.solve(V_f + V_b, self._analog_frontend.B[0, :, :L]).T

            dt = self._analog_frontend.dt

            if self._analog_frontend.digital_control.dac_waveform == "nrz":
                tmp_arg = np.vstack(
                    (
                        np.hstack(
                            (
                                self._analog_frontend.A[0] - V_f / self._eta2,
                                self._analog_frontend.B[0, :, L:],
                            )
                        ),
                        np.zeros((M, N + M), dtype=np.double),
                    )
                )
                tmp = _expm(tmp_arg * dt)
                self._Af = tmp[:N, :N]
                self._Bf = tmp[:N, N:]

                tmp_arg = np.vstack(
                    (
                        np.hstack(
                            (
                                -self._analog_frontend.A[0] - V_b / self._eta2,
                                -self._analog_frontend.B[0, :, L:],
                            )
                        ),
                        np.zeros((M, N + M), dtype=np.double),
                    )
                )
                tmp = _expm(tmp_arg * dt)
                self._Ab = tmp[:N, :N]
                self._Bb = tmp[:N, N:]
            else:
                tmp_Af = self._analog_frontend.A[0] - V_f / self._eta2
                tmp_Ab = -self._analog_frontend.A[0] - V_b / self._eta2
                self._A_f = _expm(tmp_Af * dt)
                self._A_b = _expm(tmp_Ab * dt)

                def der_f(t: np.double, x: np.ndarray):
                    return tmp_Af @ x + self._analog_frontend.B[
                        0, :, L:
                    ] * self._analog_frontend.digital_control.impulse_response(
                        np.array([t])
                    ).reshape((1, -1))

                res = _solve_ivp(der_f, (0, dt), np.zeros(N * M, dtype=np.double))
                self._Bf = res.y[:, -1].reshape((N, M))

                def der_b(t: np.double, x: np.ndarray):
                    return tmp_Ab @ x - self._analog_frontend.B[
                        0, :, L:
                    ] * self._analog_frontend.digital_control.impulse_response(
                        np.array([t])
                    ).reshape((1, -1))

                res = _solve_ivp(der_b, (0, dt), np.zeros(N * M, dtype=np.double))
                self._Bb = res.y[:, -1].reshape((N, M))

    def G(self, jw: np.ndarray) -> np.ndarray:
        """Compute the open loop transfer function.

        Parameters
        ----------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        Returns
        -------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        G : :py:class:`numpy.ndarray`, shape=(size, L, N)
            the open loop transfer function.

        """
        _, h = self._analog_frontend.transfer_function(
            jw, state_output=True, open_loop=True
        )
        return h

    def evaluate(self, v: np.ndarray) -> np.ndarray:
        """Evaluate the Wiener filter.

        Parameters
        ----------
        v : :py:class:`numpy.ndarray`, shape=(size, M, J)
            input control signals.

        Returns
        -------
        y : :py:class:`numpy.ndarray`, shape=(size, L, J)
            output estimates signal.

        """
        M = self._analog_frontend.M
        if v.shape[1] != M:
            raise ValueError(
                "s must have the shape = (size, M), where M is the number of control signals"
            )

        # allocate memory
        size = v.shape[0]
        N = self._analog_frontend.N
        L = self._analog_frontend.L
        J = self._analog_frontend.J
        m_v = np.zeros((size + 2, N, J), dtype=np.double)
        u_hat = np.zeros((size, L, J), dtype=np.double)

        # Forward message passing
        for i in range(size):
            m_v[i + 1] = self._Af @ m_v[i] + self._Bf @ v[i]
        # Backward message passing
        if self._analog_frontend.is_discrete_time:
            for i in range(size - 1, -1, -1):
                m_v[i] = self._Ab @ m_v[i + 1] + self._Bb @ m_v[i]
                u_hat[i] = -self._W @ m_v[i]
        else:
            for i in range(size - 1, -1, -1):
                m_v[i + 1] = self._Ab @ m_v[i + 2] + self._Bb @ v[i]
                u_hat[i] = self._W @ (m_v[i + 1] - m_v[i])
        return u_hat

    def ntf(self, jw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the noise transfer function.

        NTF(jw) = G(jw)^H (G(jw) G(jw)^H + eta^2 I)^{-1}

        where G(jw) is the open loop transfer function of the
        analog frontend.

        Parameters
        ----------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        Returns
        -------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        x_H : :py:class:`numpy.ndarray`, shape=(size, L, N)
            the noise transfer function.

        """
        # G^H (G G^H + eta^2 I)^{-1}
        # G^H = x (G G^H + eta^2 I)
        # G = (GGH + eta^2 I)^H x^H
        # G = (GGH + eta^2 I) x^H
        # shape = (size, N, L+M)
        G = self.G(jw)
        # shape = (size, N, L)
        G = G[:, :, : self._analog_frontend.L]
        # shape = (size, L, N)
        GH = G.conj().transpose(0, 2, 1)
        # shape = (size, N, N)
        GGH = G @ GH
        N = self._analog_frontend.N
        # shape = (size, N, L)
        x_H = np.linalg.solve(
            GGH + self.eta2 * np.eye(N, dtype=complex)[np.newaxis, :, :], G
        )
        return jw, x_H.conj().transpose(0, 2, 1)

    def stf(self, jw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the signal transfer function.

        STF(jw) = NTF(jw) G(jw)
                = G(jw)^H (G(jw) G(jw)^H + eta^2 I)^{-1} G(jw)

        where G(jw) is the open loop transfer function of the
        analog frontend.

        Parameters
        ----------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        Returns
        -------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        y_H : :py:class:`numpy.ndarray`, shape=(size, L, L)
            the signal transfer function.

        """
        # G^H (G G^H + eta^2 I)^{-1} G
        # (G G^H + eta^2 I) x = G
        # y = G^H x
        # shape = (size, N, L+M)
        G = self.G(jw)
        # shape = (size, N, L)
        G = G[:, :, : self._analog_frontend.L]
        # shape = (size, L, N)
        GH = G.conj().transpose(0, 2, 1)
        # shape = (size, N, N)
        GGH = G @ GH
        N = self._analog_frontend.N
        # shape = (size, N, L)
        x = np.linalg.solve(
            GGH + self.eta2 * np.eye(N, dtype=complex)[np.newaxis, :, :], G
        )
        return jw, GH @ x

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """Evaluate the Wiener filter.

        Parameters
        ----------
        v : :py:class:`numpy.ndarray`, shape=(size, M)
            input control signals.

        Returns
        -------
        y : :py:class:`numpy.ndarray`, shape=(size, L)
            output estimates signal.

        """
        return self.evaluate(v)

    # def to_adaptive_FIRFilter(
    #     self,
    #     K: int,
    #     dtype: np.dtype = np.double,
    #     seed: int = 34981723498712594372,
    # ):
    #     """Convert the Wiener filter to an adaptive FIR filter.

    #     Returns
    #     -------
    #     adaptive_fir_filter : :py:class:`cbadc.AdaptiveFIRFilter`
    #         The adaptive FIR filter.
    #     """

    #     fir_filter = AdaptiveFIRFilter(M=self.M, K=K, L=self.L, dtype=dtype, dt=self.dt)
    #     # fir_filter._h = np.zeros((L, M, K), dtype=dtype)
    #     mid_K = K // 2
    #     AbBb =self._Bb[:, :]
    #     for k in range(mid_K, K):
    #         fir_filter._h[:, :, k] = np.dot(self._W, AbBb).astype(dtype)
    #         AbBb = np.dot(self._Ab, AbBb)
    #     AfBf =

    #     return fir_filter


class AdaptiveFIRFilter:
    """
    Adaptive FIR filter model.

    Capable of being calibrated against a reference using the LMS algorithm.

    Parameters
    ----------
    M : int
        The number of control signals (note this excludes any references).
    K : int
        The number of filter taps per control signal.
    L : int
        The number of references.
    dtype : np.dtype
        The data type of the filter coefficients.

    Attributes
    ----------
    K : int
        The number of filter taps per control signal.
    L : int
        The number of references.
    M : int
        The number of control signals (note this excludes any references).

    """

    def __init__(
        self,
        M: int,
        K: int,
        L: int = 1,
        h0: np.ndarray = None,
        dtype: np.dtype = np.double,
        seed: int = 34981723498712594372,
        dt: np.double = 1.0,
        analog_frontend: Optional[AnalogFrontend] = None,
    ):
        self._K = K
        self._L = L
        self._M = M
        self._h = np.zeros((K, M, L), dtype=dtype)
        if h0 is not None:
            self.h0 = h0
        else:
            h0 = np.zeros((K, L), dtype=dtype)
            # simple delta function with effective delay of K//2
            h0[K // 2 - 1, :] = 1.0
            self._h0 = h0
        self.dt = dt
        self._offset = np.zeros((L), dtype=dtype)
        self._rng = np.random.default_rng(seed)
        if analog_frontend is not None:
            if analog_frontend.L != L:
                raise ValueError(
                    "L must be equal to the number of references in the analog frontend"
                )
            if analog_frontend.M != M:
                raise ValueError(
                    "M must be equal to the number of control signals in the analog frontend"
                )
        self._analog_frontend = analog_frontend

    @property
    def h0(self) -> np.ndarray:
        """The initial filter coefficients.

        Returns
        -------
        h0 : :py:class:`numpy.ndarray`, shape=(L, K)
            The initial filter coefficients.
        """
        return self._h0.transpose((1, 0))

    @h0.setter
    def h0(self, h0: np.ndarray):
        if not isinstance(h0, np.ndarray):
            raise ValueError("h0 must be a numpy array")
        if h0.shape != (self.K, self.L):
            raise ValueError("h0 must have the shape (K, L)")
        if h0.dtype != self.dtype:
            h0 = h0.astype(self.dtype)
        self._h0 = h0

    @property
    def dtype(self) -> np.dtype:
        """The data type of the filter coefficients.

        See :py:class:`numpy.dtype` for more information.
        """
        return self._h.dtype

    @dtype.setter
    def dtype(self, dtype):
        self._h = self._h.astype(dtype)
        self._offset = self._offset.astype(dtype)

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, K):
        if not isinstance(K, int) or K <= 0:
            raise ValueError("K must be a positive integer")
        if self._h.shape[0] != K:
            self._h = np.zeros((K, self.M, self.L), dtype=self.dtype)
            self._offset = np.zeros((self.L), dtype=self.dtype)
            logger.warning(
                "Filter taps and offset have been set to zero due to K change"
            )
        self._K = K

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, L):
        if not isinstance(L, int) or L <= 0:
            raise ValueError("L must be a positive integer")
        if self._h.shape[2] != L:
            self._h = np.zeros((self.K, self.M, L), dtype=self.dtype)
            self._offset = np.zeros((L), dtype=self.dtype)
            logger.warning(
                "Filter taps and offset have been set to zero due to L change"
            )
        self._L = L

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, M):
        if not isinstance(M, int) or M <= 0:
            raise ValueError("M must be a positive integer")
        if self._h.shape[1] != M:
            self._h = np.zeros((self.K, M, self.L), dtype=self.dtype)
            self._offset = np.zeros((self.L), dtype=self.dtype)
            logger.warning(
                "Filter taps and offset have been set to zero due to M change"
            )
        self._M = M

    @property
    def h(self) -> list[list[TransferFunction]]:
        tfs = []
        a = np.zeros(self.K, dtype=self.dtype)
        a[-1] = 1
        for l in range(self.L):
            tmp_tfs = []
            for m in range(self.M):
                tmp_tfs.append(TransferFunction(self._h[::-1, m, l], a, dt=self.dt))
            tfs.append(tmp_tfs)
        return tfs

    @h.setter
    def h(self, h):
        if not isinstance(h, np.ndarray):
            raise ValueError("h must be a numpy array")
        if h.shape != self._h.shape:
            raise ValueError("h must have the shape (K, M, L)")
        if h.dtype != self.dtype:
            self.dtype = h.dtype
        self._h = h

    @property
    def offset(self):
        return self._offset[:]

    @offset.setter
    def offset(self, offset):
        if not isinstance(offset, np.ndarray):
            raise ValueError("offset must be a numpy array")
        if offset.shape != self._offset.shape:
            raise ValueError("offset must have the shape (L,)")
        if offset.dtype != self.dtype:
            self.dtype = offset.dtype
        self._offset

    def convolve(self, x: np.ndarray, method: str = "direct", DSR: int = 1):
        """Convolve filter taps with input data.

        Parameters
        ----------
        x : :py:class:`numpy.ndarray`, shape=(size, M, J)
            The input data, shape (nr_samples, M, J).
        method : str
            The convolution method to use, defaults to "auto". Options are:
            - "auto" lets scipy decide the best method.
            - "direct" uses direct convolution, typically more efficient for small arrays.
            - "fft" uses the FFT to compute the convolution, typically
            more efficient for sizeable arrays.
        DSR : int
            Apply decimation by this factor before filtering.

        Returns
        -------
        y : :py:class:`numpy.ndarray`, shape=(size, L, J)
            The output data, shape (nr_samples, nr_references, J).
        """
        # Decimate input signal if required
        if DSR > 1:
            if method == "fft":
                x = decimate(x, DSR, axis=0, method="fft")
            else:
                x = decimate(x, DSR, axis=0, method="direct")

        size = x.shape[0]
        if x.shape[1] != self.M:
            raise ValueError(
                f"x must have the shape = (size, M, J), where M is the number of control signals\n: got {x.shape[1]} instead of {self.M}"
            )
        M = self.M
        J = x.shape[2]

        if method in ["fft", "direct", "auto"]:
            # x.shape = (size, M, J)
            # self._h.shape = (K, M, L)
            # x_reshaped = x.transpose(2, 1, 0)[:, np.newaxis, :, :]
            # x_reshaped = x[:, :, np.newaxis, :]
            # h_reshaped = self._h[:, :, :, np.newaxis]
            # x_reshaped.shape = (size, M, 1, J)
            # h_reshaped.shape = (K, M, L, 1)

            # return (
            #     np.sum(
            #         # convolve results in shape (size, M, L, J)
            #         _convolve(
            #             x_reshaped,
            #             h_reshaped,
            #             mode="same",
            #             method=method,
            #         ),
            #         # we collapse the M dimension with a sum after convolution
            #         # resulting in shape (size, L, J)
            #         axis=1,
            #     )
            #     + self._offset[np.newaxis, :, np.newaxis]
            # )
            # # flesh it out

            y = np.zeros((size, self.L, J), dtype=self.dtype)
            for j in range(J):
                for l in range(self.L):
                    for m in range(self.M):
                        y[:, l, j] += _convolve(
                            x[:, m, j],
                            self._h[:, m, l],
                            mode="same",
                            method=method,
                        )
            y += self._offset[np.newaxis, :, np.newaxis]
            return y

        else:
            raise ValueError(
                f"method must be either 'fft', 'direct', 'auto', or 'conv' not {method}"
            )

    def evaluate(self, v: np.ndarray, DSR: int = 1) -> np.ndarray:
        """Evaluate the Wiener filter.

        Parameters
        ----------
        v : :py:class:`numpy.ndarray`, shape=(size, M, J)
            input control signals.
        DSR : int
            Apply decimation by this factor before filtering.

        Returns
        -------
        y : :py:class:`numpy.ndarray`, shape=(size, L, J)
            output estimates signal.

        """
        return self.convolve(v, DSR=DSR)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """Evaluate the Wiener filter.

        Parameters
        ----------
        v : :py:class:`numpy.ndarray`, shape=(size, M, J)
            input control signals.

        Returns
        -------
        y : :py:class:`numpy.ndarray`, shape=(size, L, J)
            output estimates signal.

        """
        return self.evaluate(v)

    def _sliding_window_tensor_dot_convolution(self, x: np.ndarray):
        """
        Parameters
        ----------

        x : np.ndarray (size, M, K)
            The input data.

        Returns
        -------
        y : np.ndarray (size, L)
            The output data.
        """
        # x: (size, M, K), _h: (K, M, L)
        # contract over M (x axis 1 with h axis 1) and K (x axis 2 with h axis 0)
        return (
            np.tensordot(
                x,
                self._h,
                axes=([1, 2], [1, 0]),
            )
            + self._offset[np.newaxis, :]
        )

    def loss(self, x: np.ndarray, y: np.ndarray, method="direct") -> np.ndarray:
        """Computes the loss, i.e., the squared L2 norm, for the given FIR filter.

        Returns

        ||y - h * x||^2 / size



        Parameters
        ----------
        x : :py:class:`numpy.ndarray`, shape=(size, M, J)
            The input data.
        y: :py:class:`numpy.ndarray`, shape=(size, L, J)
            The reference data.
        method : str
            The convolution method to use, defaults to "auto". Options are:
            - "auto" lets scipy decide the best method.
            - "direct" uses direct convolution, typically more efficient for small arrays.
            - "fft" uses the FFT to compute the convolution, typically
            more efficient for sizeable arrays.

        Returns
        -------
        loss : :py:class:`numpy.ndarray`, shape=(L, J)
            The loss function evaluated on the given data.
        """
        # y_hat.shape = (size, L, J)
        y_hat = self.convolve(x, method=method)
        y_0 = self.convolve_ref(y, method=method)
        size = np.minimum(y_hat.shape[0], y_0.shape[0])
        return np.linalg.norm(y_0[:size, :, :] - y_hat[:size, :, :], axis=0) ** 2 / size

    def gradient(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the gradient of the loss function with respect to the filter
        coefficients.

        Parameters
        ----------
        x : np.ndarray (batch_size, M, K)
            The input data.
        y: np.ndarray (batch_size, L)
            The output data.

        Returns
        -------
        gradient : [np.ndarray (K, M, L), np.ndarray (L,)]
            The gradient of the loss function with respect to the filter
            coefficients.
        """
        batch_size = x.shape[0]
        error = y - self._sliding_window_tensor_dot_convolution(x)
        # tensordot(error, x, ([0],[0])): (L, M, K) -> transpose to (K, M, L)
        grad_h = -np.tensordot(error, x, axes=([0], [0])) / batch_size
        return grad_h.transpose(2, 1, 0), -error.mean(axis=0)

    def lms(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        epochs: int,
        learning_rate: np.double = 1e-5,
        momentum: np.double = 0.9,
        shuffle=False,
        verbose=True,
    ):
        """
        Fits the filter to the given data using the LMS method.

        Parameters
        ----------
        x : np.ndarray (size, M)
            The input data.
        y: np.ndarray (size, L)
            The reference data.
        batch_size : int
            The batch size.
        epochs : int
            The number of epochs.
        learning_rate : np.double
            The learning rate, defaults to 1e-5.
        momentum : np.double
            The momentum, defaults to 0.9.
        shuffle : bool
            Whether to shuffle the data, defaults to False.
        verbose : bool
            Whether to print the loss function during training.

        Returns
        -------
        loss : np.ndarray (L,)
            The loss function evaluated on the given data.
        """

        if verbose:
            logger.info(
                (
                    "Training using LMS for:",
                    f"- epochs = {epochs},",
                    f"- batch size = {batch_size},",
                    f"- learning rate = {learning_rate},",
                    f"- momentum = {momentum},",
                    f"- and shuffle set to {shuffle}.",
                )
            )
        # (size - K + 1, M, K)
        x_window = _sliding_window_view(x, self.K, axis=0)
        new_size = np.minimum(x_window.shape[0], y.shape[0])
        # shape = (size - K + 1, L)
        x_window = x_window[:new_size, :, :]
        y_window = y[:new_size, :]

        # Lazy initialize momentum
        if hasattr(self, "_h_m") is False:
            self._h_m = np.zeros_like(self._h)
            self._offset_m = np.zeros_like(self._offset)

        for e in _show_status(range(epochs)):
            if shuffle:
                permuations = self._rng.permutation(new_size)
                x_window = x_window[permuations, :, :]
                y_window = y_window[permuations, :]

            for b in range(0, new_size, batch_size):
                x_batch = x_window[b : b + batch_size, :, :]
                y_batch = y_window[b : b + batch_size, :]

                # ((L, M, K), (L,))
                g_h, g_o = self.gradient(x_batch, y_batch)

                # (L, M, K)
                self._h_m *= momentum
                self._h_m += learning_rate * g_h
                self._h -= self._h_m

                # (L,)
                self._offset_m *= momentum
                self._offset_m += learning_rate * g_o
                self._offset -= self._offset_m

            if verbose:
                x3 = x if x.ndim == 3 else x[:, :, np.newaxis]
                y3 = y if y.ndim == 3 else y[:, :, np.newaxis]
                logger.info(
                    "epoch %d: loss = %s, offset = %s",
                    e,
                    self.loss(x3, y3),
                    self._offset,
                )
        x3 = x if x.ndim == 3 else x[:, :, np.newaxis]
        y3 = y if y.ndim == 3 else y[:, :, np.newaxis]
        return self.loss(x3, y3)

    def rls(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        delta: np.double = 1e-2,
        lambda_: np.double = 1e0 - 1e-12,
        shuffle=False,
        verbose=True,
    ):
        """
        Fits the filter to the given data using the RLS method.

        Parameters
        ----------
        x : np.ndarray (size, M, K)
            The input data.
        y: np.ndarray (size, L)
            The reference data.
        epochs : int
            The number of epochs.
        delta : np.double
            The delta parameter of the RLS algorithm.
        lambda : np.double
            The lambda parameter of the RLS algorithm.
        shuffle : bool
            Whether to shuffle the data, defaults to False.
        verbose : bool
            Whether to print the loss function during training.


        Returns
        -------
        loss : np.ndarray (L,)
            The loss function evaluated on the given data.
        """
        if verbose:
            logger.info(
                "Training using RLS for: "
                f"- epochs = {epochs}, "
                f"- delta = {delta}, "
                f"- lambda = {lambda_}, "
                f"- and shuffle set to {shuffle}."
            )

        # (size - K + 1, M, K)
        x_window = _sliding_window_view(x, self.K, axis=0)
        new_size = np.minimum(x_window.shape[0], y.shape[0])
        # shape = (size - K + 1, L)
        x_window = x_window[:new_size, :, :]
        y_window = y[:new_size, :]

        # Lazy initialize Covariance matrix
        if hasattr(self, "_V") is False:
            total_size: int = self.M * self.K + 1
            self._V = np.eye(total_size, dtype=self.dtype) / delta
            self._x_flatened = np.zeros((total_size), dtype=self.dtype)
            self._x_flatened[-1] = 1.0
        else:
            logger.warning(
                "RLS algorithm already initialized, delta=%s has no effect.", delta
            )

        for e in _show_status(range(epochs)):
            if shuffle:
                permutation = self._rng.permutation(new_size)
                x_window = x_window[permutation, :, :]
                y_window = y_window[permutation, :]

            for b in range(new_size):
                x_batch = x_window[b : b + 1, :, :]
                self._x_flatened[:-1] = x_batch.flatten()
                y_batch = y_window[b : b + 1, :]

                # shape(1, L)
                error = y_batch - self._sliding_window_tensor_dot_convolution(x_batch)

                alpha = np.dot(self._V, self._x_flatened)
                g = alpha / (lambda_ + np.dot(self._x_flatened.conj(), alpha))

                self._V = (self._V - np.outer(g, alpha.conj())) / lambda_

                self._offset += g[-1] * error.flatten()
                # g[:-1]: (K*M,), error[0]: (L,) -> outer product (K, M, L)
                self._h += (
                    g[:-1].reshape((self.K, self.M, 1))
                    * error[0, np.newaxis, np.newaxis, :]
                )

            if verbose:
                x3 = x if x.ndim == 3 else x[:, :, np.newaxis]
                y3 = y if y.ndim == 3 else y[:, :, np.newaxis]
                logger.info(
                    "epoch %d: loss = %s, offset = %s",
                    e,
                    self.loss(x3, y3),
                    self._offset,
                )
        x3 = x if x.ndim == 3 else x[:, :, np.newaxis]
        y3 = y if y.ndim == 3 else y[:, :, np.newaxis]
        return self.loss(x3, y3)

    def convolve_ref(self, y: np.ndarray, method: str = "direct"):
        # y.shape = (size, L, J)
        size = y.shape[0]
        if y.shape[1] != self.L:
            raise ValueError(
                f"y must have the shape = (size, L, J), where L is the number of references\n: got {y.shape[1]} instead of {self.L}"
            )
        L = self.L
        J = y.shape[2]
        # self._h0.shape = (L, K)
        if method in ["fft", "direct", "auto"]:
            # y_reshaped = y
            # h0_reshaped = self._h0.transpose((1, 0))[:, :, np.newaxis]
            # # y_reshaped.shape = (size, L, J)
            # # h0_reshaped.shape = (K, L, 1)
            # return _convolve(
            #     y_reshaped,
            #     h0_reshaped,
            #     mode="same",
            #     method=method,
            # )[self.K // 2:, :, :]
            # # flesh it out
            y_out = np.zeros((size - self.K + 1, L, J), dtype=self.dtype)
            # y_out = np.zeros((size - self.K // 2, L, J), dtype=self.dtype)
            for j in range(J):
                for l in range(L):
                    y_out[:, l, j] = _convolve(
                        y[:, l, j],
                        self._h0[:, l],
                        mode="valid",
                        method=method,
                    )  # [self.K // 2 :]
            return y_out

    def lstsq(
        self, x: np.ndarray, y: np.ndarray, verbose=True, rcond=None, method="direct"
    ):
        """
        Fits the filter to the given data using the least squares method.

        Parameters
        ----------
        x : np.ndarray (size, M, J)
            The input data.
        y: np.ndarray (size, L, J)
            The reference data.
        verbose : bool
            Whether to print the loss function during training.
        rcond : np.double
            The reciprocal condition number for the least squares method.

        Returns
        -------
        loss : np.ndarray (L,)
            The loss function evaluated on the given data.
        """
        size = x.shape[0]
        M = self.M
        L = self.L
        J = x.shape[2]
        if x.shape[1] != M:
            raise ValueError(f"x must have M={M} not {x.shape[1]}")
        if y.shape[1] != L:
            raise ValueError(f"y must have L={L} not {y.shape[1]}")
        if y.shape[2] != J:
            raise ValueError(f"y must have J={J} not {y.shape[2]}")

        # x.shape = (size, M, J)
        # x_window.shape = (size - K + 1, M, J, K)
        x_window = _sliding_window_view(x, self.K, axis=0)
        # y_window.shape = (size, L, J)
        y_window = self.convolve_ref(y, method=method)
        # y_window = y[self.K // 2 :, :, :]

        batch_size = np.minimum(x_window.shape[0], y_window.shape[0])
        # truncate to new_size
        x_window = x_window[:batch_size, :, :, :]
        y_window = y_window[:batch_size, :, :]

        # Make A x = y problem by grouping the M and K dimensions
        # A_lstsq.shape = (batch_size * J, M * K + 1)
        # A_lstsq = np.hstack(
        #     (
        #         # (batch_size, M, J, K) -> (batch_size, J, K, M) -> (batch_size * J, K * M)
        #         x_window.transpose((0, 2, 3, 1)).reshape((batch_size * J, self.K * M)),
        #         np.ones((batch_size * J, 1), dtype=np.double),
        #     )
        # )
        # # y_lstsq.shape = (batch_size * J, L)
        # y_lstsq = y_window.transpose((0, 2, 1)).reshape((batch_size * J, L))

        extended_batch_size = batch_size * J
        A_lstsq = np.empty((extended_batch_size, self.K * M + 1))
        A_lstsq[:, :-1] = x_window.transpose(0, 2, 3, 1).reshape(
            extended_batch_size, -1
        )
        A_lstsq[:, -1] = 1.0
        y_lstsq = y_window.transpose(0, 2, 1).reshape(extended_batch_size, -1)

        sol = np.linalg.lstsq(A_lstsq, y_lstsq, rcond=rcond)
        self._h[:], self._offset[:] = (
            sol[0][:-1].reshape((self.K, self.M, self.L))[::-1],
            sol[0][-1],
        )
        if verbose:
            loss = sol[1] / batch_size
            logger.info(
                "loss = %s, loss = %s dB, and offset = %s",
                loss,
                10 * np.log10(loss),
                self._offset,
            )
        return self.loss(x, y)

    def transfer_function(self, jw: np.ndarray):
        """
        Returns the transfer function of the filter.

        Parameters
        ----------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            The angular frequency vector.

        Returns
        -------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            The angular frequency vector.
        tf : :py:class:`numpy.ndarray`, shape=(size, L, M), dtype=complex
            The transfer functions.
        """
        tf = np.zeros((jw.size, self.L, self.M), dtype=np.complex128)
        h = self.h
        for l in range(self.L):
            for m in range(self.M):
                _, tf[:, l, m] = h[l][m].freqresp(np.abs(jw))
        return jw, tf

    def plot_amplitude_response(
        self, jw: np.ndarray = np.geomspace(1e-4, 0.5, 1000) * 2j * np.pi, ax=None
    ):
        if ax is None:
            _, ax = plt.subplots(2)

        jw, tf = self.transfer_function(jw)
        for l in range(self.L):
            for m in range(self.M):
                ax[0].plot(
                    np.abs(jw) / (2 * np.pi),
                    20 * np.log10(np.abs(tf[:, l, m])),
                    label=f"m={m}->l={l}",
                )
                ax[1].semilogx(
                    np.abs(jw) / (2 * np.pi),
                    20 * np.log10(np.abs(tf[:, l, m])),
                    label=f"m={m}->l={l}",
                )
        ax[1].set_xlabel("Frequency [Hz]")
        ax[0].set_ylabel("Magnitude [dB]")
        ax[1].set_ylabel("Magnitude [dB]")
        ax[0].legend()
        ax[0].grid()
        ax[1].grid()

    def G(self, jw: np.ndarray) -> np.ndarray:
        """Compute the open loop transfer function.

        Parameters
        ----------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        Returns
        -------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        G : :py:class:`numpy.ndarray`, shape=(size, L, M)
            the open loop transfer function.

        """
        if self._analog_frontend is None:
            raise ValueError(
                "Analog frontend is not set. Can't compute the open loop transfer function."
            )
        _, h = self._analog_frontend.transfer_function(
            jw, state_output=False, open_loop=True
        )
        return h

    def stf(self, jw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the signal transfer function.

        Parameters
        ----------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        Returns
        -------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        H : :py:class:`numpy.ndarray`, shape=(size, L, M)
            the signal transfer function.

        """
        # shape=(size, L, M)
        G = self.G(jw)[1]
        # shape=(size, L, M)
        H = self.transfer_function(jw)[1]
        return jw, np.sum(H * G, axis=2)

    def ntf(self, jw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the noise transfer function.

        Parameters
        ----------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        Returns
        -------
        jw : :py:class:`numpy.ndarray`, shape=(size), dtype=complex
            the angular frequency vector.

        H : :py:class:`numpy.ndarray`, shape=(size, L, M)
            the noise transfer function.

        """
        # shape=(size, L, M)
        return self.transfer_function(jw)

    def plot_impulse_response(self, ax=None):
        """
        Plot the impulse response of the filter.

        Parameters
        ----------
        ax : :py:class:`matplotlib
        """
        if ax is None:
            _, ax = plt.subplots(2)
        h = self._h
        for l in range(self.L):
            for m in range(self.M):
                ax[0].plot(h[:, m, l], label=f"l={l}, m={m}")
                ax[1].semilogy(np.abs(h[:, m, l]), label=f"l={l}, m={m}")
        ax[0].legend()
        ax[1].set_xlabel("taps")
        ax[0].set_ylabel("$h$")
        ax[1].set_ylabel("$|h|$")
        ax[0].grid()
        ax[1].grid()


class DataAidedEstimator(AdaptiveFIRFilter):
    """Data-aided reconstruction filter for an :class:`AnalogFrontend`.

    Unlike the analytical :class:`WienerFilter`, this estimator uses *nothing*
    from the analog-frontend state-space specification. It is calibrated from
    data: drive the frontend with a known reference, simulate, and fit the FIR
    taps by least squares (``data-aided`` = the calibration uses a known
    reference/training sequence). Construct it via :meth:`AnalogFrontend.calibrate`.
    """

    def __init__(
        self,
        analog_frontend: AnalogFrontend,
        DSR: int = 1,
        K=1 << 6,
        max_amplitude: float = 1.0,
        sim_size: int = 1 << 16,
        J: int = 1 << 1,
        seed: int = 213236546233421,
        rel_bw: float = 0.5,
        reference=None,
    ):
        M = analog_frontend.M
        L = analog_frontend.L
        dt = analog_frontend.dt
        self.DSR = DSR
        super().__init__(
            M=M, K=K, L=L, dt=dt, analog_frontend=analog_frontend, seed=seed
        )
        _ = self.learn_from_analog_frontend(
            DSR=DSR,
            max_amplitude=max_amplitude,
            sim_size=sim_size,
            J=J,
            rel_bw=rel_bw,
            reference=reference,
        )

    def learn_from_analog_frontend(
        self,
        DSR: int = 1,
        max_amplitude: float = 1.0,
        sim_size: int = 1 << 18,
        J: int = 1 << 0,
        rel_bw: float = 0.5,
        reference=None,
    ):
        self.DSR = DSR
        if reference is None:
            # full-scale, persistently-exciting reference (J parallel sequences)
            reference = ZeroOrderHold.uniform_reference_signal(
                self._analog_frontend.dt * DSR,
                -max_amplitude * np.ones((1, J), dtype=np.double),
                max_amplitude * np.ones((1, J), dtype=np.double),
                size=sim_size // J + self.K,
                seed=self._rng.integers(0, 1 << 62),
            )

        old_input_signal = _deepcopy(self._analog_frontend.analog_signal)
        self._analog_frontend.analog_signal = reference
        sim_res = self._analog_frontend.simulate(sim_size + self.K)

        dec_v = decimate(sim_res["v"][self.K :, :, :], DSR, method="direct")
        dec_u = decimate(sim_res["u"][self.K :, :, :], DSR, method="direct")

        self.lstsq(dec_v, dec_u, verbose=True, method="direct")
        self._analog_frontend.analog_signal = old_input_signal
        return sim_res

    def reconstruct(self, v: np.ndarray) -> np.ndarray:
        """Estimate the input from control signals ``v`` (decimates by ``DSR``).

        A thin alias over :meth:`convolve` that applies the calibrated
        decimation factor, so the typical readout is a single call.

        Parameters
        ----------
        v : numpy.ndarray, shape=(size, M, J)
            the control signals.

        Returns
        -------
        numpy.ndarray, shape=(size // DSR, L, J)
            the reconstructed input estimate.
        """
        return self.convolve(v, DSR=self.DSR)


# Deprecated alias: the estimator was previously named BlackBoxEstimator.
BlackBoxEstimator = DataAidedEstimator


def decimate(
    signal: np.ndarray,
    DSR: int,
    axis: int = 0,
    method="direct",
    ftype: str = "iir",
    n: int = 9,
) -> np.ndarray:
    """
    Decimate the signal by a factor of DSR

    Parameters
    ----------
    signal : array_like
        The signal to decimate
    DSR : int
        The decimation factor
    axis : int, optional
        The axis along which to decimate the signal, by default 0
    method : str, optional
        The decimation method, by default "direct". Options are:
        - "fft": uses Fourier method for resampling.
        - "direct": uses direct convolution method for decimation.
    ftype : str, optional
        The type of filter to use for direct method, by default 'iir'. Options are:
        - 'iir': uses an IIR filter.
        - 'fir': uses a FIR filter.
    n : int, optional
        The order of the filter to use for direct method, by default 15.
    """
    if isinstance(DSR, np.double):
        DSR = int(np.ceil(DSR))
    if method == "fft":
        logger.warning(
            """while more computationally efficient, FFT resampling has previously 
            shown to produce articfacts such as odd order harmonics.
            
            Prefered method is 'direct' convolution for decimation.
            """
        )
        return _resample(signal, signal.shape[axis] // DSR, axis=axis)
    elif method == "direct":
        # Find closest lower power of 2
        signal = signal.copy()
        while DSR % 2 == 0 and DSR > 1:
            signal = _decimate(signal, 2, axis=axis, ftype=ftype, n=n)
            DSR = DSR >> 1
        if DSR > 1:
            signal = _decimate(signal, DSR, axis=axis, ftype=ftype, n=n)
        return signal
    else:
        # TODO, test decimate, and sinc filters.
        raise NotImplementedError("Only Fourier and direct method is implemented")
