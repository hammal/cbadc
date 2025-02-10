"""The digital backend.

This module contains classes for digital signal processing and general 
post-processing of the analog frontend output.
"""

from typing import Optional, Union
import numpy as np
from .analog_frontend import AnalogFrontend
from scipy.linalg import (
    solve_continuous_are as _care,
    expm as _expm,
    solve_discrete_are as _dare,
)
from scipy.integrate import solve_ivp as _solve_ivp
from scipy.signal import (
    fftconvolve as _fftconvolve,
    TransferFunction,
    resample as _resample,
    decimate as _decimate,
)
from numpy.lib.stride_tricks import sliding_window_view as _sliding_window_view
from .utilities import show_status as _show_status
import logging as _logging
import matplotlib.pyplot as plt


logger = _logging.getLogger(__name__)


class WienerFilter:
    """The analytical Wiener filter for control-bounded converters.

    The Wiener filter is a linear filter that minimizes the mean square error
    between the output of the analog frontend and the output of the filter.

    Parameters
    ----------
    analog_frontend : :py:class:`cbadc.AnalogFrontend`
        The analog frontend.
    eta2 : `float`
        The noise variance and bandwidth term.

    """

    def __init__(self, analog_frontend: AnalogFrontend, eta2: float):
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
    def eta2(self) -> float:
        """Return the noise variance.

        Returns
        -------
        eta2 : `float`
            The noise variance.
        """
        return self._eta2

    @eta2.setter
    def eta2(self, eta2: float):
        """Set the noise variance and bandwidth term.

        Parameters
        ----------
        eta2 : `float`
            The noise variance ratio between input and
            output noise.
        """
        if not isinstance(eta2, (float)) or eta2 <= 0:
            raise ValueError("eta2 must be a positive float")

        self._eta2 = eta2

        L = self._analog_frontend.L
        N = self._analog_frontend.N
        M = self._analog_frontend.M

        if self._analog_frontend.is_discrete_time:
            logger.warning(
                "Discrete time Wiener filter not properly implemented. Results may be incorrect."
            )
        # #     # Compute modified Bryson-Frazier smoother
        # #     A_dare: np.ndarray = self._analog_frontend.A.T.conjugate()
        # #     B_dare = np.eye(N, dtype=float)
        # #     Q_dare = self._analog_frontend.B[:, :L] @ self._analog_frontend.B[:, :L].T
        # #     R_dare = self._eta2 * np.eye(N, dtype=float)
        # #     V_X_f = _dare(A_dare, B_dare, Q_dare, R_dare)
        # #     G = np.linalg.inv(R_dare + V_X_f)
        # #     F = np.eye(N, dtype=float) - V_X_f @ G
        # #     self._Af = self._analog_frontend.A @ F
        # #     self._Bf = self._analog_frontend.B[:, L:]
        # #     self._Ab = F.transpose() @ self._analog_frontend.A.transpose()
        # #     self._Bb = G
        # #     self._W = -self._analog_frontend.B[:, :L].transpose()
        # else:
        # Compute the Wiener filter
        # Algebraic Riccati equation Notation
        A_care: np.ndarray = self._analog_frontend.A.T
        B_care = np.eye(N, dtype=float)
        # Q = B B^T
        Q_care = self._analog_frontend.B[:, :L] @ self._analog_frontend.B[:, :L].T
        R_care = self._eta2 * np.eye(N, dtype=float)

        # Compute stationary covariance matrices
        V_f = _care(A_care, B_care, Q_care, R_care)
        V_b = _care(-A_care, B_care, Q_care, R_care)

        self._W = np.linalg.solve(V_f + V_b, self._analog_frontend.B[:, :L]).T

        dt = self._analog_frontend.dt

        if self._analog_frontend.digital_control.dac_waveform == "nrz":
            tmp_arg = np.vstack(
                (
                    np.hstack(
                        (
                            self._analog_frontend.A - V_f / self._eta2,
                            self._analog_frontend.B[:, L:],
                        )
                    ),
                    np.zeros((M, N + M), dtype=float),
                )
            )
            tmp = _expm(tmp_arg * dt)
            self._Af = tmp[:N, :N]
            self._Bf = tmp[:N, N:]

            tmp_arg = np.vstack(
                (
                    np.hstack(
                        (
                            -self._analog_frontend.A - V_b / self._eta2,
                            -self._analog_frontend.B[:, L:],
                        )
                    ),
                    np.zeros((M, N + M), dtype=float),
                )
            )
            tmp = _expm(tmp_arg * dt)
            self._Ab = tmp[:N, :N]
            self._Bb = tmp[:N, N:]
        else:
            tmp_Af = self._analog_frontend.A - V_f / self._eta2
            tmp_Ab = -self._analog_frontend.A - V_b / self._eta2
            self._A_f = _expm(tmp_Af * dt)
            self._A_b = _expm(tmp_Ab * dt)

            def der_f(t: float, x: np.ndarray):
                return tmp_Af @ x + self._analog_frontend.B[
                    :, L:
                ] * self._analog_frontend.digital_control.impulse_response(
                    np.array([t])
                ).reshape(
                    (1, -1)
                )

            res = _solve_ivp(der_f, (0, dt), np.zeros(N * M, dtype=float))
            self._Bf = res.y[:, -1].reshape((N, M))

            def der_b(t: float, x: np.ndarray):
                return tmp_Ab @ x - self._analog_frontend.B[
                    :, L:
                ] * self._analog_frontend.digital_control.impulse_response(
                    np.array([t])
                ).reshape(
                    (1, -1)
                )

            res = _solve_ivp(der_b, (0, dt), np.zeros(N * M, dtype=float))
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

    def evaluate(self, s: np.ndarray) -> np.ndarray:
        """Evaluate the Wiener filter.

        Parameters
        ----------
        s : :py:class:`numpy.ndarray`, shape=(size, M, J)
            input control signals.

        Returns
        -------
        y : :py:class:`numpy.ndarray`, shape=(size, L, J)
            output estimates signal.

        """
        M = self._analog_frontend.M
        if s.shape[1] != M:
            raise ValueError(
                "s must have the shape = (size, M), where M is the number of control signals"
            )

        # allocate memory
        size = s.shape[0]
        N = self._analog_frontend.N
        L = self._analog_frontend.L
        J = self._analog_frontend.J
        m_v = np.zeros((size + 2, N, J), dtype=float)
        u_hat = np.zeros((size, L, J), dtype=float)

        # Forward message passing
        for i in range(size):
            m_v[i + 1] = self._Af @ m_v[i] + self._Bf @ s[i]
        # Backward message passing
        for i in range(size - 1, -1, -1):
            m_v[i + 1] = self._Ab @ m_v[i + 2] + self._Bb @ s[i]
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

    def __call__(self, s: np.ndarray) -> np.ndarray:
        """Evaluate the Wiener filter.

        Parameters
        ----------
        s : :py:class:`numpy.ndarray`, shape=(size, M)
            input control signals.

        Returns
        -------
        y : :py:class:`numpy.ndarray`, shape=(size, L)
            output estimates signal.

        """
        return self.evaluate(s)


class FIRFilter:

    def __init__(self, h: np.ndarray, offset: np.ndarray):
        self.h = h
        self.offset = offset


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
        dtype: np.dtype = np.float64,
        seed: int = 34981723498712594372,
        dt: float = 1.0,
        analog_frontend: Optional[AnalogFrontend] = None,
    ):
        self._K = K
        self._L = L
        self._M = M
        self._h = np.zeros((L, M, K), dtype=dtype)
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
        if self._h.shape[2] != K:
            self._h = np.zeros((self.L, self.M, K), dtype=self.dtype)
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
        if self._h.shape[0] != L:
            self._h = np.zeros((L, self.M, self.K), dtype=self.dtype)
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
            self._h = np.zeros((self.L, M, self.K), dtype=self.dtype)
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
                tmp_tfs.append(TransferFunction(self._h[l, m, ::-1], a, dt=self.dt))
            tfs.append(tmp_tfs)
        return tfs

    @h.setter
    def h(self, h):
        if not isinstance(h, np.ndarray):
            raise ValueError("h must be a numpy array")
        if h.shape != self._h.shape:
            raise ValueError("h must have the shape (L, M, K)")
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

    def convolve(self, x: np.ndarray, method: str = "fft"):
        """Convolve filter taps with input data.

        Parameters
        ----------
        x : :py:class:`numpy.ndarray`, shape=(size, M)
            The input data, shape (nr_samples, M - nr_references).
        method : str
            The convolution method to use, defaults to "fft".

            - "fft" uses the FFT to compute the convolution, typically
            more efficient for sizeable arrays.
            - "conv" uses dot products and sliding windows


        Returns
        -------
        y : :py:class:`numpy.ndarray`, shape=(size, L)
            The output data, shape (nr_samples, nr_references).
        """

        if method == "fft":
            x_reshaped = x.transpose(1, 0)[np.newaxis, :, :]
            # x_reshaped.shape = (1, M, size)

            # _fftconvolve(x_reshaped, self._h, mode="same", axes=2).shape = (L, M, size)

            # shape = (size, L)
            return (
                np.sum(
                    _fftconvolve(x_reshaped, self._h, mode="same", axes=2), axis=1
                ).transpose(1, 0)
                + self._offset[np.newaxis, :]
            )
        elif method == "conv":
            # sliding view
            x_window = _sliding_window_view(x, self.K, axis=0)
            # x_window.shape = (size - K + 1, K, M)
            return self._sliding_window_tensor_dot_convolution(x_window)

        else:
            raise ValueError("method must be either 'fft' or 'conv'")

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
        return (
            np.tensordot(
                x,
                self._h,
                axes=([1, 2], [1, 2]),
            )
            + self._offset[np.newaxis, :]
        )

    def loss(self, x: np.ndarray, y: np.ndarray):
        """Computes the loss, i.e., the squared L2 norm, for the given FIR filter.

        Returns

        ||y - h * x||^2 / size



        Parameters
        ----------
        x : :py:class:`numpy.ndarray`, shape=(size, M)
            The input data.
        y: :py:class:`numpy.ndarray`, shape=(size, L)
            The reference data.

        Returns
        -------
        loss : :py:class:`numpy.ndarray`, shape=(L,)
            The loss function evaluated on the given data.
        """
        y_hat = self.convolve(x)
        size = np.minimum(y_hat.shape[0], y.shape[0])
        return np.linalg.norm(y[:size, :] - y_hat[:size, :], axis=0) ** 2 / size

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
        gradient : [np.ndarray (L, M, K), np.ndarray (L,)]
            The gradient of the loss function with respect to the filter
            coefficients.
        """
        batch_size = x.shape[0]
        error = y - self._sliding_window_tensor_dot_convolution(x)
        return -np.tensordot(error, x, axes=([0], [0])) / batch_size, -error.mean(
            axis=0
        )

    def lms(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        epochs: int,
        learning_rate: float = 1e-5,
        momentum: float = 0.9,
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
        learning_rate : float
            The learning rate, defaults to 1e-5.
        momentum : float
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
                logger.info(
                    "epoch %d: loss = %s, offset = %s", e, self.loss(x, y), self._offset
                )
        return self.loss(x, y)

    def rls(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        delta: float = 1e-2,
        lambda_: float = 1e0 - 1e-12,
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
        delta : float
            The delta parameter of the RLS algorithm.
        lambda : float
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
                self._h += (
                    g[:-1].reshape((1, self.M, self.K))
                    * error[0, :, np.newaxis, np.newaxis]
                )

            if verbose:
                logger.info(
                    "epoch %d: loss = %s, offset = %s", e, self.loss(x, y), self._offset
                )
        return self.loss(x, y)

    def lstsq(self, x: np.ndarray, y: np.ndarray, verbose=True, rcond=None):
        """
        Fits the filter to the given data using the least squares method.

        Parameters
        ----------
        x : np.ndarray (size, M)
            The input data.
        y: np.ndarray (size, L)
            The reference data.
        verbose : bool
            Whether to print the loss function during training.
        rcond : float
            The reciprocal condition number for the least squares method.

        Returns
        -------
        loss : np.ndarray (L,)
            The loss function evaluated on the given data.
        """
        if x.shape[1] != self.M:
            raise ValueError("x must have the shape (size, M)")
        if y.shape[1] != self.L:
            raise ValueError("y must have the shape (size, L)")

        # (size - K + 1, M, J, K)
        x_window = _sliding_window_view(x, self.K, axis=0)
        new_size = np.minimum(x_window.shape[0], y.shape[0])
        # shape = (size - K + 1, L)
        x_window = x_window[:new_size, :, :]
        y_window = y[:new_size, :]

        # (batch_size, M * K + 1)
        x_vec_with_offset = np.hstack(
            (x_window.reshape((new_size, -1)), np.ones((new_size, 1)))
        )
        # (M * K + 1, L)
        sol = np.linalg.lstsq(np.conj(x_vec_with_offset), y_window, rcond=rcond)
        self._offset[:] = sol[0][-1, :]
        self._h[:] = (
            sol[0][:-1, :].reshape((self.M, self.K, self.L)).transpose(2, 0, 1)[:, :, :]
        )
        if verbose:
            loss = sol[1] / new_size
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

    def plot_amplitude_response(self, jw: np.ndarray, ax=None):
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
                ax[0].plot(h[l, m, :], label=f"l={l}, m={m}")
                ax[1].semilogy(np.abs(h[l, m, :]), label=f"l={l}, m={m}")
        ax[0].legend()
        ax[1].set_xlabel("taps")
        ax[0].set_ylabel("$h$")
        ax[1].set_ylabel("$|h|$")
        ax[0].grid()
        ax[1].grid()


def decimate(
    signal: np.ndarray, DSR: int, axis: int = 0, method="Fourier"
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
    """
    if isinstance(DSR, float):
        DSR = int(np.ceil(DSR))
    if method == "Fourier":
        return _resample(signal, signal.shape[axis] // DSR, axis=axis)
    else:
        # TODO, test decimate, and sinc filters.
        raise NotImplementedError("Only Fourier method is implemented")
