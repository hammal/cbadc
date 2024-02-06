""" The adaptive FIR filter model.
"""

from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import logging
import cbadc

logger = logging.getLogger(__name__)


def batch(signal: np.ndarray, batch_size: int, axis: int = 0) -> np.ndarray:
    """
    Batch the signal for FIR filter processing

    Parameters
    ----------
    signal : array_like
        The signal to batch
    batch_size : int
        The batch size
    axis : int, optional
        The axis along which to batch the signal, by default 0

    Returns
    -------
    batched_signal : np.ndarray
        The batched signal where the last dimension is the batch dimension
    """

    if axis != 0:
        raise NotImplementedError("Only axis=0 is supported for now")

    old_shape = signal.shape
    batch_shape = []
    total_number_of_samples = signal.shape[axis] - batch_size
    for dim in old_shape:
        if dim == axis:
            batch_shape.append(total_number_of_samples)
        else:
            batch_shape.append(dim)
    batch_shape += [batch_size]

    batched_signal = np.zeros(batch_shape, dtype=signal.dtype)
    for i in range(total_number_of_samples):
        batched_signal[i, ...] = signal[i : i + batch_size, ...].transpose()

    return batched_signal


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

    def __init__(self, M, K, L=1, dtype=np.float64):
        self.K = K
        self.L = L
        self.M = M
        self._h_m = np.zeros((L, M, K), dtype=dtype)
        self._h = np.zeros((L, M, K), dtype=dtype)
        self._offset = np.zeros((L, 1), dtype=dtype)
        self._offset_m = np.zeros((L,), dtype=dtype)

    def loss(self, x: np.ndarray, y: np.ndarray):
        """
        Computes the loss function for the given FIR filter.

        Parameters
        ----------
        x : np.ndarray (batch_size, M, K)
            The input data.
        y: np.ndarray (L, batch_size)
            The reference data.

        Returns
        -------
        loss : float
            The loss function evaluated on the given data.
        """
        return np.linalg.norm(y - self.call(x), axis=1) ** 2 / x.shape[0]

    def gradient(self, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """
        Computes the gradient of the loss function with respect to the filter
        coefficients.

        Parameters
        ----------
        x : np.ndarray (batch_size, M, K)
            The input data.
        y: np.ndarray (L, batch_size)
            The output data.

        Returns
        -------
        gradient : [np.ndarray (L, M, K), np.ndarray (L,)]
            The gradient of the loss function with respect to the filter
            coefficients.
        """
        # (L, batch_size)
        error = np.conj(y - self.call(x))
        return [
            -np.tensordot(
                error / error.shape[1],
                x,
                axes=([1], [0]),
            ),
            -error.mean(axis=1),
        ]

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
        force_real_valued=False,
    ):
        """
        Fits the filter to the given data using the LMS method.

        Parameters
        ----------
        x : np.ndarray (batch_size, M, K)
            The input data.
        y: np.ndarray (L, batch_size)
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
        force_real_valued : bool
            Whether to force the filter coefficients to be real valued, defaults to False.
        """

        for e in cbadc.utilities.show_status(range(epochs)):
            if shuffle:
                permutation = np.random.permutation(x.shape[0])
                x = x[permutation, :, :]
                y = y[:, permutation]

            for b in range(0, x.shape[0], batch_size):
                x_batch = x[b : b + batch_size, :, :]
                y_batch = y[:, b : b + batch_size]

                # [(L, M, K), (L,)]
                gradient = self.gradient(x_batch, y_batch)
                if force_real_valued:
                    gradient[0] = np.real(gradient[0])
                    gradient[1] = np.real(gradient[1])

                # (L, M, K)
                self._h_m *= momentum
                self._h_m += learning_rate * gradient[0]
                self._h -= self._h_m

                # (L,)
                self._offset_m *= momentum
                self._offset_m += learning_rate * gradient[1]
                self._offset[:, 0] -= self._offset_m

            if verbose:
                logger.info(
                    f"epoch {e}: loss = {self.loss(x, y)}, offset = {self._offset}"
                )

    def rls(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        delta: float = 1e-2,
        lambda_: float = 1e0 - 1e-12,
        shuffle=False,
        verbose=True,
        force_real_valued=False,
    ):
        """
        Fits the filter to the given data using the RLS method.

        Parameters
        ----------
        x : np.ndarray (batch_size, M, K)
            The input data.
        y: np.ndarray (L, batch_size)
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
        force_real_valued : bool
            Whether to force the filter coefficients to be real valued, defaults to False.
        """

        # Lazy initialize Covariance matrix
        if hasattr(self, "V") is False:
            total_size = self.M * self.K + 1
            self.V = np.eye(total_size, dtype=x.dtype) / delta
            self._x_flatened = np.zeros((total_size), dtype=x.dtype)
            self._x_flatened[-1] = 1.0
        else:
            logger.warning(
                f"RLS algorithm already initialized, delta={delta} has no effect."
            )

        for e in cbadc.utilities.show_status(range(epochs)):
            if shuffle:
                permutation = np.random.permutation(x.shape[0])
                x = x[permutation, :, :]
                y = y[:, permutation]

            for b in range(x.shape[0]):
                x_batch = x[b : b + 1, :, :]
                self._x_flatened[:-1] = x_batch.flatten()
                y_batch = y[:, b : b + 1]

                error = np.conj(y_batch - self.call(x_batch))

                alpha = np.dot(self.V, self._x_flatened)
                g = alpha / (lambda_ + np.dot(self._x_flatened.conj(), alpha))

                self.V = (self.V - np.outer(g, alpha.conj())) / lambda_

                for l in range(self.L):
                    self._offset[l, 0] += g[-1] * error[l]
                    self._h[l, ...] += g[:-1].reshape((self.M, self.K)) * error[l]

            if verbose:
                logger.info(
                    f"epoch {e}: loss = {self.loss(x, y)}, offset = {self._offset}"
                )

    def call(self, x: np.ndarray):
        """
        Parameters
        ----------
        x : np.ndarray (batch_size, M, K)
            The input data, shape (nr_samples, M - nr_references).

        Returns
        -------
        y : np.ndarray (L, batch_size)
            The output data, shape (nr_samples, nr_references).
        """
        return np.tensordot(self._h.conj(), x, axes=([1, 2], [1, 2])) + np.conj(
            self._offset
        )

    def predict(self, x: np.ndarray):
        """
        Parameters
        ----------
        x : np.ndarray (batch_size, M, K)
            The input data, shape (nr_samples, M - nr_references).

        Returns
        -------
        y : np.ndarray (L, batch_size)
            The output data, shape (nr_samples, nr_references).
        """
        return self.call(x)

    def predict_full(self, x, y):
        """
        In contrast to the predict method, this method additionally removes
        the reference signals from the output.

        Parameters
        ----------
        x : np.ndarray [batch_size, M, K]
            The input data, shape (nr_samples, M - nr_references).
        y: np.ndarray [L, batch_size]
            The reference data.
        """
        return self.predict(x) - y

    def get_filter(self):
        """
        Returns the FIR filter.

        Returns
        -------
        h : np.ndarray (L, M, K)
            The FIR filter.
        """
        return np.copy(self._h.reshape((self.L, self.M, self.K)))

    def get_offset(self):
        """
        Returns the offset.

        Returns
        -------
        offset : np.ndarray (L,)
            The offset.
        """
        return np.copy(self._offset[:, 0])

    def plot_impulse_response(self):
        """
        Plots the impulse response of the filter for each channel.

        Parameters:
        -----------
        M : int
            The number of channels.

        Returns:
        --------
        f_h : matplotlib.figure.Figure
            The figure object.
        ax_h : numpy.ndarray
            The axes objects.
        """
        h = self.get_filter()
        if h.dtype == np.complex128:
            f_h, ax_h = plt.subplots(2, 2, sharex=True)
            for l in range(self.L):
                for m in range(self.M):
                    h_version = h[l, m, :]
                    ax_h[0, 0].plot(
                        np.arange(h_version.size) - h_version.size // 2,
                        np.real(h_version),
                        label="$real(h_{" + f"{l},{m}" + "})$",
                    )
                    ax_h[0, 1].plot(
                        np.arange(h_version.size) - h_version.size // 2,
                        np.imag(h_version),
                        label="$imag(h_{" + f"{l},{m}" + "})$",
                    )
                    ax_h[1, 0].semilogy(
                        np.arange(h_version.size) - h_version.size // 2,
                        np.abs(np.real(h_version)),
                        label="$real(h_{" + f"{l},{m}" + "})$",
                    )
                    ax_h[1, 1].semilogy(
                        np.arange(h_version.size) - h_version.size // 2,
                        np.abs(np.imag(h_version)),
                        label="$imag(h_{" + f"{l},{m}" + "})$",
                    )

                for m in range(2):
                    ax_h[0, m].legend()
                    ax_h[0, m].set_title(f"impulse responses, L={self.L}")
                    ax_h[1, m].set_xlabel("filter taps")
                    ax_h[0, m].set_ylabel("$h[.]$")
                    ax_h[1, m].set_ylabel("$|h[.]|$")
                    ax_h[0, m].grid(True)
                    ax_h[1, m].grid(True)
        else:
            f_h, ax_h = plt.subplots(2, 1, sharex=True)
            for l in range(self.L):
                for m in range(self.M):
                    h_version = h[l, m, :]
                    ax_h[0].plot(
                        np.arange(h_version.size) - h_version.size // 2,
                        h_version,
                        label="$h_{" + f"{l},{m}" + "}$",
                    )
                    ax_h[1].semilogy(
                        np.arange(h_version.size) - h_version.size // 2,
                        np.abs(h_version),
                        label="$h_{" + f"{l},{m}" + "}$",
                    )

                ax_h[0].legend()
                ax_h[0].set_title(f"impulse responses, L={self.L}")
                ax_h[1].set_xlabel(f"filter taps")
                ax_h[0].set_ylabel("$h[.]$")
                ax_h[1].set_ylabel("$|h[.]|$")
                ax_h[0].grid(True)
                ax_h[1].grid(True)

        return f_h, ax_h

    def plot_bode(self, linear_frequency: bool = False):
        """
        Plots the Bode diagram for the filter.

        Parameters:
        -----------
        M : int
            The number of channels.
        linear_frequency : bool
            If true, the frequency axis is linear, otherwise it is logarithmic.


        Returns:
        --------
        f_h : matplotlib.figure.Figure
            The figure object.
        ax_h : numpy.ndarray
            The axes objects.
        """
        h = self.get_filter()
        if self.L == 1:
            f_h, ax_h = plt.subplots(2, 1, sharex=True)
            for m in range(self.M):
                h_version = h[0, m, :]
                h_freq = np.fft.rfft(h_version)
                freq = np.fft.rfftfreq(h_version.size)
                ax = ax_h
                if linear_frequency:
                    ax[1].plot(
                        freq,
                        np.angle(h_freq),
                        label="$h_{" + f"{1},{m}" + "}$",
                    )
                    ax[0].plot(
                        freq,
                        20 * np.log10(np.abs(h_freq)),
                        label="$h_{" + f"{1},{m}" + "}$",
                    )
                else:
                    ax[1].semilogx(
                        freq,
                        np.angle(h_freq),
                        label="$h_{" + f"{1},{m}" + "}$",
                    )
                    ax[0].semilogx(
                        freq,
                        20 * np.log10(np.abs(h_freq)),
                        label="$h_{" + f"{1},{m}" + "}$",
                    )
                ax[0].legend()
                ax[0].set_title(f"Bode diagram, L={self.L}")
                ax[1].set_xlabel("frequency [Hz]")
                ax[1].set_ylabel("$ \\angle h[.]$ rad")
                ax[0].set_ylabel("$|h[.]|$ dB")
                ax[0].grid(True)
                ax[1].grid(True)
        else:
            f_h, ax_h = plt.subplots(2, self.L, sharex=True)
            for l in range(self.L):
                for m in range(self.M):
                    h_version = h[l, m, :]
                    h_freq = np.fft.rfft(h_version)
                    freq = np.fft.rfftfreq(h_version.size)
                    ax = ax_h[:, l]
                    if linear_frequency:
                        ax[1].plot(
                            freq,
                            np.angle(h_freq),
                            label="$h_{" + f"{l},{m}" + "}$",
                        )
                        ax[0].plot(
                            freq,
                            20 * np.log10(np.abs(h_freq)),
                            label="$h_{" + f"{l},{m}" + "}$",
                        )
                    else:
                        ax[1].semilogx(
                            freq,
                            np.angle(h_freq),
                            label="$h_{" + f"{l},{m}" + "}$",
                        )
                        ax[0].semilogx(
                            freq,
                            20 * np.log10(np.abs(h_freq)),
                            label="$h_{" + f"{l},{m}" + "}$",
                        )
                    ax[0].legend()
                    ax[0].set_title(f"Bode diagram, L={self.L}")
                    ax[1].set_xlabel("frequency [Hz]")
                    ax[1].set_ylabel("$ \\angle h[.]$ rad")
                    ax[0].set_ylabel("$|h[.]|$ dB")
                    ax[0].grid(True)
                    ax[1].grid(True)
        return f_h, ax_h
