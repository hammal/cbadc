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
    for index, dim in enumerate(old_shape):
        if index == axis:
            batch_shape.append(total_number_of_samples)
        else:
            batch_shape.append(dim)
    batch_shape += [batch_size]

    batched_signal = np.zeros(batch_shape, dtype=signal.dtype)
    for i in range(total_number_of_samples):
        batched_signal[i, ...] = signal[i : i + batch_size, ...].T

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
        loss : np.ndarray (L,)
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
        delay: int = 0,
        shuffle=False,
        verbose=True,
    ):
        """
        Fits the filter to the given data using the LMS method.

        Parameters
        ----------
        x : np.ndarray (batch_size, M)
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
        delay : int
            The delay of the FIR filter. must be non-negative.
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
                f"""
Training using LMS for:
- epochs = {epochs},
- batch size = {batch_size},
- learning rate = {learning_rate},
- momentum = {momentum},
- and shuffle set to {shuffle}."""
            )
        x, y = self._batch(x, y, delay)

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
        return self.loss(x, y)

    def _batch(self, x: np.ndarray, y: np.ndarray, delay: int):
        """
        Parameters
        ----------
        x : np.ndarray (batch_size, M)
            The input data.
        y: np.ndarray (L, batch_size)
            The reference data.
        delay : int
            The delay of the FIR filter. must be non-negative.

        Returns
        -------
        x : np.ndarray (length, M, K)
            The modified input data.
        y : np.ndarray (L, length)
            The modified reference data.

        where length = min(x.shape[0], y.shape[1])
        """
        if delay < 0:
            raise ValueError("delay must be non-negative")
        x = batch(x[delay:, :], self.K)
        length = min(x.shape[0], y.shape[1])
        x = x[:length, ...]
        y = y[:, :length]
        return x, y

    def rls(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        delta: float = 1e-2,
        lambda_: float = 1e0 - 1e-12,
        delay: int = 0,
        shuffle=False,
        verbose=True,
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
        delay : int
            The delay of the FIR filter. must be non-negative.
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
                f"""
Training using RLS for:
- epochs = {epochs},
- delta = {delta},
- lambda = {lambda_},
- and shuffle set to {shuffle}."""
            )

        x, y = self._batch(x, y, delay)

        # Lazy initialize Covariance matrix
        if hasattr(self, "V") is False:
            total_size = np.prod(self._h.shape[1:]) + 1
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
                    self._h[l, ...] += g[:-1].reshape((-1, self.K)) * error[l]

            if verbose:
                logger.info(
                    f"epoch {e}: loss = {self.loss(x, y)}, offset = {self._offset}"
                )
        return self.loss(x, y)

    def lstsq(self, x: np.ndarray, y: np.ndarray, delay: int = 0, verbose=True):
        """
        Fits the filter to the given data using the least squares method.

        Parameters
        ----------
        x : np.ndarray (batch_size, M)
            The input data.
        y: np.ndarray (L, batch_size)
            The reference data.
        delay : int
            The delay of the FIR filter. must be non-negative.
        verbose : bool
            Whether to print the loss function during training.

        Returns
        -------
        loss : np.ndarray (L,)
            The loss function evaluated on the given data.
        """
        x, y = self._batch(x, y, delay)

        # (batch_size, M * K + 1)
        x_with_offset = np.hstack(
            (x.reshape((x.shape[0], -1)), np.ones((x.shape[0], 1)))
        )
        # (M * K + 1, L)
        sol = np.linalg.lstsq(x_with_offset, y.T, rcond=None)
        self._offset = sol[0][-1, :].T
        self._h = sol[0][:-1, :].T.reshape((self.L, -1, self.K))
        if verbose:
            logger.info(f"loss = {sol[1]}, offset = {self._offset}")
        return sol[1]

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
        x : np.ndarray (batch_size, M)
            The input data.

        Returns
        -------
        y : np.ndarray (L, batch_size)
            The output data, shape (nr_samples, nr_references).
        """

        return self.call(batch(x, self.K))

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
        return np.copy(self._h.reshape((self.L, -1, self.K)))

    def get_offset(self):
        """
        Returns the offset.

        Returns
        -------
        offset : np.ndarray (L,)
            The offset.
        """
        return np.copy(self._offset[:, 0])

    def impulse_response(self, number_of_points: int = 0):
        """
        Returns the impulse response of the filter.

        Parameters
        ----------
        number_of_points : int
            The number of uniformly spaced frequency points to evaluate the transfer function at, defaults to K.

        Returns
        -------
        h : np.ndarray (L, M, K)
            The impulse response.
        """
        if number_of_points != 0:
            raise ValueError("number_of_points must be 0 for FIR filters")
        return self.get_filter()

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
        h = self.impulse_response()
        if h.dtype == np.complex128:
            f_h, ax_h = plt.subplots(2, 2, sharex=True)
            for l in range(h.shape[0]):
                for m in range(h.shape[1]):
                    h_version = h[l, m, :]
                    ax_h[0, 0].plot(
                        np.arange(h_version.size) - h_version.size // 2,
                        np.real(h_version),
                        label="$real(h_{" + f"{l + 1},{m + 1}" + "})$",
                    )
                    ax_h[0, 1].plot(
                        np.arange(h_version.size) - h_version.size // 2,
                        np.imag(h_version),
                        label="$imag(h_{" + f"{l + 1},{m + 1}" + "})$",
                    )
                    ax_h[1, 0].semilogy(
                        np.arange(h_version.size) - h_version.size // 2,
                        np.abs(np.real(h_version)),
                        label="$real(h_{" + f"{l + 1},{m + 1}" + "})$",
                    )
                    ax_h[1, 1].semilogy(
                        np.arange(h_version.size) - h_version.size // 2,
                        np.abs(np.imag(h_version)),
                        label="$imag(h_{" + f"{l + 1},{m + 1}" + "})$",
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
            for l in range(h.shape[0]):
                for m in range(h.shape[1]):
                    h_version = h[l, m, :]
                    ax_h[0].plot(
                        np.arange(h_version.size) - h_version.size // 2,
                        h_version,
                        label="$h_{" + f"{l + 1},{m + 1}" + "}$",
                    )
                    ax_h[1].semilogy(
                        np.arange(h_version.size) - h_version.size // 2,
                        np.abs(h_version),
                        label="$h_{" + f"{l + 1},{m + 1}" + "}$",
                    )

                ax_h[0].legend()
                ax_h[0].set_title(f"impulse responses, L={self.L}")
                ax_h[1].set_xlabel(f"filter taps")
                ax_h[0].set_ylabel("$h[.]$")
                ax_h[1].set_ylabel("$|h[.]|$")
                ax_h[0].grid(True)
                ax_h[1].grid(True)

        return f_h, ax_h

    def transfer_function(self, number_of_points: int = 0):
        """
        Returns the transfer function of the filter.

        Parameters
        ----------
        number_of_points : int
            The number of uniformly spaced frequency points to evaluate the transfer function at, defaults to K.


        Returns
        -------
        freqs : np.ndarray
            The frequency points.
        tf : np.ndarray (L, M, number_of_points)
            The transfer function.
        """
        if number_of_points == 0:
            number_of_points = self.K
        tf = np.zeros(
            (self.L, self.M, (number_of_points >> 1) + 1), dtype=np.complex128
        )
        freqs = np.fft.rfftfreq(number_of_points)
        for l in range(tf.shape[0]):
            for m in range(tf.shape[1]):
                tf[l, m, :] = np.fft.rfft(self._h[l, m, :])
        return freqs, tf

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
        freq, h_freq = self.transfer_function()
        f_h, ax_h = plt.subplots(2, self.L, sharex=True)
        for l in range(self.L):
            for m in range(self.M):
                if self.L == 1:
                    ax = ax_h
                else:
                    ax = ax_h[:, l]
                if linear_frequency:
                    ax[1].plot(
                        freq,
                        np.angle(h_freq[l, m, :]),
                        label="$h_{" + f"{l + 1},{m + 1}" + "}$",
                    )
                    ax[0].plot(
                        freq,
                        20 * np.log10(np.abs(h_freq[l, m, :])),
                        label="$h_{" + f"{l + 1},{m + 1}" + "}$",
                    )
                else:
                    ax[1].semilogx(
                        freq,
                        np.angle(h_freq[l, m, :]),
                        label="$h_{" + f"{l + 1},{m + 1}" + "}$",
                    )
                    ax[0].semilogx(
                        freq,
                        20 * np.log10(np.abs(h_freq[l, m, :])),
                        label="$h_{" + f"{l + 1},{m + 1}" + "}$",
                    )
                ax[0].legend()
                ax[0].set_title(f"Bode diagram, L={self.L}")
                ax[1].set_xlabel("frequency [Hz]")
                ax[1].set_ylabel("$ \\angle h[.]$ rad")
                ax[0].set_ylabel("$|h[.]|$ dB")
                ax[0].grid(True)
                ax[1].grid(True)
        return f_h, ax_h


class AdaptiveIIRFilter(AdaptiveFIRFilter):
    """The adaptive IIR filter model."""

    def __init__(self, M, K, L=1, dtype=np.float64):
        super().__init__(M + 1, K, L, dtype)
        self.M = M

    def _numerator(self):
        return self._h[:, :-1, :]

    def _denominator(self):
        return self._h[:, -1, :]

    def _dlti(self, l, m):
        return signal.dlti(self._numerator()[l, m, ::1], self._denominator()[l, ::1])

    def predict(self, x: np.ndarray):
        """
        Parameters
        ----------
        x : np.ndarray (batch_size, M)
            The input data batch.

        Returns
        -------
        y : np.ndarray (L, batch_size)
            The output data, shape (nr_samples, nr_references).
        """
        x = batch(x, self.K)
        # FIR filter part (L, batch_size)
        batch_size = x.shape[0]
        numerator = self._numerator()
        denominator = self._denominator()

        fir_output = np.tensordot(numerator, x, axes=([1, 2], [1, 2])) + np.conj(
            self._offset
        )
        res = np.zeros((self.L, x.shape[0], self.K + 1), dtype=x.dtype)
        for index in range(batch_size):
            # IIR filter part (L, batch_size)
            res[:, index, -1] = fir_output[:, index] + np.tensordot(
                denominator, res[:, index, :-1], axes=([1], [1])
            )
            if index < batch_size - 1:
                res[:, index + 1, :-1] = res[:, index, 1:]

        return res[..., -1]

    def transfer_function(self, number_of_points: int = 0):
        """
        Returns the transfer function of the filter.

        Parameters
        ----------
        number_of_points : int
            The number of uniformly spaced frequency points to evaluate the transfer function at, defaults to K.

        Returns
        -------
        freqs : np.ndarray
            The frequency points.
        tf : np.ndarray (L, M, number_of_points)
        """

        if number_of_points == 0:
            number_of_points = self.K

        tf = np.zeros((self.L, self.M, number_of_points), dtype=np.complex128)
        for l in range(self.L):
            for m in range(self.M):
                freqs, tf[l, m, :] = self._dlti(l, m).freqresp(n=number_of_points)
        return freqs, tf

    # def impulse_response(self, number_of_points: int = 0):
    #     """
    #     Returns the impulse response of the filter.

    #     Parameters
    #     ----------
    #     number_of_points : int
    #         The number of uniformly spaced frequency points to evaluate the transfer function at, defaults to K.

    #     Returns
    #     -------
    #     h : np.ndarray (L, M, number_of_points)
    #         The impulse response.
    #     """
    #     if number_of_points == 0:
    #         number_of_points = self.K
    #     h = np.zeros((self.L, self.M, number_of_points))
    #     for l in range(self.L):
    #         for m in range(self.M):
    #             h[l, m, :] = (
    #                 self._dlti(l, m).impulse(n=number_of_points)[1][l].flatten()
    #             )
    #     return h

    def _reshape_data_for_FIR_filter_training(
        self, x: np.ndarray, y: np.ndarray, delay: int = 0
    ):
        """
        Parameters
        ----------
        x : np.ndarray (batch_size, M, K)
            The input data.
        y: np.ndarray (L, batch_size)
            The reference data.
        delay : int
            The delay of the IIR filter. must be non-negative.

        Returns
        -------
        x_concat : np.ndarray (length, (M + L) * K)
            The modified input data.
        y_batched : np.ndarray (L, length)
            The modified reference data.
        """
        x = x[self.K + delay :, :]
        length = min(x.shape[0], y.shape[1] - self.K)
        # (length, M + L)
        x_concat = np.concatenate((x[:length, :], y[:, :length].T), axis=1)
        return x_concat, y[:, self.K : self.K + length]

    def lstsq(self, x: np.ndarray, y: np.ndarray, delay: int = 0, verbose=True):
        """
        Fits the filter to the given data using the least squares method.

        Parameters
        ----------
        x : np.ndarray (batch_size, M)
            The input data.
        y: np.ndarray (L, batch_size)
            The reference data.
        delay : int
            The delay of the IIR filter. must be non-negative.
        verbose : bool
            Whether to print the loss function during training.

        Returns
        -------
        loss : np.ndarray (L,)
            The loss function evaluated on the given data.
        """
        x, y = self._reshape_data_for_FIR_filter_training(x, y, delay)
        return super().lstsq(x, y, delay=0, verbose=verbose)

    def lms(self, x: np.ndarray, y: np.ndarray, delay: int = 0, **kwargs):
        """
        Fits the filter to the given data using the LMS method.

        Parameters
        ----------
        x : np.ndarray (batch_size, M)
            The input data.
        y: np.ndarray (L, batch_size)
            The reference data.
        delay : int
            The delay of the IIR filter. must be non-negative.
        batch_size : int
            The batch size.
        epochs : int
            The number of epochs.
        learning_rate : float
            The learning rate, defaults to 1e-5.
        momentum : float
            The momentum, defaults to 0.9.
        verbose : bool
            Whether to print the loss function during training.


        Returns
        -------
        loss : np.ndarray (L,)
            The loss function evaluated on the given data.
        """
        if "shuffle" in kwargs:
            raise ValueError("shuffle is not supported for IIR filters")
        x, y = self._reshape_data_for_FIR_filter_training(x, y, delay)
        return super().lms(x, y, **kwargs)

    def rls(self, x: np.ndarray, y: np.ndarray, delay: int = 0, **kwargs):
        """
        Fits the filter to the given data using the RLS method.

        Parameters
        ----------
        x : np.ndarray (batch_size, M)
            The input data.
        y: np.ndarray (L, batch_size)
            The reference data.
        delay : int
            The delay of the IIR filter. must be non-negative.
        epochs : int
            The number of epochs.
        delta : float
            The delta parameter of the RLS algorithm.
        lambda : float
            The lambda parameter of the RLS algorithm.
        verbose : bool
            Whether to print the loss function during training.


        Returns
        -------
        loss : np.ndarray (L,)
            The loss function evaluated on the given data.
        """
        if "shuffle" in kwargs:
            raise ValueError("shuffle is not supported for IIR filters")
        x, y = self._reshape_data_for_FIR_filter_training(x, y, delay)
        return super().rls(x, y, **kwargs)


def find_IIR_filter(
    x_train: np.ndarray,
    y_train: np.ndarray,
    K: int = 1 << 5,
    delay=None,
    dtype=np.float64,
) -> AdaptiveIIRFilter:
    """

    Parameters
    ----------
    x_train : np.ndarray (train_size, M)
        The input training data.
    y_train: np.ndarray (L, train_size)
        The reference training data.
    K : int
        The number of filter taps per control signal.
    delay : [int, int]
        A range of possible delays for the IIR filter, defaults to [0, K].
    dtype : np.dtype
        The data type of the filter coefficients.
    """
    if delay is None:
        delay = [0, K]

    best_loss = np.inf
    best_filter = None
    x_train_batch = batch(x_train, K)
    for d in range(delay[0], delay[1] + 1):
        filter = AdaptiveIIRFilter(x_train.shape[1], K, y_train.shape[0], dtype)
        loss = np.linalg.norm(
            filter.lstsq(x_train_batch, y_train, delay=d, verbose=False)
        )
        if loss < best_loss:
            best_loss = loss
            best_filter = filter
    if best_filter is None:
        raise ValueError("No filter found")
    return best_filter


def find_IIR_filter_snr(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    K: int = 1 << 5,
    delay=None,
    dtype=np.float64,
) -> AdaptiveIIRFilter:
    """

    Parameters
    ----------
    x_train : np.ndarray (train_size, M)
        The input training data.
    x_test : np.ndarray (test_size, M)
        The input test data.
    y_train: np.ndarray (L, train_size)
        The reference training data.
    K : int
        The number of filter taps per control signal.
    delay : [int, int]
        A range of possible delays for the IIR filter, defaults to [0, K].
    dtype : np.dtype
        The data type of the filter coefficients.
    """
    if delay is None:
        delay = [0, K]

    best_snr = -np.inf
    best_filter = None
    x_train_batch = batch(x_train, K)
    x_test_batch = batch(x_test, K)
    for d in range(delay[0], delay[1] + 1):
        filter = AdaptiveIIRFilter(x_train.shape[1], K, y_train.shape[0], dtype)
        _ = np.linalg.norm(filter.lstsq(x_train_batch, y_train, delay=d, verbose=False))
        u_hat = filter.predict(x_test_batch).flatten()
        _, psd = cbadc.utilities.compute_power_spectral_density(
            u_hat,
        )
        signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
        noise_index = np.ones(psd.size, dtype=bool)
        noise_index[:3] = False
        noise_index[signal_index] = False
        res = cbadc.utilities.snr_spectrum_computation_extended(
            psd, signal_index, noise_index
        )
        if res["snr"] > best_snr:
            best_snr = res["snr"]
            best_filter = filter
    if best_filter is None:
        raise ValueError("No filter found")
    return best_filter
