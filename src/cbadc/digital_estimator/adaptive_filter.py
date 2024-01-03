from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import logging
import cbadc

logger = logging.getLogger(__name__)


class AdaptiveFIRFilter:
    """
    Adaptive FIR filter model.

    The model is a simple linear model with a single dense layer (in ML language).
    Alternatively, a simple sum of FIR filters.

    The model is implemented as a Keras model, which means that it can be trained
    using the [Keras API](https://keras.io).
    Additionally, the model can be saved and loaded using the Keras API.

    To better understand the usecase of this class see
    [this notebooke](https://github.com/hammal/cbadc/blob/develop/notebooks/calibrate_a_leapfrog_ADC.ipynb).

    """

    def __init__(self, M, K, L=1, dtype=np.float64):
        self.K = K
        self.L = L
        self.M = M
        self._h_m = np.zeros((L, M, K), dtype=dtype)
        self._h = np.zeros((L, M, K), dtype=dtype)
        self.offset = np.zeros((L,), dtype=dtype)
        self.offset_m = np.zeros((L,), dtype=dtype)

    def compile(self, optimizer: Dict[str, Any]):
        """
        Compiles the model.

        Parameters
        ----------
        optimizer : Dict[str, Any]
            The optimizer to use.
        """
        self._learning_rate = optimizer.get("learning_rate", 1e-5)
        self._momentum = optimizer.get("momentum", 0.1)

    def loss(self, x: np.ndarray, y: np.ndarray):
        """
        Computes the loss function.

        Parameters
        ----------
        x : np.ndarray (batch_size, M, K)
            The input data.
        y: np.ndarray (L, batch_size)
            The output data.

        Returns
        -------
        loss : float
        """
        return np.linalg.norm(self.call(x) - y, axis=1) ** 2 / x.shape[0]

    def gradient(self, x: np.ndarray, y: np.ndarray):
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
        error = self.call(x) - y
        return (
            np.tensordot(
                error / error.shape[1],
                np.conj(x),
                axes=([1], [0]),
            ),
            error.mean(axis=1),
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        epochs: int,
        shuffle=False,
        verbose=True,
    ):
        if shuffle:
            raise NotImplementedError
        for e in cbadc.utilities.show_status(range(epochs)):
            for b in range(0, x.shape[0], batch_size):
                x_batch = x[b : b + batch_size, :, :]
                y_batch = y[:, b : b + batch_size]

                # [(L, M, K), (L,)]
                gradient = self.gradient(x_batch, y_batch)

                # (L, M, K)
                self._h_m *= self._momentum
                self._h_m += self._learning_rate * gradient[0]
                self._h -= self._h_m

                # (L,)
                self.offset_m *= self._momentum
                self.offset_m += self._learning_rate * gradient[1]
                self.offset -= self.offset_m
            if verbose:
                logger.info(
                    f"epoch {e}: loss = {self.loss(x, y)}, offset = {np.abs(self.offset)}"
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
        return np.tensordot(self._h, x, axes=([1, 2], [1, 2])) + self.offset

    def predict(self, x: np.ndarray):
        """
        Parameters
        ----------
        x : np.ndarray (batch_size, M, K)
            The input data, shape (nr_samples, M - nr_references).

        Returns
        -------
        y : np.ndarray (batch_size, L)
            The output data, shape (nr_samples, nr_references).
        """
        return self.call(x)

    def predict_full(self, x, y):
        """
        In contrast to the predict method, this method additionally removes
        the reference signals from the output.

        Parameters
        ----------
        x : np.ndarray
            The input data, shape (nr_samples, M - nr_references).
        y: np.ndarray
            The (h * s_reference)(kT) sequence, shape (nr_samples, nr_references).
        """
        return self.predict(x) - y

    def get_filters(self):
        """
        Returns the filters of the FIR filter.

        Parameters
        ----------
        M : int
            The number of control signals (note this excludes any reference
            signals as these are not part of the model).

        Returns
        -------
        (h, offset): tuple[np.ndarray, np.ndarray]
            The filter coefficients h and the offset.
            Where the shape of h is (1, M, K) and the shape of offset is (1,).
        """
        return self._h.reshape((self.L, self.M, self.K))

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
        h = self.get_filters()
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
                ax_h[0, m].set_title(f"impulse responses, L={l}")
                ax_h[1, m].set_xlabel(f"filter taps, L={l}")
                ax_h[0, m].set_ylabel("$h[.]$")
                ax_h[1, m].set_ylabel("$|h[.]|$")
                ax_h[0, m].grid(True)
                ax_h[1, m].grid(True)
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
        h = self.get_filters()
        f_h, ax_h = plt.subplots(2, self.L, sharex=True)
        for l in range(self.L):
            for m in range(self.M):
                h_version = h[l, m, :]

                h_freq = np.fft.rfft(h_version)
                freq = np.fft.rfftfreq(h_version.size)
                if self.L == 1:
                    ax = ax_h[:]
                else:
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
                ax[0].set_title(f"Bode diagram, L={l}")
                ax[1].set_xlabel("frequency [Hz]")
                ax[1].set_ylabel("$ \\angle h[.]$ rad")
                ax[0].set_ylabel("$|h[.]|$ dB")
                ax[0].grid(True)
                ax[1].grid(True)
            return f_h, ax_h
