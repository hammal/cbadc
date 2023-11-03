import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class AdaptiveFIRFilter(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.input_layer = tf.keras.layers.Flatten()
        self.output_layer = tf.keras.layers.Dense(1, activation=None, dtype=tf.float64)

    def call(self, x):
        x = self.input_layer(x)
        x = self.output_layer(x)
        return x

    def generate_dataset(self, control_signals: np.ndarray, h: np.ndarray):
        """
        Generates a dataset for training the FIR filter

        Parameters
        ----------
        control_signals : np.ndarray
            The control signals used to generate the dataset, shape (nr_samples, M).
            Note that the X number of reference sequences are assumed to be the in the first X columns.
        h: np.ndarray
            The reference filter, shape (nr_references, K).
            Note that the shape of h determines both number of references and general filter length K.

        """
        number_of_references = h.shape[0]
        K = h.shape[1]
        size = control_signals.shape[0] - K
        M = control_signals.shape[1]

        if number_of_references > M:
            raise ValueError(
                "The number of references must be less than or equal to the number of control signals"
            )

        x = np.zeros((size, (M - number_of_references) * K), dtype=np.int8)
        y = np.zeros((size, 1), dtype=float)
        for i in range(size):
            s_window = 2 * control_signals[i : i + K, :] - 1
            x[i, :] = s_window[:, number_of_references:].flatten()
            y[i, :] = np.sum(
                np.multiply(s_window[:, :number_of_references].transpose(), h)
            )

        logger.info(f"sizeof incoming data {control_signals.nbytes / (1 << 20)} MB")
        logger.info(f"sizeof outgoing training features {x.nbytes / (1 << 20)} MB")
        logger.info(f"sizeof outgoing training labels {y.nbytes / (1 << 20)} MB")

        return x, y

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
        return super().predict(x).flatten() - y.flatten()

    def get_filters(self, M: int):
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
        weights = self.get_weights()
        K = weights[0].size // M
        return weights[0].reshape((1, M, K), order="F"), weights[1]

    def plot_impulse_response(self, M: int):
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
        h = self.get_filters(M)[0]
        f_h, ax_h = plt.subplots(2, 1, sharex=True)
        for m in range(M):
            h_version = h[0, m, :]
            ax_h[0].plot(
                np.arange(h_version.size) - h_version.size // 2,
                h_version,
                label="$h_{" + f"{m}" + "}$",
            )
            ax_h[1].semilogy(
                np.arange(h_version.size) - h_version.size // 2,
                np.abs(h_version),
                label="$h_{" + f"{m}" + "}$",
            )
        ax_h[0].legend()
        ax_h[0].set_title("impulse responses")
        ax_h[1].set_xlabel("filter taps")
        ax_h[0].set_ylabel("$h[.]$")
        ax_h[1].set_ylabel("$|h[.]|$")
        ax_h[0].grid(True)
        ax_h[1].grid(True)
        return f_h, ax_h

    def plot_bode(self, M: int, linear_frequency: bool = False):
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
        h = self.get_filters(M)[0]
        f_h, ax_h = plt.subplots(2, 1, sharex=True)
        for m in range(M):
            h_version = h[0, m, :]

            h_freq = np.fft.rfft(h_version)
            freq = np.fft.rfftfreq(h_version.size)

            if linear_frequency:
                ax_h[1].plot(
                    freq,
                    np.angle(h_freq),
                    label="$h_{" + f"{m}" + "}$",
                )
                ax_h[0].plot(
                    freq,
                    20 * np.log10(np.abs(h_freq)),
                    label="$h_{" + f"{m}" + "}$",
                )
            else:
                ax_h[1].semilogx(
                    freq,
                    np.angle(h_freq),
                    label="$h_{" + f"{m}" + "}$",
                )
                ax_h[0].semilogx(
                    freq,
                    20 * np.log10(np.abs(h_freq)),
                    label="$h_{" + f"{m}" + "}$",
                )

        ax_h[0].legend()
        ax_h[0].set_title("Bode diagram")
        ax_h[1].set_xlabel("frequency [Hz]")
        ax_h[1].set_ylabel("$ \\angle h[.]$ rad")
        ax_h[0].set_ylabel("$|h[.]|$ dB")
        ax_h[0].grid(True)
        ax_h[1].grid(True)
        return f_h, ax_h
