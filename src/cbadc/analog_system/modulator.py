import numpy as np
import scipy.signal
import scipy.linalg
import logging
from typing import Union
import sympy as sp
from .analog_system import AnalogSystem

logger = logging.getLogger(__name__)


class SineWaveModulator(AnalogSystem):
    """Sinewave modulator

    This class represents a sinewave modulator with a given modulation
    frequency and permutation matrix.

    Parameters
    ----------
    modulation_frequency : `float`
        the modulation frequency
    permuation_matrix : `array_like`, shape=(N, N)
        the permutation matrix

    """

    pre_computable: bool = False

    def __init__(
        self,
        analog_system: AnalogSystem,
        modulation_frequency: float,
    ):
        super().__init__(
            analog_system.A,
            analog_system.B,
            analog_system.CT,
            analog_system.Gamma,
            analog_system.Gamma_tildeT,
            analog_system.D,
            analog_system.B_tilde,
            analog_system.A_tilde,
        )
        self.angular_modulation_frequency = 2 * np.pi * modulation_frequency

        if self.N % 2 != 0:
            raise ValueError("A must be even sized.")
        self._N_half = self.N // 2
        self._x_shuffler = np.zeros(self.N, dtype=int)
        _indices = np.arange(self._x_shuffler.size)
        self._x_shuffler[::2] = _indices[: self._N_half]
        self._x_shuffler[1::2] = _indices[self._N_half :]
        self._x_de_shuffler = np.zeros_like(self._x_shuffler)
        self._x_de_shuffler[self._x_shuffler] = _indices

        self._M_half = self.M // 2
        self._s_shuffler = np.zeros(self.M, dtype=int)
        _indices = np.arange(self._s_shuffler.size)
        self._s_shuffler[::2] = _indices[: self._M_half]
        self._s_shuffler[1::2] = _indices[self._M_half :]
        self._s_de_shuffler = np.zeros_like(self._s_shuffler)
        self._s_de_shuffler[self._s_shuffler] = _indices

        self._rotation_matrix_data = np.eye(self.N)

    def derivative(
        self, x: np.ndarray, t: float, u: np.ndarray, s: np.ndarray
    ) -> np.ndarray:
        """Compute the derivative of the analog system.

        Specifically, produces the state derivative

        :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`

        as a function of the state vector :math:`\mathbf{x}(t)`, the given time
        :math:`t`, the input signal value :math:`\mathbf{u}(t)`, and the
        control contribution value :math:`\mathbf{s}(t)`.

        Parameters
        ----------
        x : `array_like`, shape=(N,)
            the state vector evaluated at time t.
        t : `float`
            the time t.
        u : `array_like`, shape=(L,)
            the input signal vector evaluated at time t.
        s : `array_like`, shape=(M,)
            the control contribution evaluated at time t.

        Returns
        -------
        `array_like`, shape=(N,)
            the derivative :math:`\dot{\mathbf{x}}(t)`.
        """
        return (
            np.dot(self.A, x).flatten()
            + np.dot(self.B, u).flatten()
            + np.dot(
                self.Gamma,
                np.dot(self.modulate(t)[: self.M, : self.M], s[self._s_shuffler])[
                    self._s_de_shuffler
                ],
            ).flatten()
        )

    def control_observation(
        self, t: float, x: np.ndarray, u: np.ndarray = None, s: np.ndarray = None
    ) -> np.ndarray:
        """Computes the control observation for a given state vector :math:`\mathbf{x}(t)`
        evaluated at time :math:`t`.

        Specifically, returns

        :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t) + \\tilde{\mathbf{D}} \mathbf{u}(t)`

        Parameters
        ----------
        x : `array_like`, shape=(N,)
            the state vector.
        u : `array_like`, shape=(L,)
            the input vector
        s : `array_like`, shape=(M,)
            the control signal
        Returns
        -------
        `array_like`, shape=(M_tilde,)
            the control observation.

        """
        x = np.dot(self.demodulate(t), x[self._x_shuffler])[
            self._x_de_shuffler
        ].flatten()
        if u is None:
            return np.dot(self.Gamma_tildeT, x)
        if s is None:
            np.dot(self.Gamma_tildeT, x) + np.dot(self.B_tilde, u)
        return (
            np.dot(self.Gamma_tildeT, x)
            + np.dot(self.B_tilde, u)
            + np.dot(self.A_tilde, s)
        )

    def modulate(self, t: float) -> np.ndarray:
        """Upmodulate the given signal phi."""
        return self._rotation_matrix(self.angular_modulation_frequency * t)

    def demodulate(self, t: float) -> np.ndarray:
        """Downmodulate the given signal phi."""
        return self._rotation_matrix(-self.angular_modulation_frequency * t)

    def _rotation_matrix(self, phi: float) -> np.ndarray:
        self._rotation_matrix_data[: self.N, : self.N] = np.eye(self.N) * np.cos(phi)
        self._rotation_matrix_data[: self._N_half, self._N_half :] = -np.eye(
            self._N_half
        ) * np.sin(phi)
        self._rotation_matrix_data[self._N_half :, : self._N_half] = np.eye(
            self._N_half
        ) * np.sin(phi)
        return self._rotation_matrix_data


class SquareWaveModulator(SineWaveModulator):
    """Square-wave modulator

    This class represents a sinewave modulator with a given modulation
    frequency and permutation matrix.

    Parameters
    ----------
    modulation_frequency : `float`
        the modulation frequency
    permuation_matrix : `array_like`, shape=(N, N)
        the permutation matrix

    """

    pre_computable: bool = False

    def __init__(
        self,
        analog_system: AnalogSystem,
        modulation_frequency: float,
        permuation_matrix: np.ndarray,
        base_band_bandwidth: float,
    ):
        self.omega_q = 2 * np.pi * base_band_bandwidth
        A = np.zeros((2 * analog_system.N, 2 * analog_system.N))
        A[: analog_system.N, : analog_system.N] = analog_system.A
        A[analog_system.N :, analog_system.N :] = -self.omega_q * np.eye(
            analog_system.N
        )
        B = np.vstack((analog_system.B, np.zeros_like(analog_system.B)))
        CT = np.hstack((analog_system.CT, np.zeros_like(analog_system.CT)))
        Gamma = np.vstack((analog_system.Gamma, np.zeros_like(analog_system.Gamma)))
        Gamma_tildeT = np.hstack(
            (np.zeros_like(analog_system.Gamma_tildeT), analog_system.Gamma_tildeT)
        )
        super().__init__(
            AnalogSystem(
                A,
                B,
                CT,
                Gamma,
                Gamma_tildeT,
            ),
            modulation_frequency,
            permuation_matrix,
        )
        self._square_rotation = np.zeros((2, 2))
        self._full_rotation_matrix = np.zeros((self.N // 2, self.N // 2))

    def _rotation_matrix(self, phi: float) -> np.ndarray:
        cos_wave = 2 * (np.cos(phi) > 0) - 1
        sin_wave = 2 * (np.sin(phi) > 0) - 1
        self._square_rotation[0, 0] = cos_wave
        self._square_rotation[1, 1] = cos_wave
        self._square_rotation[0, 1] = -sin_wave
        self._square_rotation[1, 0] = sin_wave
        for n in range(self.N // 4):
            self._full_rotation_matrix[
                2 * n : 2 * (n + 1), 2 * n : 2 * (n + 1)
            ] = self._square_rotation
        return self._full_rotation_matrix

    def _rotation_matrix_half_clock_cycle(self, phi: float) -> np.ndarray:
        cos_wave = int(np.cos(phi) > 0)
        sin_wave = int(np.sin(phi) > 0)
        self._square_rotation[0, 0] = cos_wave
        self._square_rotation[1, 1] = cos_wave
        self._square_rotation[0, 1] = -sin_wave
        self._square_rotation[1, 0] = sin_wave
        for n in range(self.N // 4):
            self._full_rotation_matrix[
                2 * n : 2 * (n + 1), 2 * n : 2 * (n + 1)
            ] = self._square_rotation
        return self._full_rotation_matrix

    def demodulate(self, t: float) -> np.ndarray:
        """Downmodulate the given signal phi."""
        return np.dot(
            self.inverted_permuation_matrix,
            np.dot(
                self._rotation_matrix_half_clock_cycle(
                    -2 * np.pi * self.angular_modulation_frequency * t
                ),
                self._permuation_matrix,
            ),
        )

    def derivative(
        self, x: np.ndarray, t: float, u: np.ndarray, s: np.ndarray
    ) -> np.ndarray:
        """Compute the derivative of the analog system.

        Specifically, produces the state derivative

        :math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`

        as a function of the state vector :math:`\mathbf{x}(t)`, the given time
        :math:`t`, the input signal value :math:`\mathbf{u}(t)`, and the
        control contribution value :math:`\mathbf{s}(t)`.

        Parameters
        ----------
        x : `array_like`, shape=(N,)
            the state vector evaluated at time t.
        t : `float`
            the time t.
        u : `array_like`, shape=(L,)
            the input signal vector evaluated at time t.
        s : `array_like`, shape=(M,)
            the control contribution evaluated at time t.

        Returns
        -------
        `array_like`, shape=(N,)
            the derivative :math:`\dot{\mathbf{x}}(t)`.
        """
        N_2 = self.N // 2
        self.A[N_2:, :N_2] = self.omega_q * self.demodulate(t)
        return (
            np.dot(self.A, x).flatten()
            + np.dot(self.B, u).flatten()
            + np.dot(self.Gamma, np.dot(self.modulate(t), s)).flatten()
        )

    def control_observation(
        self, t: float, x: np.ndarray, u: np.ndarray = None, s: np.ndarray = None
    ) -> np.ndarray:
        """Computes the control observation for a given state vector :math:`\mathbf{x}(t)`
        evaluated at time :math:`t`.

        Specifically, returns

        :math:`\\tilde{\mathbf{s}}(t) = \\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t) + \\tilde{\mathbf{D}} \mathbf{u}(t)`

        Parameters
        ----------
        x : `array_like`, shape=(N,)
            the state vector.
        u : `array_like`, shape=(L,)
            the input vector
        s : `array_like`, shape=(M,)
            the control signal
        Returns
        -------
        `array_like`, shape=(M_tilde,)
            the control observation.

        """
        if u is None:
            return np.dot(self.Gamma_tildeT, x)
        if s is None:
            np.dot(self.Gamma_tildeT, x) + np.dot(self.B_tilde, u)
        return (
            np.dot(self.Gamma_tildeT, x)
            + np.dot(self.B_tilde, u)
            + np.dot(self.A_tilde, s)
        )
