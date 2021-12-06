from typing import Iterator
import cbadc
import logging
import scipy.linalg
import scipy.integrate
import numpy as np

logger = logging.getLogger(__name__)


class NUVEstimator:
    """The NUV batch estimator

    The NUV estimator iteratively estimates estimates a filtered version
    :math:`\hat{\mathbf{u}}(t)` of the input signal :math:`\mathbf{u}(t)` from
    a sequence of control signals :math:`\mathbf{s}[k]`.

    In comparision to the linear complexity estimators

    - :py:class:`cbadc.digital_estimator.DigitalEstimator`
    - :py:class:`cbadc.digital_estimator.ParallelEstimator`
    - :py:class:`cbadc.digital_estimator.IIRFilter`
    - :py:class:`cbadc.digital_estimator.FIRFilter`

    the NUV estimator enforces the bounded outputs in an iterative
    scheme as described in

    * `R. Keusch and H.-A. Loeliger, A Binarizing NUV Prior and its Use for M-Level Control and Digital-to-Analog Conversion. <https://arxiv.org/pdf/2105.02599.pdf>`_

    Parameters
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system (necessary to compute the estimators filter
        coefficients).
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        a digital control (necessary to determine the corresponding DAC
        waveform).
    covU: `float`
        a prior covariance matrix for the sought estimate.
    bound_y: `float`
        the bound enforced on the outputs :math:`\mathbf{y}(t)` of the
        system.
    gamma: `float`
        a scale factor.
    K1 : `int`
        batch size.
    K2 : `int`, `optional`
        lookahead size, defaults to 0.
    iterations_per_batch: `int`, `optional`
        number of iteration steps per batch, defaults to 100.
    Ts: `float`, `optional`
        the sampling time, defaults to the time period of the digital control.
    oversample: `int`, `optional`
        add observations per control signal, defaults to 1, i.e., no oversampling.

    Attributes
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        analog system as in :py:class:`cbadc.analog_system.AnalogSystem` or
        from derived class.
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        digital control as in :py:class:`cbadc.digital_control.DigitalControl`
        or from derived class.
    K1: `int`
        number of samples per estimate batch.
    K2: `int`
        number of lookahead samples per computed batch.
    covU: `float`
        the prior covariance matrix (or scalar in the single input case)
        of the estimate :math:`\hat{\mathbf{u}}(t)`.
    bound_y: `float`
        the bound enforced on the outputs :math:`\mathbf{y}(t)` of the
        system.
    Ts: `float`
        the sample rate of the estimates
    iterations_per_batch: `int`
        number of iteration steps per batch.
    gamma: `float`
        a scale factor.
    """

    def __init__(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
        covU: float,
        bound_y: float,
        gamma: float,
        K1: int,
        K2: int = 0,
        iterations_per_batch: int = 100,
        Ts: float = None,
        oversample: int = 1,
    ):
        # Check inputs
        if K1 < 1:
            raise BaseException("K1 must be a positive integer.")
        self.K1 = K1
        if K2 < 0:
            raise BaseException("K2 must be a non negative integer.")
        self.K2 = K2
        self.K3 = K1 + K2
        self.analog_system = analog_system
        self.covU = covU
        self.bound_y = bound_y
        if not np.allclose(self.analog_system.D, np.zeros_like(self.analog_system.D)):
            raise BaseException(
                """Can't compute filter coefficients for system with non-zero
                D matrix. Consider chaining for removing D"""
            )

        self.digital_control = digital_control
        if Ts:
            self.Ts = Ts / oversample
        else:
            self.Ts = digital_control.clock.T / oversample
        self.control_signal = None
        self._stop_iteration = False
        self._estimate_pointer = self.K1

        self.iterations_per_batch = iterations_per_batch
        self.gamma = gamma
        # Initialize filters
        self._compute_filter_coefficients(analog_system, digital_control)
        self._allocate_memory_buffers()

        # Set initial Covariances
        self._forward_CoVariance[0, :, :] = 1e12 * np.eye(self.analog_system.N)
        self._oversample = oversample

    def _allocate_memory_buffers(self):
        # Allocate memory buffers
        self._control_signal = np.zeros((self.K3, self.analog_system.M), dtype=np.int8)
        self._estimate = np.zeros((self.K1, self.analog_system.L), dtype=np.double)
        self._control_signal_in_buffer = 0
        self._forward_mean = np.zeros((self.K3, self.analog_system.N), dtype=np.double)
        self._forward_CoVariance = np.zeros(
            (self.K3, self.analog_system.N, self.analog_system.N), dtype=np.double
        )
        self._G = np.zeros(
            (self.K3, self.analog_system.N_tilde, self.analog_system.N_tilde),
            dtype=np.double,
        )
        self._F = np.zeros(
            (self.K3, self.analog_system.N, self.analog_system.N), dtype=np.double
        )

        self._y_mean = np.zeros((self.K3, self.analog_system.N_tilde))
        self._y_CoVariance = np.zeros(
            (self.K3, self.analog_system.N_tilde, self.analog_system.N_tilde)
        )
        # self._W_tilde = np.zeros(
        #     (
        #         self.K3,
        #         self.analog_system.N,
        #         self.analog_system.N
        #     ),
        #     dtype=np.double
        # )
        self._xi_tilde = np.zeros((self.K3 + 1, self.analog_system.N), dtype=np.double)
        self._sigma_squared_1 = np.ones(
            (
                self.K3,
                self.analog_system.N_tilde,
            ),
            dtype=np.double,
        )
        self._sigma_squared_2 = np.ones(
            (
                self.K3,
                self.analog_system.N_tilde,
            ),
            dtype=np.double,
        )
        self._posterior_observation_mean = np.zeros(
            (self.K3, self.analog_system.N_tilde), dtype=np.double
        )

    def set_iterator(self, control_signal_sequence: Iterator[np.ndarray]):
        """Set iterator of control signals

        Parameters
        -----------
        control_signal_sequence : iterator
            a iterator which outputs a sequence of control signals.
        """
        self.control_signal = control_signal_sequence

    def __call__(self, control_signal_sequence: Iterator[np.ndarray]):
        return self.set_iterator(control_signal_sequence)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        # Check if control signal iterator is set.
        if self.control_signal is None:
            raise BaseException("No iterator set.")

        # Check if there are estimates in the estimate buffer
        if self._estimate_pointer < self.K1:
            temp = np.array(self._estimate[self._estimate_pointer, :], dtype=np.double)
            self._estimate_pointer += 1
            return temp

        # Check if stop iteration has been raised in previous batch
        if self._stop_iteration:
            logger.warning("Warning: StopIteration received by estimator.")
            raise StopIteration
        # Otherwise start receiving control signals
        full = False

        # Fill up batch with new control signals.
        while not full:
            # next(self.control_signal) calls the control signal
            # iterator and thus recives new control
            # signal samples
            try:
                control_signal_sample = next(self.control_signal)
            except RuntimeError:
                self._stop_iteration = True
                control_signal_sample = np.zeros((self.analog_system.M), dtype=np.int8)
            for _ in range(self._oversample):
                full = self._input(control_signal_sample)

        # Compute new batch of K1 estimates
        self._reset_between_batch()
        self._compute_batch()
        # adjust pointer to indicate that estimate buffer
        # is non empty
        self._estimate_pointer -= self.K1

        # recursively call itself to return new estimate
        return self.__next__()

    def _input(self, s: np.ndarray) -> bool:
        if self._control_signal_in_buffer == (self.K3):
            raise BaseException(
                """Input buffer full. You must compute batch before adding
                more control signals"""
            )
        for _ in range(self.analog_system.M):
            self._control_signal[self._control_signal_in_buffer, :] = np.asarray(
                2 * s - 1, dtype=np.int8
            )
        self._control_signal_in_buffer += 1
        return self._control_signal_in_buffer > (self.K3 - 1)

    def _compute_filter_coefficients(
        self,
        analog_system: cbadc.analog_system.AnalogSystem,
        digital_control: cbadc.digital_control.DigitalControl,
    ):
        logger.info("Compute filter coefficients.")
        # Compute filter coefficients
        self.Af: np.ndarray = np.asarray(scipy.linalg.expm(analog_system.A * self.Ts))
        Gamma = np.array(analog_system.Gamma)
        # Solve IVPs
        self.Bf: np.ndarray = np.zeros((self.analog_system.N, self.analog_system.M))
        atol = 1e-200
        rtol = 1e-12
        max_step = self.Ts / 1000.0

        for m in range(self.analog_system.M):

            def _derivative_forward_2(t, x):
                return np.dot(analog_system.A, x) + np.dot(
                    Gamma, digital_control.impulse_response(m, t)
                )

            solBf = scipy.integrate.solve_ivp(
                _derivative_forward_2,
                (0, self.Ts),
                np.zeros(self.analog_system.N),
                atol=atol,
                rtol=rtol,
                max_step=max_step,
                method="Radau",
            ).y[:, -1]
            self.Bf[:, m] = solBf

        BBT = np.dot(analog_system.B, np.dot(self.covU, analog_system.B.transpose()))

        def _derivative_input(t, x):
            t_minus_tau = self.Ts - t
            A_t_minus_tau = analog_system.A * t_minus_tau
            return np.dot(
                scipy.linalg.expm(A_t_minus_tau),
                np.dot(BBT, scipy.linalg.expm(A_t_minus_tau.transpose())),
            ).flatten()

        self.Vu = np.dot(
            self.covU,
            scipy.integrate.solve_ivp(
                _derivative_input,
                (0, self.Ts),
                np.zeros(self.analog_system.N ** 2),
                atol=atol,
                rtol=rtol,
                max_step=max_step,
                method="Radau",
            )
            .y[:, -1]
            .reshape((self.analog_system.N, self.analog_system.N)),
        )

    def _compute_batch(self):
        logger.info("Computing batch.")
        # check if ready to compute buffer
        if self._control_signal_in_buffer < self.K3:
            raise BaseException("Control signal buffer not full")

        self._update_observation_statistics()
        self._MBF()
        for _ in range(self.iterations_per_batch):
            self._update_observation_bound_variances()
            self._update_observation_statistics()
            self._MBF()

        self._input_estimation()

        # Save forward mean and covariance for next batch
        self._forward_mean[0, :] = self._forward_mean[self.K1 - 1, :]
        # Compute posterior mean
        # self._forward_mean[0, :] = self._forward_mean[self.K1 - 1] - np.dot(
        #     self._forward_CoVariance[self.K1 - 1, :, :],
        #     self._xi_tilde[self.K1 - 1, :]
        # )
        self._forward_CoVariance[0, :, :] = self._forward_CoVariance[self.K1 - 1, :, :]

        # Compute posterior covariance
        # self._forward_CoVariance[0, :, :] = self._forward_CoVariance[self.K1 - 1, :, :] - np.dot(
        #     self._forward_CoVariance[self.K1 - 1, :, :],
        #     np.dot(
        #         self._W_tilde[self.K1 - 1, :, :],
        #         self._forward_CoVariance[self.K1 - 1, :, :]
        #     )
        # )

        self._sigma_squared_1[0, :] = self._sigma_squared_1[self.K1 - 1, :]
        self._sigma_squared_2[0, :] = self._sigma_squared_2[self.K1 - 1, :]

        # rotate buffer to make place for new control signals
        self._control_signal = np.roll(self._control_signal, -self.K1, axis=0)
        self._control_signal_in_buffer -= self.K1

    def _MBF(self):
        # Forward pass
        temp_K3 = self.K3 - 1
        for k in range(self.K3):
            self._G[k, :, :] = np.linalg.inv(
                self._y_CoVariance[k, :, :]
                + np.dot(
                    self.analog_system.CT,
                    np.dot(
                        self._forward_CoVariance[k, :, :],
                        self.analog_system.CT.transpose(),
                    ),
                )
            )
            self._F[k, :, :] = np.eye(self.analog_system.N) - np.dot(
                self._forward_CoVariance[k, :, :],
                np.dot(
                    self.analog_system.CT.transpose(),
                    np.dot(self._G[k, :, :], self.analog_system.CT),
                ),
            )

            if k < temp_K3:
                self._forward_mean[k + 1, :] = (
                    np.dot(
                        self.Af,
                        np.dot(self._F[k, :, :], self._forward_mean[k, :])
                        + np.dot(
                            self._forward_CoVariance[k, :, :],
                            np.dot(
                                self.analog_system.CT.transpose(),
                                np.dot(self._G[k, :, :], self._y_mean[k, :]),
                            ),
                        ),
                    )
                    + np.dot(self.Bf, self._control_signal[k, :])
                )

                self._forward_CoVariance[k + 1, :, :] = (
                    np.dot(
                        self.Af,
                        np.dot(
                            np.dot(self._F[k, :, :], self._forward_CoVariance[k, :, :]),
                            self.Af.transpose(),
                        ),
                    )
                    + self.Vu
                )

        # Backward pass
        for k in range(self.K3 - 1, -1, -1):

            if k < temp_K3:
                xi_tilde_z = np.dot(self.Af.transpose(), self._xi_tilde[k + 1, :])
                # W_tilde_z = np.dot(
                #     self.Af.transpose(),
                #     np.dot(
                #         self._W_tilde[k+1, :, :],
                #         self.Af
                #     )
                # )
            else:
                # Initialize with zero vector.
                xi_tilde_z = np.zeros(self.analog_system.N)
                # W_tilde_z = np.zeros(
                #     (self.analog_system.N, self.analog_system.N))

            # self._W_tilde[k, :, :] = np.dot(
            #     self._F[k, :, :],
            #     np.dot(
            #         W_tilde_z,
            #         self._F[k, :, :]
            #     )
            # ) + np.dot(
            #     self.analog_system.CT.transpose(),
            #     np.dot(
            #         self._G[k, :, :],
            #         self.analog_system.CT
            #     )
            # )

            temp = (
                np.dot(self.analog_system.CT.transpose(), self._forward_mean[k, :])
                - self._y_mean[k, :]
            )

            self._xi_tilde[k, :] = np.dot(
                self._F[k, :, :].transpose(), xi_tilde_z
            ) + np.dot(
                self.analog_system.CT.transpose(), np.dot(self._G[k, :, :], temp)
            )
            # if (k < 5):
            #     print(k,  self._xi_tilde[k, :])

    def _input_estimation(self):
        for k in range(self.K1):
            self._estimate[k, :] = -np.dot(
                self.covU,
                np.dot(self.analog_system.B.transpose(), self._xi_tilde[k, :]),
            )

    def _update_observation_bound_variances(self):
        for k in range(self.K3):
            self._posterior_observation_mean[k, :] = np.dot(
                self.analog_system.CT,
                self._forward_mean[k]
                - np.dot(self._forward_CoVariance[k, :, :], self._xi_tilde[k, :]),
            )
            # posterior_CoVariance = np.dot(
            #     self.analog_system.CT,
            #     np.dot(
            #         self._forward_CoVariance[k, :, :] - np.dot(
            #             self._forward_CoVariance[k, :, :],
            #             np.dot(
            #                 self._W_tilde[k, :, :],
            #                 self._forward_CoVariance[k, :, :]
            #             )
            #         ),
            #         self.analog_system.CT.transpose()
            #     )
            # )
            min_sigma_squared = 1e-100
            for ell in range(self.analog_system.N_tilde):
                self._sigma_squared_1[k, ell] = max(
                    np.abs(self._posterior_observation_mean[k, ell] - self.bound_y)
                    / self.gamma,
                    min_sigma_squared,
                )
                self._sigma_squared_2[k, ell] = max(
                    np.abs(self._posterior_observation_mean[k, ell] + self.bound_y)
                    / self.gamma,
                    min_sigma_squared,
                )

    def _update_observation_statistics(self):
        for k in range(self.K3):
            W1_back = np.diag(1.0 / self._sigma_squared_1[k, :])
            W2_back = np.diag(1.0 / self._sigma_squared_2[k, :])

            W_back = W1_back + W2_back
            Wm_back = self.bound_y * np.sum(W1_back, axis=-1) - self.bound_y * np.sum(
                W2_back, axis=-1
            )

            self._y_CoVariance[k, :, :] = np.linalg.inv(W_back)

            self._y_mean[k, :] = np.dot(self._y_CoVariance[k, :, :], Wm_back)

    def _reset_between_batch(self):
        self._sigma_squared_1[1:, :] = np.ones(
            (self.K3 - 1, self.analog_system.N_tilde), dtype=np.double
        )
        self._sigma_squared_2[1:, :] = np.ones(
            (self.K3 - 1, self.analog_system.N_tilde), dtype=np.double
        )

    def __str__(self):
        return f"""NUVEstimator is parameterized as
        \nobservation bound = {self.bound_y:.2f},
        \ncovU = {self.covU},
        \ngamma = {self.gamma},
        \nTs = {self.Ts},\nK1 = {self.K1},\nK2 = {self.K2},
        \nand\nnumber of iterations per batch = {self.iterations_per_batch}
        \nResulting in the filter coefficients\nAf = \n{self.Af},
        \nBf = \n{self.Bf},
        \nand Vu = \n{self.Vu}.
        """
