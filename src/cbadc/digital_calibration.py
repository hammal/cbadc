"""digital estimator calibration.
"""
from typing import Callable, Union
import numpy as np
import logging

import matplotlib.pyplot as plt
from .digital_estimator import AdaptiveFilter

from .simulator.numerical_simulator import (
    FullSimulator,
    PreComputedControlSignalsSimulator,
)
from .simulator.numpy_simulator import NumpySimulator
from .utilities import show_status

logger = logging.getLogger(__name__)

Simulators = Union[
    FullSimulator,
    PreComputedControlSignalsSimulator,
    NumpySimulator,
]


class Calibration:
    """A framework for training an adaptive filter

    Parameters
    ----------
    filter: :py:class:`cbadc.digital_estimator.AdaptiveFilter`
        the adaptive filter to be calibrated
    training_simulator: :py:class:`cbadc.simulator.FullSimulator`
        an iterator capable of producing training control signals (no input signal).
    testing_simulator: :py:class:`cbadc.simulator.FullSimulator`
        an iterator of control signals for evaluating the resulting filter on.
    """

    def __init__(
        self,
        filter: AdaptiveFilter,
        training_simulator: Simulators,
        testing_simulator: Simulators,
    ):
        self.filter = filter
        self.training_simulator = training_simulator
        self.testing_simulator = testing_simulator
        self.batch_error = []
        self.total_number_of_data_points = 0

    def train(
        self,
        epochs: int,
        step_size: Callable[[int], float],
        batch_size: int,
        stochastic_delay=0,
        method="sgd",
        **kwargs,
    ):
        """train the adaptive filter on the training data

        Parameters
        ----------
        epochs: `int`
            number of batches or iterations to train over.
        step_size: Callable[[int], float]
            a function from iteration index to a scalar step size.
        batch_size: `int`
            the size of batches to be trained (passed to the gradient method)
        stochastic_delay: `int`, `optional`
            a number determining a uniformly random delay, in the
            interval [0, stochastic_delay), between data points used
            in the gradient descent. Defaults to 0. Note that this only
            applies the first time a gradient method is run if the data is buffered.
        method: 'str'
            a string determining if the 'sgd' or 'adadelta' gradient method or `rls` (recursive least squares)
            method should be used, defaults to 'sgd'.
        epsilon: `float`, `optional`
            a lower bound on step size, defaults to 1e-12.
        gamma: `float`, `optional`
            related to momentum, defaults to 0.99 .
        """
        self.training_simulator.reset()
        self.filter(self.training_simulator)
        self.filter.warm_up(self.filter.K3)
        data_points = epochs * batch_size
        logger.info(f"Starting training round for {data_points} additional data points")
        for index in show_status(range(epochs)):
            if method == "sgd":
                self.batch_error.append(
                    np.average(
                        np.array(
                            self.filter.stochastic_gradient_decent(
                                step_size(index + self.total_number_of_data_points),
                                batch_size,
                                stochastic_delay=stochastic_delay,
                            )
                        )
                    )
                )
            elif method == "adadelta":
                epsilon = kwargs.get("epsilon", 1e-12)
                gamma = kwargs.get("gamma", 1 - 1e-2)
                self.batch_error.append(
                    np.average(
                        np.array(
                            self.filter.adadelta(
                                batch_size=batch_size,
                                epsilon=epsilon,
                                gamma=gamma,
                                stochastic_delay=stochastic_delay,
                            )
                        )
                    )
                )
            elif method == "rls":
                self.batch_error.append(
                    np.average(
                        np.array(
                            self.filter.recursive_least_squares(
                                batch_size=batch_size,
                                forgetting_factor=kwargs.get(
                                    "forgetting_factor",
                                    kwargs.get("forgetting_factor", 1e0 - 1e-9),
                                ),
                                stochastic_delay=stochastic_delay,
                            )
                        )
                    )
                )
            else:
                raise Exception(f"gradient method: {method} not supported.")

        self.total_number_of_data_points += data_points

    def test(self, size: int):
        """test estimate

        returns a size length numpy array with
        input estimates from the test_simulator dataset.

        Parameters
        ----------
        size: `int`
            number of estimates to be computed.
        """
        self.testing_simulator.reset()
        self.filter(self.testing_simulator)
        # self.filter.warm_up(self.filter.K3)
        uncut_size = size + self.filter.K3
        u_hat = np.zeros(uncut_size)
        logger.info(f"Testing {size} number of samples")
        for index in show_status(range(uncut_size)):
            u_hat[index] = next(self.filter)
        return u_hat[self.filter.K3 :]

    def plot_test_accuracy(self):
        """A utility function to plot the training error decay"""
        temp = np.array(self.batch_error)
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
        ax[1].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
        ax[0].plot(temp)
        ax[1].semilogy(np.abs(temp))
        ax[0].set_ylabel("error")
        ax[1].set_ylabel("|error|")
        ax[1].set_xlabel("batch iteration")
        plt.suptitle("Training error")
        fig.tight_layout()

    def stats(self) -> str:
        """return a summary statement of the training progress."""
        return f"""Number of training samples: {self.total_number_of_data_points}
        Number of filter coefficients: {np.sum(self.filter.h != 0)}
        Largets / smallest training error: {np.max(np.abs(self.batch_error)):0.1e} / {np.min(np.abs(self.batch_error)):0.1e}
        Number of gradient evaluations: {self.filter._number_of_gradient_evaluations}
        Calibration Time: {self.filter.digital_control.clock.T * self.total_number_of_data_points:0.1e} [s]
        """

    def compute_step_size_template(self, averaging_window_size=30):
        """force the adaptive filter to recompute the step size profile."""
        self.filter.compute_step_size_template(averaging_window_size)
