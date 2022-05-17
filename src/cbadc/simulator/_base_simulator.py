import logging
import cbadc.analog_system
import cbadc.digital_control
import cbadc.analog_signal
import numpy as np
import math
import scipy.integrate
import scipy.linalg
from typing import Iterator, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class _BaseSimulator(Iterator[np.ndarray]):
    """Simulate the analog system and digital control interactions
    in the presence on analog signals.

    Parameters
    ----------
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        the analog system
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        the digital control
    input_signals : [:py:class:`cbadc.analog_signal.AnalogSignal`]
        a python list of analog signals (or a derived class)
    clock: :py:class:`cbadc.simulator.clock`, `optional`
        a clock to syncronize simulator output against, defaults to
        a phase delayed version of the digital_control clock.
    t_stop : `float`, optional
        determines a stop time, defaults to :py:obj:`math.inf`
    initial_state_vector: `array_like`, shape=(N), `optional`
        initial state vector.
    cov_x: `array_like`, shape=(N, N)
        the covariance matrix of white noise contributions, `optional`
        defaults to all zero. The variance magnitude corresponds to
        V^2 in the case that the states are represented in volts.

    Attributes
    ----------
    analog_system : :py:class:`cbadc.analog_system.AnalogSystem`
        the analog system being simulated.
    digital_control : :py:class:`cbadc.digital_control.DigitalControl`
        the digital control being simulated.
    t : `float`
        current time of simulator.
    clock: `float`
        a clock to which the outputs of the simulator are synchronized.
    t_stop : `float`
        end time at which the generator raises :py:class:`StopIteration`.
    initial_state_vector: `array_like`
        the initial state of the simulator.

    Yields
    ------
    `array_like`, shape=(M,)

    """

    def __init__(
        self,
        analog_system: cbadc.analog_system._valid_analog_system_types,
        digital_control: cbadc.digital_control._valid_digital_control_types,
        input_signal: List[cbadc.analog_signal._AnalogSignal],
        clock: cbadc.analog_signal._valid_clock_types = None,
        t_stop: float = math.inf,
        initial_state_vector=None,
        cov_x: np.ndarray = None,
    ):
        if analog_system.L != len(input_signal):
            raise Exception(
                """The analog system does not have as many inputs as in input
            list"""
            )
        # if not np.allclose(analog_system.D, np.zeros_like(analog_system.D)):
        #     raise Exception(
        #         """Can't simulate system with non-zero
        #         D matrix. Consider chaining systems to remove D."""
        #     )
        self.analog_system = analog_system
        self.digital_control = digital_control
        self.input_signals = input_signal
        self.t: float = 0.0
        self.t_stop = t_stop
        if isinstance(clock, cbadc.analog_signal.Clock):
            self.clock = clock
            if self.clock.T != self.digital_control.clock.T:
                if self.clock.T > self.digital_control.clock.max_step():
                    logger.critical(
                        f"Ts={self.clock.T} is larger than the digital control's smallest allowed step. This might lead to missing control updates."
                    )
        else:
            # Default is to delay readout until negative edge of clock
            logger.info("No clock specified. Deriving one from digital control.")
            self.clock = cbadc.analog_signal.clock.delay_clock_by_duty_cycle(
                self.digital_control.clock
            )

        if initial_state_vector is not None:
            self._state_vector = np.array(initial_state_vector, dtype=np.float64)
            logger.debug(f"initial state vector: {self._state_vector}")
            if (
                self._state_vector.size != self.analog_system.N
                or len(self._state_vector.shape) > 1
            ):
                raise Exception("initial_state_vector not single dimension of length N")
        else:
            self._state_vector = np.zeros(self.analog_system.N, dtype=np.double)
        self._res = np.zeros(self.analog_system.N, dtype=np.double)
        self.noise = False
        if not (cov_x is None):
            self.set_covariance_matrix(cov_x)

    def set_covariance_matrix(self, cov_x: np.ndarray):
        """Introduce a i.i.d. white noise process
        with covariance matrix cov_x

        Parameters
        ----------
        cov_x: array_like, shape=(N, N)
            the covariance matrix
        """
        self._compute_noise_covariance(cov_x)
        self.noise = True

    def _compute_noise_covariance(self, cov_x):
        noise_covariance_per_unit_time = cov_x / self.digital_control.clock.T

        def derivative(t: float, y: np.ndarray):
            A_exp = np.asarray(scipy.linalg.expm(np.asarray(self.analog_system.A) * t))
            return np.dot(
                A_exp, np.dot(noise_covariance_per_unit_time, A_exp.transpose())
            ).flatten(order="C")

        sol = scipy.integrate.solve_ivp(
            derivative,
            (0, self.digital_control.clock.T),
            np.zeros(self.analog_system.N**2),
        )
        self.covariance_matrix = np.array(sol.y[:, -1]).reshape(
            (self.analog_system.N, self.analog_system.N), order="C"
        )
        self._cholesky_covariance_matrix = np.linalg.cholesky(self.covariance_matrix)

    def _noise_sample(self):
        return np.dot(
            self._cholesky_covariance_matrix,
            np.random.normal(size=(self.analog_system.N)),
        )

    def reset(self, t: float = 0.0):
        """reset initial time of simulator and digital control"""
        self.t = t
        self.digital_control.reset(t)

    def state_vector(self) -> np.ndarray:
        """return current analog system state vector :math:`\mathbf{x}(t)`
        evaluated at time :math:`t`.

        Returns
        -------
        `array_like`, shape=(N,)
            returns the state vector :math:`\mathbf{x}(t)`
        """
        return self._state_vector[:]

    def observations(self) -> np.ndarray:
        """return the current analog system observations.

        in other words we return

        :math:`\mathbf{y}(t) = \mathbf{C}^{\mathsf{T}} \mathbf{x}(t)`

        Returns
        -------
        `array_like`, shape=(N_tilde,)
            returns the analog system observation :math:`\mathbf{y}(t)
        """
        return self.analog_system.signal_observation(self.state_vector())

    def __iter__(self):
        """Use simulator as an iterator"""
        return self

    def __next__(self) -> np.ndarray:
        """Computes the next control signal :math:`\mathbf{s}[k]`"""

        raise NotImplementedError

    def __str__(self) -> str:
        _input_signals_string = "\n\n\n".join(
            [str(input) for input in self.input_signals]
        )
        return f"""{80 * '='}

The Simulator is parameterized by the:

{80 * '-'}

Analog System:
{self.analog_system}

Digital Control:
{self.digital_control}

Input signals:
{_input_signals_string}

Clock:
{self.clock}

t_stop:
{self.t_stop}

{80 * '-'}

Currently the

state vector is:
{self.state_vector()}

t:
{self.t}

{80 * '-'}
        """
