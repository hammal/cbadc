"""Chain of integrators datasets.
"""
import numpy as np
import cbadc
import logging
logger = logging.getLogger(__name__)

# map (N, beta, rho, kappa, input_signal_type, amplitude, frequency, phase, offset) -> url_string
chain_of_integrators_pre_simulations = {
    (1, 6250.0, 0.0, 1.0, "sin", 1.0, 1.0, 0.0, 0.0): ['http:localhost.com']
}


class ChainOfIntegrators:
    """Control-bounded Chain-of-integrators ADC example.

    This class instantiates the chain-of-integrators control-bounded ADC
    which was frequently used in

    * `H. Malmberg, Control-bounded converters, Ph.D. dissertation, Dept. Inf. Technol. Elect. Eng., ETH Zurich, 2020.  <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf>`_
    * `H.-A. Loeliger, H. Malmberg, and G. Wilckens, Control-bounded analog-to-digital conversion: transfer function analysis, proof of concept, and digital filter implementation," arXiv:2001.05929 <https://arxiv.org/abs/2001.05929>`_
    * `H.-A. Loeliger and G. Wilckens, Control-based analog-to-digital conversion without sampling and quantization, 2015 Information Theory & Applications Workshop (ITA), UCSD, La Jolla, CA, USA, Feb. 1-6, 2015 <https://ieeexplore.ieee.org/document/7308975>`_
    * `H.-A. Loeliger, L. Bolliger, G. Wilckens, and J. Biveroni, Analog-to-digital conversion using unstable filters, 2011 Information Theory & Applications Workshop (ITA), UCSD, La Jolla, CA, USA, Feb. 6-11, 2011 <https://ieeexplore.ieee.org/abstract/document/5743620>`_

    Furthermore, many pre-computed control-signals can be conveniently
    be accessed through the :py:func:`cbadc.datasets.chain_of_integrators.ChainOfIntegrators.sin`
    and :py:func:`cbadc.datasets.chain_of_integrators.ChainOfIntegrators.ramp` methods.

    Parameters
    ----------
    N: `int`
        number of analog states, (for the chain-of-integrator this also 
        determines M=N), defaults to N=5.
    beta: `float`
        the integration slope or amplification, defaults to beta=6250.
    rho: `float`
        the local feedback term, defaults to rho=0.
    kappa: `float`
        the control gain, defaults to kappa=1
    """

    def __init__(self, N: int = 5, beta: float = 6250.0, rho: float = 0.0, kappa: float = 1.0) -> None:
        self.size = int(1e12)
        self.N = N
        self.beta = beta
        self.rho = rho
        self.kappa = kappa
        A = beta * np.eye(N, k=-1) + rho * np.eye(N)
        B = np.zeros((N, 1))
        B[0, 0] = beta
        CT = np.eye(N)
        Gamma = - kappa * beta * np.eye(N)
        Gamma_tilde = np.eye(N)
        self.M = self.N
        self.T = 1.0 / (2 * beta)
        self.analog_system = cbadc.analog_system.AnalogSystem(
            A, B, CT, Gamma, Gamma_tilde)
        self.digital_control = cbadc.digital_control.DigitalControl(
            self.T, self.N)

    def sin(self, amplitude: float, frequency: float, phase: float = 0.0, offset: float = 0.0):
        """Provide control signals and simulation settings for a sinusoidal input signal.

        Specifically for an input signal

        :math:`u(t) = \mathrm{amplitude} \cdot \sin(2 \pi \mathrm{frequency} t + \mathrm{phase}) + \mathrm{offset}`

        the resulting simulator and possibly pre-computed control-signals are
        retrived.

        Parameters
        ----------
        amplitude: `float`
        frequency: `float`
            specified in [Hz].
        phase: `float`
            defaults to phase=0.
        offset: `float`
            defaults to offset=0.

        Returns
        -------
        control_signal: Generator[np.ndarray, None, None]
            a control signal sequence, possibly retrived over http.
        simulator: :py:class:`cbadc.simulator.StateSpaceSimulator`
            an instantated simulator.
        size: `int`
            the maximum length of the simulation (1G control signal samples)
        """
        input_signal = cbadc.analog_signal.Sinusodial(
            amplitude, frequency, phase, offset)
        simulator = cbadc.simulator.StateSpaceSimulator(
            self.analog_system, self.digital_control, [input_signal])
        params = (self.N, self.beta, self.rho, self.kappa,
                  "sin", amplitude, frequency, phase, offset)
        if params in chain_of_integrators_pre_simulations:
            control_signal = cbadc.utilities.byte_stream_2_control_signal(
                cbadc.utilities.read_byte_stream_from_url(
                    chain_of_integrators_pre_simulations[params],
                    self.M), self.M)
        else:
            logger.warn(
                "No pre-computed simulation found. Iterating will invlove time consuming simulating.")
            control_signal = simulator
        return control_signal, simulator, self.size

    def ramp(self, amplitude: float, frequency: float, phase: float = 0.0, offset: float = 0.0):
        """Provide control signals and simulation settings for a ramp input signal.

        Specifically for an input signal

        :math:`u(t) = \mathrm{amplitude} \cdot \mathrm{ramp}(t / \mathrm{frequency} + \mathrm{phase}) + \mathrm{offset}`

        where :math:`\mathrm{ramp}(\cdot)` is an unit-scale ramp
        function.

        Parameters
        ----------
        amplitude: `float`
        frequency: `float`
            specified in [Hz].
        phase: `float`
            defaults to phase=0.
        offset: `float`
            defaults to offset=0.

        Returns
        -------
        control_signal: Generator[np.ndarray, None, None]
            a control signal sequence, possibly retrived over http.
        simulator: :py:class:`cbadc.simulator.StateSpaceSimulator`
            an instantated simulator.
        size: `int`
            the maximum length of the simulation (1G control signal samples)
        """
        input_signal = cbadc.analog_signal.Ramp(
            amplitude, frequency, phase, offset)
        simulator = cbadc.simulator.StateSpaceSimulator(
            self.analog_system, self.digital_control, [input_signal])
        params = (self.N, self.beta, self.rho, self.kappa,
                  "ramp", amplitude, frequency, phase, offset)
        if params in chain_of_integrators_pre_simulations:
            control_signal = cbadc.utilities.byte_stream_2_control_signal(
                cbadc.utilities.read_byte_stream_from_url(chain_of_integrators_pre_simulations[params], self.M), self.M)
        else:
            logger.warn(
                "No pre-computed simulation found. Iterating will invlove time consuming simulating.")
            control_signal = simulator
        return control_signal, simulator, self.size
