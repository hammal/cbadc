"""Data wrappers for control-bounded Hadamard ADC datasets.
"""
import numpy as np
import cbadc
import scipy.linalg


class HadamardPCB():
    """The Hadamard PCB...

    Parameters
    ----------
    pcb: `str`
        string determining prototype version, currently
        pcb = {'A', 'B'}, defaults to 'B'.
    """

    def __init__(self, pcb="B"):
        """Initialize a HadamardPCB Simulation for a given PCB version
        """
        self.r_1 = 1.62e3
        self.r_2 = 3.24e3
        self.r_3 = 10e3
        self.r_4 = np.inf

        self.v_ctrl = 2.5
        self.pcb = pcb
        if pcb == "B":
            self.T = 1e-6
            self.c_1 = 3.3e-9
        elif pcb == "A":
            self.T = 1.0/24e6
            self.c_1 = 120e-12
        else:
            raise BaseException("Specified PCB version does not supported.")
        # Hadamard matrix
        H_n = scipy.linalg.hadamard(4)
        buffer_loss = self.r_3 / (4*self.r_3 + self.r_1)
        A_chain_of_integrators = np.diag(
            np.ones(3), k=-1) - self.r_1 / self.r_4 * np.eye(4, k=0)
        A = 1/(self.r_1 * self.c_1) * np.matmul(-H_n,
                                                np.matmul(A_chain_of_integrators,
                                                          -H_n * buffer_loss))
        B = -1/(self.r_1 * self.c_1) * np.ones((4, 1))
        CT = np.eye(4)
        # assuming inputs +/-1 from the digital control
        Gamma = -self.v_ctrl/(self.r_2 * self.c_1) * \
            np.hstack((-H_n, np.eye(4)))
        # scaling does not matter, sign is positive since inverting comparators
        # and inverting flipflip cancel out
        Gamma_tildeT = np.vstack((-H_n, np.eye(4)))
        self.analog_system = cbadc.analog_system.AnalogSystem(
            A, B, CT, Gamma, Gamma_tildeT)
        self.digital_control = cbadc.digital_control.DigitalControl(self.T, 8)

    def simulation_ramp_1_B(self):
        """20 mHz period length, 4.5 Vpp ramp input simulation

        Prototype:

        - Version B
        - R1 = 1.62 kOhm
        - R2 = 3.24 kOhm
        - R3 = 10 kOhm
        - R5 = np.inf
        - T = 1 / (1 MHz)
        - C = 3.3 nF

        Input signal:

        - type: ramp signal
        - period: 1/20 mHz
        - amplitude: 4.5 Vpp

        Simulation Setup:

        - signal generator: HP...
        - signal length: 100s (100M control signal samples)
        - total size: 100 MB

        Returns
        -------
        Iterator[np.ndarray, None, None]
            the measured control signal sequence from hardware
        Iterator[np.ndarray, None, None]
            an ideal control signal sequence from simulation
        :py:class:`cbadc.simulator.StateSpaceSimulator`
            an instantiated ideal simulator.

        """
        if self.pcb != "B":
            raise BaseException("This simulation was made with PCB-B")
        input_signal = cbadc.analog_signal.Ramp(4.5, 1/20e-3)
        size = int(100e6)
        ideal_simulator = cbadc.simulator.StateSpaceSimulator(
            self.analog_system, self.digital_control, [input_signal],
            t_stop=self.digital_control.T * size)
        control_signal_hardware = cbadc.utilities.byte_stream_2_control_signal(
            cbadc.utilities.read_byte_stream_from_url(
                'https://people.ee.ethz.ch/~merik/ramp_20mHz_4.5Vpp_1MHz_100s_pcbB.adc', self.analog_system.M),
            self.analog_system.M)
        control_signal_software_simulated = cbadc.utilities.byte_stream_2_control_signal(
            cbadc.utilities.read_byte_stream_from_url(
                'https://people.ee.ethz.ch/~merik/ideal_ramp_20mHz_4.5Vpp_1MHz_100s_pcbB.adc', self.analog_system.M),
            self.analog_system.M)
        return control_signal_hardware, control_signal_software_simulated, ideal_simulator, size
