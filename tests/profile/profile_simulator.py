# import module
import cProfile
from cbadc.analog_system import ChainOfIntegrators
from cbadc.digital_control import DigitalControl
from cbadc.analog_signal import Sinusodial
from cbadc.simulator import StateSpaceSimulator
from cbadc.utilities import write_byte_stream_to_file
from cbadc.utilities import control_signal_2_byte_stream
import numpy as np
import os


# We fix the number of analog states.
N = 6
# Set the amplification factor.
beta = 6250.
# In this example, each nodes amplification and local feedback will be set
# identically.
betaVec = beta * np.ones(N)
rhoVec = -betaVec * 1e-2
kappaVec = - beta * np.eye(N)

# Set the time period which determines how often the digital control updates.
T = 1.0/(2 * beta)
# Set the number of digital controls to be same as analog states.
M = N

# Set the peak amplitude.
amplitude = 0.5
# Choose the sinusodial frequency via an oversampling ratio (OSR).
OSR = 1 << 9
frequency = 1.0 / (T * OSR)

# We also specify a phase an offset these are hovewer optional.
phase = np.pi / 3
offset = 0.0


# Simulate for 2^18 control cycles.
end_time = T * (1 << 14)


def main():
    # Analog system
    analog_system = ChainOfIntegrators(betaVec, rhoVec, kappaVec)

    # Initialize the digital control.
    digital_control = DigitalControl(T, M)

    # Instantiate the analog signal
    analog_signal = Sinusodial(amplitude, frequency, phase, offset)

    # Instantiate the simulator.
    simulator = StateSpaceSimulator(analog_system, digital_control, [
                                    analog_signal], t_stop=end_time)
    # Depending on your analog system the step above might take some time to
    # compute as it involves precomputing solutions to initial value problems.

    # Construct byte stream.
    byte_stream = control_signal_2_byte_stream(simulator, M)
    write_byte_stream_to_file("./temp.adcs", byte_stream)
    os.remove("./temp.adcs")


if __name__ == '__main__':
    cProfile.run('main()', 'profile_simulator_stats')
