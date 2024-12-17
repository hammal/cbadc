# import module
import cProfile
from cbadc.analog_filter import ChainOfIntegrators
from cbadc.digital_control import DigitalControl
from cbadc.analog_signal import Sinusoidal, Clock
from cbadc.simulator import get_simulator
from cbadc.digital_estimator.batch_estimator import BatchEstimator
from cbadc.utilities import write_byte_stream_to_file
from cbadc.utilities import control_signal_2_byte_stream
import numpy as np
import os


# We fix the number of analog states.
N = 6
# Set the amplification factor.
beta = 6250.0
# In this example, each nodes amplification and local feedback will be set
# identically.
betaVec = beta * np.ones(N)
rhoVec = -betaVec * 1e-2
kappaVec = -beta * np.eye(N)

# Set the time period which determines how often the digital control updates.
T = 1.0 / (2 * beta)
# Set the number of digital controls to be same as analog states.
M = N

# Set the peak amplitude.
amplitude = 0.5
# Choose the sinusoidal frequency via an oversampling ratio (OSR).
OSR = 1 << 9
frequency = 1.0 / (T * OSR)

# We also specify a phase an offset these are hovewer optional.
phase = np.pi / 3
offset = 0.0


# Simulate for 2^18 control cycles.
end_time = T * (1 << 14)


def main():
    # Analog system
    analog_filter = ChainOfIntegrators(betaVec, rhoVec, kappaVec)

    # Clock
    clock = Clock(T)

    # Initialize the digital control.
    digital_control = DigitalControl(clock, M)

    # Instantiate the analog signal
    analog_signal = Sinusoidal(amplitude, frequency, phase, offset)

    # Instantiate the simulator.
    simulator = get_simulator(
        analog_filter, digital_control, [analog_signal], t_stop=end_time
    )
    K1 = 1 << 11
    K2 = K1
    eta2 = (
        np.linalg.norm(
            analog_filter.transfer_function_matrix(np.array([2 * np.pi * 100]))
        )
        ** 2
    )
    estimator = BatchEstimator(analog_filter, digital_control, eta2, K1=K1, K2=K2)
    estimator(simulator)
    size = 1 << 12
    u_hat = np.zeros(size)
    for index in range(size):
        u_hat[index] = next(estimator)


if __name__ == "__main__":
    cProfile.run("main()", "profile_full_system_stats")
