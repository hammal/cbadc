"""
Digital Estimation (Post Filtering)
===================================

How to convert a stream of control signals into a signal estimate.
"""
###############################################################################
# Setting up the Analog System and Digital Control
# ------------------------------------------------
#
# In this example we are assuming that we have access to a control signal 
# s[k] generated by the iteractions of an analog system and digital control.
# Furthermore, we assume the analog system to be a third-order 
# chain-of-integrators converter with a standard digital control. 

# Setup analog system and digital control
from cbadc.analog_system import AnalogSystem
from cbadc.digital_control import DigitalControl
from cbadc.digital_estimator import DigitalEstimator
N = 6
M = N
beta = 6250.
rho = - beta * 1e-2
A = [[rho, 0, 0, 0, 0, 0], 
     [beta, rho, 0, 0, 0, 0], 
     [0, beta, rho, 0, 0, 0], 
     [0, 0, beta, rho, 0, 0],
     [0, 0, 0, beta, rho, 0],
     [0, 0, 0, 0, beta, rho]]
B = [[beta], [0], [0], [0], [0], [0]]
CT = [[0, 0, 0, 0, 0, 1]]
Gamma = [[-beta, 0, 0, 0, 0, 0], 
         [0, -beta, 0, 0, 0, 0], 
         [0, 0, -beta, 0, 0, 0],
         [0, 0, 0, -beta, 0, 0],
         [0, 0, 0, 0, -beta, 0],
         [0, 0, 0, 0, 0, -beta]]
Gamma_tildeT = [[1, 0, 0, 0, 0, 0], 
                [0, 1, 0, 0, 0, 0], 
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]]
T = 1.0/(2 * beta)

analog_system = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
digital_control = DigitalControl(T, M)

# Summarize the analog system, digital control, and digital estimator.
print(analog_system, "\n")
print(digital_control)

###############################################################################
# Creating a Placehold Control Signal
# -----------------------------------
#
# We could of course simulate the analog system and digital control above
# for a given analog signal. However, this might not always be the use case
# instead imagine we have acquired such a control signal from a real circuit
# simulation. 
import numpy as np
from cbadc.utilities import random_control_signal

# In principle, we can create a dummy generator by just 
def dummy_control_sequence_signal():
    while(True):
        yield np.zeros(M, dtype=np.int8)
# and then pass dummy_control_sequence_signal as the control_sequence
# to the digital estimator. 

# Another way would be to use a random control signal. Such a generator
# is already provided in the :func:`cbadc.utilities.random_control_signal` function.
# Subsequently, a random (random 1-0 valued M tuples) control signal of length
sequence_length = 10
# can conveniently be created as
control_signal_sequences = random_control_signal(M, stop_after_number_of_iterations=sequence_length, random_seed=42)
# where random_seed and stop_after_number_of_iterations are fully optional

###############################################################################
# Setting up the Filter
# ------------------------------------
#
# To produce estimates we need to compute the filter coefficients of the
# digital estimator. This is part of the instantiation process of the 
# DigitalEstimator class. However, these computations require us to 
# specify both the analog system, the digital control and the filter parameters
# such as eta2, the batch size K1, and possible the lookahead K2.

# Set the bandwidth of the estimator
eta2 = 1e7
# Set the batch size
K1 = sequence_length

# Instantiate the digital estimator (this is where the filter coefficients are computed).
digital_estimator = DigitalEstimator(control_signal_sequences, analog_system, digital_control, eta2, K1)

print(digital_estimator, "\n")


###############################################################################
# Producing Estimates
# -------------------
#
# At this point we can produce estimates by simply calling the iterator

for i in digital_estimator:
    print(i)


###############################################################################
# Batch Size and Lookahead
# ------------------------
#
# Note that batche sizes and lookahead sizes are automatically handled such that
# for
K1 = 5
K2 = 1
sequence_length = 11
control_signal_sequences = random_control_signal(M, stop_after_number_of_iterations=sequence_length, random_seed=42)
digital_estimator = DigitalEstimator(control_signal_sequences, analog_system, digital_control, eta2, K1, K2)

# The iterator is still called the same way.
for i in digital_estimator:
    print(i)
# However, this time this iterator involves computing two batches each involving a lookahead of size one.

###############################################################################
# Loading Control Signal from File
# --------------------------------
#
# Next we will load an actual control signal to demonstrate the digital 
# estimators capabilities. To this end we will use the 
# `sinusodial_simulation.adc` file that was produced in 
# :doc:`./plot_b_simulate_a_control_bounded_adc`.
#
# The control signal file is encoded as raw binary data so to unpack it 
# correctly we will use the :func:`cbadc.utilities.read_byte_stream_from_file`
# and :func:`cbadc.utilities.byte_stream_2_control_signal` functions.
from cbadc.utilities import read_byte_stream_from_file, byte_stream_2_control_signal

byte_stream = read_byte_stream_from_file('sinusodial_simulation.adc', M)
control_signal_sequences = byte_stream_2_control_signal(byte_stream, M)

###############################################################################
# Estimating the input
# --------------------
#
# Fortunately, we used the same
# analog system, and digital controls as in this example so 
#
import matplotlib.pyplot as plt

stop_after_number_of_iterations = 1 << 17
u_hat = np.zeros(stop_after_number_of_iterations)
K1 = 1 << 10
K2 = 1 << 11
digital_estimator = DigitalEstimator(
    control_signal_sequences, 
    analog_system, digital_control, 
    eta2, 
    K1, 
    K2,
    stop_after_number_of_iterations=stop_after_number_of_iterations
    )
for index, u_hat_temp in enumerate(digital_estimator):
    u_hat[index] = u_hat_temp

t = np.arange(u_hat.size)
plt.plot(t, u_hat)
plt.xlabel('$t / T$')
plt.ylabel('$\hat{u}(t)$')
plt.title("Estimated input signal")
plt.grid()
plt.xlim((0, 750))
plt.tight_layout()

###############################################################################
# Plotting the PSD
# ----------------
#
# As is typical for delta-sigma modulators we often visualize the performance 
# of the estimate by plotting the power spectral density (PSD).
from cbadc.utilities import compute_power_spectral_density

f, psd = compute_power_spectral_density(u_hat[K2:])
plt.figure()
plt.semilogx(f, 10 * np.log10(psd))
plt.xlabel('frequency [Hz]')
plt.ylabel('$ \mathrm{V}^2 \, / \, \mathrm{Hz}$')
plt.xlim((f[1], f[-1]))
plt.grid(which='both')

# sphinx_gallery_thumbnail_number = 2
