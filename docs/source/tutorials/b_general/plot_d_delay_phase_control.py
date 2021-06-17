"""
Using Phase Shifted Digital Control (draft)
===========================================

This example shows the benefit of using the
phase shifted digital control delay.
"""
import matplotlib.pyplot as plt
import numpy as np
import cbadc

###############################################################################
# The Analog System
# -----------------
#
# In this example we commit to using a forth order leap-frog analog system,
# see :py:class:`cbadc.analog_system.LeapFrog`.

# We fix the number of analog states.
N = 4
# Set the amplification factor.
beta = 6250.
# In this example, each nodes amplification and local feedback will be set
# identically.
betaVec = beta * np.ones(N)
rhoVec = -betaVec * 1e-2
kappaVec = - beta * np.eye(N)

# Instantiate a chain-of-integrators analog system.
analog_system = cbadc.analog_system.LeapFrog(betaVec, rhoVec, kappaVec)
# print the analog system such that we can very it being correctly initalized.
print(analog_system)

###############################################################################
# The Digital Control
# -------------------
#
# we use the delayed version :py:class:`cbadc.digital_control.PhaseDelayedControl`
# as well as the
# :py:class:`cbadc.digital_control.DigitalControl` for comparision.

# Set the time period which determines how often the digital control updates.
T = 1.0/(2 * beta)

# Set the number of digital controls to be same as analog states.
M = N
# Initialize the digital control. Note that we decrease the control period by
# M to have the same number of switches per unit-of-time as the reference.
digital_control_phase = cbadc.digital_control.PhaseDelayedControl(T / M, M)
digital_control_ref = cbadc.digital_control.DigitalControl(T, M)


###############################################################################
# The Analog Signal
# -----------------
#
# The final and third component of the simulation is an analog signal.
# For this tutorial, we will choose a
# :py:class:`cbadc.analog_signal.Sinusodial`.

# Set the peak amplitude.
amplitude = 0.5
# Choose the sinusodial frequency via an oversampling ratio (OSR).
OSR = 1 << 9
frequency = 1.0 / (T * OSR)

# We also specify a phase an offset these are hovewer optional.
phase = np.pi / 3
offset = 0.0

# Instantiate the analog signal
analog_signal = cbadc.analog_signal.Sinusodial(
    amplitude, frequency, phase, offset)
# print to ensure correct parametrization.
print(analog_signal)

###############################################################################
# Simulating
# -------------
#
# Next, we set up the simulator. Here we use the
# :py:class:`cbadc.simulator.StateSpaceSimulator` for simulating the
# involved differential equations as outlined in
# :py:class:`cbadc.analog_system.AnalogSystem`.
#

size = 1 << 16
end_time = T * (size + 100)

# Instantiate the simulator.
simulator_phase = cbadc.simulator.StateSpaceSimulator(analog_system, digital_control_phase, [
    analog_signal], t_stop=end_time)
simulator_ref = cbadc.simulator.StateSpaceSimulator(analog_system, digital_control_ref, [
    analog_signal], t_stop=end_time)


###############################################################################
# Setting up the Digital Estimation Filters
# -----------------------------------------
#

# Set the bandwidth of the estimator

eta2 = 1e4

# Set the batch size

K1_phase = 1 << 10
K1_ref = K1_phase
# K1_ref = K1_phase // M

# Instantiate the digital estimator (this is where the filter coefficients are
# computed).

digital_estimator_phase = cbadc.digital_estimator.FIRFilter(
    analog_system, digital_control_phase, eta2, K1_phase, K1_phase)
digital_estimator_ref = cbadc.digital_estimator.FIRFilter(
    analog_system, digital_control_ref, eta2, K1_ref, K1_ref)

# Set control signal iterator
digital_estimator_phase(simulator_phase)
digital_estimator_ref(simulator_ref)

###############################################################################
# Simulating and Estimating
# --------------------------
#

sequence_length = size

u_hat_phase = np.zeros(sequence_length)
u_hat_ref = np.zeros(sequence_length)

for index in range(sequence_length):
    u_hat_phase[index] = next(digital_estimator_phase)
    u_hat_ref[index] = next(digital_estimator_ref)


###############################################################################
# Visualize in Time Domain
# --------------------------
#

t = np.arange(sequence_length // M) * T
plt.plot(t, u_hat_phase[::M])
plt.plot(t, u_hat_ref[:sequence_length // M])
plt.xlabel('$t$')
plt.ylabel('$\hat{u}(t)$')
plt.title("Estimated input signal")
plt.grid()
plt.xlim((0, T * sequence_length // M))
plt.ylim((-0.75, 0.75))
plt.tight_layout()

###############################################################################
# Plotting the PSD
# ----------------
#
# As is typical for delta-sigma modulators, we often visualize the performance
# of the estimate by plotting the power spectral density (PSD).

f_phase, psd_phase = cbadc.utilities.compute_power_spectral_density(
    u_hat_phase[K1_phase:], fs=1.0/digital_control_phase.T)
f_ref, psd_ref = cbadc.utilities.compute_power_spectral_density(
    u_hat_ref[K1_ref:], fs=1.0/digital_control_ref.T)
plt.figure()
plt.semilogx(f_phase, 10 * np.log10(psd_phase), label="Phase")
plt.semilogx(f_ref, 10 * np.log10(psd_ref), label="Ref")
plt.legend()
plt.xlim((1e1, 0.5/digital_control_phase.T))
plt.xlabel('frequency [Hz]')
plt.ylabel('$ \mathrm{V}^2 \, / \, \mathrm{Hz}$')
plt.grid(which='both')

# sphinx_gallery_thumbnail_number = 2
