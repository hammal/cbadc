"""
Using Phase Shifted Digital Control
===================================

This example shows the benefit of using the
phase shifted digital control delay.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
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
C_x = 1e-9
C_Gamma = C_x / 2
R_s = 10.0
R_beta = 1e3

beta = 1 / (R_beta * C_x)
T = 1 / (2 * beta)
rho = -1e-2
kappa = -1.0

# In this example, each nodes amplification and local feedback will be set
# identically.
betaVec = beta * np.ones(N)
rhoVec = betaVec * rho
kappaVec = -1 / (C_x * R_s) * np.eye(N)

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
T = 1.0 / (2 * beta)

# Set the number of digital controls to be same as analog states.
M = N
# Initialize the digital control. Note that we decrease the control period by
# M to have the same number of switches per unit-of-time as the reference.
digital_control_phase = cbadc.digital_control.MultiPhaseDigitalControl(
    T,
    T * np.arange(M) / M,
    impulse_response=[
        cbadc.digital_control.RCImpulseResponse(R_s * C_Gamma) for _ in range(M)
    ],
)
digital_control_ref = cbadc.digital_control.DigitalControl(
    T,
    M,
    impulse_response=cbadc.digital_control.RCImpulseResponse(R_s * C_Gamma),
)


###############################################################################
# The Analog Signal
# -----------------
#
# The final and third component of the simulation is an analog signal.
# For this tutorial, we will choose a
# :py:class:`cbadc.analog_signal.Sinusoidal`.

# Set the peak amplitude.
amplitude = 0.5
# Choose the sinusoidal frequency via an oversampling ratio (OSR).
OSR = 1 << 5
frequency = 1.0 / (T * (OSR << 3))

# We also specify a phase an offset these are hovewer optional.
phase = np.pi / 3
offset = 0.0

# Instantiate the analog signal
analog_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)
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

size = 1 << 14

# Instantiate the simulator.
simulator_phase = cbadc.simulator.StateSpaceSimulator(
    analog_system, digital_control_phase, [analog_signal], Ts=T / M
)
simulator_ref = cbadc.simulator.StateSpaceSimulator(
    analog_system, digital_control_ref, [analog_signal], Ts=T / M
)


###############################################################################
# Setting up the Digital Estimation Filters
# -----------------------------------------
#

# Set the bandwidth of the estimator

eta2 = (
    np.linalg.norm(
        analog_system.transfer_function_matrix(np.array([2 * np.pi / T / OSR]))
    ).flatten()
    ** 2
)
# Set the batch size

K1_phase = 1 << 13
K1_ref = K1_phase
# K1_ref = K1_phase // M

# Instantiate the digital estimator (this is where the filter coefficients are
# computed).

digital_estimator_phase = cbadc.digital_estimator.FIRFilter(
    analog_system, digital_control_phase, eta2, K1_phase, K1_phase
)
digital_estimator_ref = cbadc.digital_estimator.FIRFilter(
    analog_system, digital_control_ref, eta2, K1_ref, K1_ref
)

# Set control signal iterator
digital_estimator_phase(simulator_phase)
digital_estimator_ref(simulator_ref)

###############################################################################
# Post filtering the FIR filter coefficients
# -----------------------------------------------------------
#
# Yet another approach is to instead post filter
# the resulting FIR filter digital_estimator.h with another lowpass FIR filter

numtaps = 1001
f_cutoff = 1.0 / OSR
fir_filter_phase = scipy.signal.firwin(numtaps, f_cutoff / M)
fir_filter_ref = scipy.signal.firwin(numtaps, f_cutoff)

digital_estimator_phase.convolve(fir_filter_phase)
digital_estimator_ref.convolve(fir_filter_ref)

###############################################################################
# Simulating and Estimating
# --------------------------
#

sequence_length = size // OSR // M

u_hat_phase = np.zeros(sequence_length)
u_hat_ref = np.zeros(sequence_length)

for index in range(sequence_length):
    u_hat_phase[index] = next(digital_estimator_phase)
    u_hat_ref[index] = next(digital_estimator_ref)


###############################################################################
# Visualize in Time Domain
# --------------------------
#

t = np.arange(sequence_length)
plt.plot(t, u_hat_phase)
plt.plot(t, u_hat_ref)
plt.xlabel("$t / T$")
plt.ylabel("$\hat{u}(t)$")
plt.title("Estimated input signal")
plt.grid()
# plt.xlim((0, T * sequence_length // M // OSR))
plt.ylim((-0.75, 0.75))
plt.tight_layout()

###############################################################################
# Plotting the PSD
# ----------------
#
# As is typical for delta-sigma modulators, we often visualize the performance
# of the estimate by plotting the power spectral density (PSD).

f_phase, psd_phase = cbadc.utilities.compute_power_spectral_density(
    u_hat_phase[K1_phase:], fs=1.0 / digital_control_phase.T / M
)
f_ref, psd_ref = cbadc.utilities.compute_power_spectral_density(
    u_hat_ref[K1_ref:], fs=1.0 / digital_control_ref.T
)
plt.figure()
plt.semilogx(f_phase, 10 * np.log10(psd_phase), label="Phase")
plt.semilogx(f_ref, 10 * np.log10(psd_ref), label="Ref")
plt.legend()
# plt.xlim((1e1, 0.5/digital_control_phase.T))
plt.xlabel("frequency [Hz]")
plt.ylabel("$ \mathrm{V}^2 \, / \, \mathrm{Hz}$")
plt.grid(which="both")

###############################################################################
# Evaluating the Analog State Vector For both controls
# ----------------------------------------------------
#

# Set sampling time three orders of magnitude smaller than the control period
Ts = T / M / 10.0

# Simulate for 10000 control cycles.
size = 15000
end_time = (size + 100) * Ts

# Initialize a new digital control.
digital_control_phase = cbadc.digital_control.MultiPhaseDigitalControl(
    T, T * np.arange(M) / M
)
digital_control_ref = cbadc.digital_control.DigitalControl(T, M)

# With or without input signal?
# analog_signal = cbadc.analog_signal.Sinusoidal(0 * amplitude, frequency, phase, offset)
analog_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)

# Instantiate a new simulator with a sampling time.
simulator_phase = cbadc.simulator.extended_simulation_result(
    cbadc.simulator.StateSpaceSimulator(
        analog_system, digital_control_phase, [analog_signal], Ts=Ts
    )
)
simulator_ref = cbadc.simulator.extended_simulation_result(
    cbadc.simulator.StateSpaceSimulator(
        analog_system, digital_control_ref, [analog_signal], Ts=Ts
    )
)

# Create data containers to hold the resulting data.
time_vector = np.arange(size) * Ts / T
states = np.zeros((2, N, size))
control_signals = np.zeros((2, M, size), dtype=np.int8)

# Iterate through and store states and control_signals.
for index in range(size):
    res = next(simulator_phase)
    states[0, :, index] = res["analog_state"]
    control_signals[0, :, index] = res["control_signal"]
    print(digital_control_phase._t_next, digital_control_phase.control_signal())
    res = next(simulator_ref)
    states[1, :, index] = res["analog_state"]
    control_signals[1, :, index] = res["control_signal"]

# reset figure size and plot individual results.
plt.rcParams["figure.figsize"] = [6.40, 6.40 * 2]
fig, ax = plt.subplots(N, 2)
for index in range(N):
    color1 = next(ax[0, 0]._get_lines.prop_cycler)["color"]
    color2 = next(ax[0, 0]._get_lines.prop_cycler)["color"]
    ax[index, 0].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[index, 1].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[index, 0].plot(time_vector, states[0, index, :], color=color1, label="Phase")
    ax[index, 0].plot(time_vector, states[1, index, :], color=color2, label="Ref")
    ax[index, 1].plot(
        time_vector, control_signals[0, index, :], color=color1, label="Phase"
    )
    ax[index, 1].plot(
        time_vector, control_signals[1, index, :], color=color2, label="Ref"
    )
    ax[index, 0].set_ylabel(f"$x_{index + 1}(t)$")
    ax[index, 1].set_ylabel(f"$s_{index + 1}(t)$")
    ax[index, 0].set_xlim((0, 5))
    ax[index, 1].set_xlim((0, 5))
    ax[index, 0].set_ylim((-1, 1))
    ax[index, 0].legend()
    ax[index, 1].legend()
fig.suptitle("Analog state and control contribution evolution")
ax[-1, 0].set_xlabel("$t / T$")
ax[-1, 1].set_xlabel("$t / T$")
fig.tight_layout()

###############################################################################
# Analog State Statistics
# ------------------------------------------------------------------
#
# As in the previous section, visualizing the analog state trajectory is a
# good way of identifying problems and possible errors. Another way of making
# sure that the analog states remain bounded is to estimate their
# corresponding densities (assuming i.i.d samples).

# Compute L_2 norm of analog state vector.
L_2_norm = np.linalg.norm(states, ord=2, axis=1)
# Similarly, compute L_infty (largest absolute value) of the analog state
# vector.
L_infty_norm = np.linalg.norm(states, ord=np.inf, axis=1)

# Estimate and plot densities using matplotlib tools.
bins = 150
plt.rcParams["figure.figsize"] = [6.40, 4.80]
fig, ax = plt.subplots(2, sharex=True)
ax[0].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
ax[1].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
ax[0].hist(L_2_norm[0, :], bins=bins, density=True, label="Phase")
ax[0].hist(L_2_norm[1, :], bins=bins, density=True, label="Ref")
ax[1].hist(L_infty_norm[0, :], bins=bins, density=True, color="orange", label="Phase")
ax[1].hist(L_infty_norm[1, :], bins=bins, density=True, color="purple", label="Ref")
plt.suptitle("Estimated probability densities")
ax[0].set_xlabel("$\|\mathbf{x}(t)\|_2$")
ax[1].set_xlabel("$\|\mathbf{x}(t)\|_\infty$")
ax[0].set_ylabel("$p ( \| \mathbf{x}(t) \|_2 ) $")
ax[1].set_ylabel("$p ( \| \mathbf{x}(t) \|_\infty )$")
ax[0].legend()
ax[1].legend()
fig.tight_layout()

# sphinx_gallery_thumbnail_number = 2
