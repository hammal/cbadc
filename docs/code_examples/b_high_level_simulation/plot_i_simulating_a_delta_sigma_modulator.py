"""
Simulating a Continuous-Time Delta-Sigma Modulator
==================================================
"""
import cbadc
import numpy as np
import matplotlib.pyplot as plt
import json

T = 0.5e-8
N = 4
K1 = 1 << 9
K2 = K1
OSR = 24
BW = 1 / (2 * T * OSR)

###############################################################################
# Instantiating the Analog System and Digital Control
# ---------------------------------------------------
#
# We start by loading a delta sigma modulator constructed
# using [www.sigma-delta.de](www.sigma-delta.de) framework.
#

with open('CTSD_N4_OSR16_Q16_CRFB_OPT1_HINF60.json') as f:
    analog_frontend_ctsd = cbadc.synthesis.ctsd_dict2af(json.load(f), T)

print(analog_frontend_ctsd.analog_system)


eta2_ctsd = (
    np.linalg.norm(
        analog_frontend_ctsd.analog_system.transfer_function_matrix(
            np.array([2 * np.pi * BW])
        )
    )
    ** 2
)

# analog_frontend_ctsd.analog_system.Gamma = -analog_frontend_ctsd.analog_system.Gamma

digital_estimator_ctsd = cbadc.digital_estimator.BatchEstimator(
    analog_frontend_ctsd.analog_system,
    analog_frontend_ctsd.digital_control,
    eta2_ctsd,
    K1,
    K2,
)

###############################################################################
# Leap Frog
# ---------------------------------------------------
#

analog_frontend_leap_frog = cbadc.synthesis.get_leap_frog(OSR=OSR, N=N, BW=BW)

print(analog_frontend_leap_frog.analog_system)

analog_frontend_leap_frog.digital_control = (
    cbadc.digital_control.MultiLevelDigitalControl(
        analog_frontend_leap_frog.digital_control.clock, N, [1] * N
    )
)

eta2_leap_frog = (
    np.linalg.norm(
        analog_frontend_leap_frog.analog_system.transfer_function_matrix(
            np.array([2 * np.pi * BW])
        )
    )
    ** 2
)

digital_estimator_leap_frog = cbadc.digital_estimator.BatchEstimator(
    analog_frontend_leap_frog.analog_system,
    analog_frontend_leap_frog.digital_control,
    eta2_leap_frog,
    K1,
    K2,
)

###############################################################################
#  Input Signal
# ---------------------------------------------------
#
amplitude = 1e-0
phase = 0.0
offset = 0.0
frequency = 1.0 / analog_frontend_ctsd.digital_control.clock.T

while frequency > BW:
    frequency /= 2
input_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)


###############################################################################
#  Transfer Functions
# ---------------------------------------------------
#

# Logspace frequencies
frequencies = np.logspace(3, 8, 1000)
omega = 2 * np.pi * frequencies

# Compute transfer functions for each frequency in frequencies
transfer_function_ctsd = analog_frontend_ctsd.analog_system.transfer_function_matrix(
    omega
)
transfer_function_ctsd_dB = 20 * np.log10(np.abs(transfer_function_ctsd))

transfer_function_leap_frog = (
    analog_frontend_leap_frog.analog_system.transfer_function_matrix(omega)
)
transfer_function_leap_frog_dB = 20 * np.log10(np.abs(transfer_function_leap_frog))

G_omega = 20 * np.log10(np.linalg.norm(transfer_function_ctsd[:, 0, :], axis=0))

plt.semilogx([BW, BW], [np.min(G_omega), np.max(G_omega)], '--', label="BW")

# Add the norm ||G(omega)||_2
plt.semilogx(
    frequencies,
    G_omega,
    label="CTSD $ ||\mathbf{G}(\omega)||_2 $",
)
plt.semilogx(
    frequencies,
    20 * np.log10(np.linalg.norm(transfer_function_leap_frog[:, 0, :], axis=0)),
    label="LF $ ||\mathbf{G}(\omega)||_2 $",
)

# Add labels and legends to figure
plt.legend()
plt.grid(which="both")
plt.title("Transfer functions, $G_1(\omega), \dots, G_N(\omega)$")
plt.xlabel("$\omega / (4 \pi \\beta ) $")
plt.ylabel("dB")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()


###############################################################################
#  Simulation Setup
# ---------------------------------------------------
#

simulator_ctsd = cbadc.simulator.get_simulator(
    analog_frontend_ctsd.analog_system,
    analog_frontend_ctsd.digital_control,
    [input_signal],
)
digital_estimator_ctsd(simulator_ctsd)

simulator_leap_frog = cbadc.simulator.get_simulator(
    analog_frontend_leap_frog.analog_system,
    analog_frontend_leap_frog.digital_control,
    [input_signal],
)
digital_estimator_leap_frog(simulator_leap_frog)


##############################################################################
# Simulate State Trajectories CTSD
# ---------------------------------------------------
#

# Simulate for 65536 control cycles.
size = 1 << 8

time_vector = np.arange(size)
states = np.zeros((N, size))
control_signals = np.zeros((N, size), dtype=np.double)

# Iterate through and store states and control_signals.
simulator = cbadc.simulator.extended_simulation_result(simulator_ctsd)
for index in cbadc.utilities.show_status(range(size)):
    res = next(simulator)
    states[:, index] = res["analog_state"]
    control_signals[:, index] = res["control_signal"]

xlim = 1 << 8
# Plot all analog state evolutions.
plt.figure()
plt.title("Analog state vectors")
for index in range(N):
    plt.plot(time_vector, states[index, :], label=f"$x_{index + 1}(t)$")
plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
plt.xlabel("$t/T$")
plt.xlim((0, xlim))
plt.legend()


# reset figure size and plot individual results.
plt.rcParams["figure.figsize"] = [6.40, 6.40 * 2]
fig, ax = plt.subplots(N, 2)
for index in range(N):
    color = next(ax[0, 0]._get_lines.prop_cycler)["color"]
    ax[index, 0].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[index, 1].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[index, 0].plot(time_vector, states[index, :], color=color)
    ax[index, 1].plot(time_vector, control_signals[0, :], "--", color=color)
    ax[index, 0].set_ylabel(f"$x_{index + 1}(t)$")
    ax[index, 1].set_ylabel(f"$s_{index + 1}(t)$")
    ax[index, 0].set_xlim((0, xlim))
    ax[index, 1].set_xlim((0, xlim))
    ax[index, 0].set_ylim((-1, 1))
fig.suptitle("Analog state and control contribution evolution")
ax[-1, 0].set_xlabel("$t / T$")
ax[-1, 1].set_xlabel("$t / T$")
fig.tight_layout()


##############################################################################
# Simulate State Trajectories Leap Frog
# ---------------------------------------------------
#

# Simulate for 65536 control cycles.
size = 1 << 8

time_vector = np.arange(size)
states = np.zeros((N, size))
control_signals = np.zeros((N, size), dtype=np.double)

# Iterate through and store states and control_signals.
simulator = cbadc.simulator.extended_simulation_result(simulator_leap_frog)
for index in cbadc.utilities.show_status(range(size)):
    res = next(simulator)
    states[:, index] = res["analog_state"]
    control_signals[:, index] = res["control_signal"]

xlim = 1 << 8
# Plot all analog state evolutions.
plt.figure()
plt.title("Analog state vectors")
for index in range(N):
    plt.plot(time_vector, states[index, :], label=f"$x_{index + 1}(t)$")
plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
plt.xlabel("$t/T$")
plt.xlim((0, xlim))
plt.legend()


# reset figure size and plot individual results.
plt.rcParams["figure.figsize"] = [6.40, 6.40 * 2]
fig, ax = plt.subplots(N, 2)
for index in range(N):
    color = next(ax[0, 0]._get_lines.prop_cycler)["color"]
    ax[index, 0].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[index, 1].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[index, 0].plot(time_vector, states[index, :], color=color)
    ax[index, 1].plot(time_vector, control_signals[0, :], "--", color=color)
    ax[index, 0].set_ylabel(f"$x_{index + 1}(t)$")
    ax[index, 1].set_ylabel(f"$s_{index + 1}(t)$")
    ax[index, 0].set_xlim((0, xlim))
    ax[index, 1].set_xlim((0, xlim))
    ax[index, 0].set_ylim((-1, 1))
fig.suptitle("Analog state and control contribution evolution")
ax[-1, 0].set_xlabel("$t / T$")
ax[-1, 1].set_xlabel("$t / T$")
fig.tight_layout()

###############################################################################
#  Simulation
# ---------------------------------------------------
#

size = 1 << 14
u_hat_ctsd = np.zeros(size)
u_hat_leap_frog = np.zeros(size)

for index in range(size):
    u_hat_ctsd[index] = next(digital_estimator_ctsd)
    u_hat_leap_frog[index] = next(digital_estimator_leap_frog)

u_hat_ctsd = u_hat_ctsd[K1 + K2 :]
u_hat_leap_frog = u_hat_leap_frog[K1 + K2 :]

# ###############################################################################
# #  Visualize Results
# # ---------------------------------------------------
# #

f, psd = cbadc.utilities.compute_power_spectral_density(
    u_hat_ctsd[:],
    fs=1 / analog_frontend_ctsd.digital_control.clock.T,
    nperseg=u_hat_ctsd.size,
)
signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
noise_index = np.ones(psd.size, dtype=bool)
noise_index[signal_index] = False
noise_index[f < (BW * 1e-2)] = False
noise_index[f > BW] = False
fom = cbadc.utilities.snr_spectrum_computation_extended(
    psd, signal_index, noise_index, fs=1 / analog_frontend_ctsd.digital_control.clock.T
)
est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
est_ENOB = cbadc.fom.snr_to_enob(est_SNR)
plt.semilogx(
    f,
    10 * np.log10(np.abs(psd)),
    label=f"CTSD, OSR={1/(2 * analog_frontend_ctsd.digital_control.clock.T * BW):.0f}, est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB",
)

f, psd = cbadc.utilities.compute_power_spectral_density(
    u_hat_leap_frog[:],
    fs=1 / analog_frontend_ctsd.digital_control.clock.T,
    nperseg=u_hat_leap_frog.size,
)
signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
noise_index = np.ones(psd.size, dtype=bool)
noise_index[signal_index] = False
noise_index[f < (BW * 1e-2)] = False
noise_index[f > BW] = False
fom = cbadc.utilities.snr_spectrum_computation_extended(
    psd, signal_index, noise_index, fs=1 / analog_frontend_ctsd.digital_control.clock.T
)
est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
est_ENOB = cbadc.fom.snr_to_enob(est_SNR)
plt.semilogx(
    f,
    10 * np.log10(np.abs(psd)),
    label=f"LF, OSR={1/(2 * analog_frontend_ctsd.digital_control.clock.T * BW):.0f}, est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB",
)

plt.title("Power spectral density of input estimate")
plt.xlabel('Hz')
plt.ylabel('$V^2$ / Hz dB')
plt.legend()
plt.grid(which="both")
plt.gcf().tight_layout()
