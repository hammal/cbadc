"""
Simulating a Continuous-Time Delta-Sigma Modulator
==================================================
"""
import cbadc
import numpy as np
import matplotlib.pyplot as plt
import json

T = 0.1e-8
N = 5
K1 = 1 << 9
K2 = K1
OSR = 16
BW = 1 / (2 * T * OSR)

###############################################################################
# Instantiating the Analog System and Digital Control
# ---------------------------------------------------
#
# We start by loading a delta sigma modulator constructed
# using [www.sigma-delta.de](www.sigma-delta.de) framework.
#

with open('CTSD_N5_OSR16_Q32_CRFF_OPT1_HINF600.json') as f:
    analog_frontend_ctsd = cbadc.synthesis.ctsd_dict2af(json.load(f), T)

eta2_ctsd = (
    np.linalg.norm(
        analog_frontend_ctsd.analog_system.transfer_function_matrix(
            np.array([2 * np.pi * BW])
        )
    )
    ** 2
)

digital_estimator_ctsd = cbadc.digital_estimator.BatchEstimator(
    analog_frontend_ctsd.analog_system,
    analog_frontend_ctsd.digital_control,
    eta2_ctsd,
    K1,
    K2,
)

print(analog_frontend_ctsd.analog_system)

###############################################################################
# Leap Frog
# ---------------------------------------------------
#

analog_frontend_leap_frog = cbadc.synthesis.get_leap_frog(
    OSR=OSR, N=N, BW=BW, opt=False
)

analog_frontend_leap_frog.digital_control = (
    cbadc.digital_control.MultiLevelDigitalControl(
        analog_frontend_leap_frog.digital_control.clock, N, [1] * N
    )
)

# Scale B
# analog_frontend_leap_frog.analog_system.B = (
#     2 * analog_frontend_leap_frog.analog_system.B
# )


# Scale Gamma
# analog_frontend_leap_frog.analog_system.Gamma = np.dot(
#     np.diag(np.array([0.5 ** (x + 1) for x in range(N)])),
#     analog_frontend_leap_frog.analog_system.Gamma,
# )

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

print(analog_frontend_leap_frog.analog_system)

###############################################################################
#  Input Signal
# ---------------------------------------------------
#
amplitude = 0.25e-0
phase = 0.0
offset = 0.0
frequency = 1.0 / analog_frontend_ctsd.digital_control.clock.T

while frequency > BW:
    frequency /= 2
input_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)
# input_signal = cbadc.analog_signal.ConstantSignal(amplitude)

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
    20 * np.log10(np.linalg.norm(transfer_function_leap_frog[:, 0, :], axis=0)),
    label="LF $ ||\mathbf{G}(\omega)||_2 $",
)
plt.semilogx(
    frequencies,
    G_omega,
    label="CTSD $ ||\mathbf{G}(\omega)||_2 $",
)


# Add labels and legends to figure
plt.legend()
plt.grid(which="both")
plt.xlabel("$f$ [Hz]")
plt.ylabel("dB")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()


for n in range(N):
    plt.figure()
    #     color = next(plt.gca()._get_lines.prop_cycler)["color"]
    plt.semilogx(
        frequencies,
        transfer_function_leap_frog_dB[n, 0, :],
        label="LF $G_" + f"{n+1}" + "(f)$",
        # color = color
    )
    plt.semilogx(
        frequencies,
        transfer_function_ctsd_dB[n, 0, :],
        '--',
        label="CTSD $G_" + f"{n+1}" + "(f)$",
        #     # color = color
    )
    plt.legend()
    plt.grid(which="both")
    plt.xlabel("$f$ [Hz]")
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
# Simulate State Trajectories
# ---------------------------------------------------
#

# Simulate for 65536 control cycles.
size = 1 << 14

time_vector = np.arange(size)
states = np.zeros((N, size, 2))
control_signals = np.zeros((N, size, 2), dtype=np.double)

# Iterate through and store states and control_signals.
simulator_ctsd = cbadc.simulator.extended_simulation_result(simulator_ctsd)
simulator_leap_frog = cbadc.simulator.extended_simulation_result(simulator_leap_frog)
for index in cbadc.utilities.show_status(range(size)):
    res_ctsd = next(simulator_ctsd)
    states[:, index, 0] = res_ctsd["analog_state"]
    control_signals[:, index, 0] = res_ctsd["control_signal"]
    res_leap_frog = next(simulator_leap_frog)
    states[:, index, 1] = res_leap_frog["analog_state"]
    control_signals[:, index, 1] = res_leap_frog["control_signal"]

xlim = 1 << 12
# Plot all analog state evolutions.
plt.figure()
plt.title("Analog state vectors")
for index in range(N):
    plt.plot(time_vector, states[index, :, 1], label=f"LF $x_{index + 1}(t)$")
    plt.plot(time_vector, states[index, :, 0], label=f"CTSD $x_{index + 1}(t)$")
plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
plt.xlabel("$t/T$")
plt.xlim((0, xlim))
plt.legend()


# reset figure size and plot individual results.
plt.rcParams["figure.figsize"] = [6.40, 6.40 * 2]
fig, ax = plt.subplots(N, 2)
for index in range(N):
    color = next(ax[0, 0]._get_lines.prop_cycler)["color"]
    color2 = next(ax[0, 0]._get_lines.prop_cycler)["color"]
    ax[index, 0].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[index, 1].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[index, 0].plot(time_vector, states[index, :, 1], color=color2, label="LF")
    ax[index, 0].plot(time_vector, states[index, :, 0], color=color, label="CTSD")
    ax[index, 1].plot(
        time_vector, control_signals[0, :, 1], "--", color=color2, label="LF"
    )
    if index == (N - 1):
        ax[index, 1].plot(
            time_vector, control_signals[0, :, 0], "--", color=color, label="CTSD"
        )
    ax[index, 0].set_ylabel(f"$x_{index + 1}(t)$")
    ax[index, 1].set_ylabel(f"$s_{index + 1}(t)$")
    ax[index, 0].set_xlim((0, xlim))
    ax[index, 1].set_xlim((0, xlim))
    ax[index, 0].set_ylim((-1, 1))
    ax[index, 0].legend()
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

###############################################################################
#  Visualize Results
# ---------------------------------------------------
#

plt.rcParams["figure.figsize"] = [6.40 * 1.34, 6.40]

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


###############################################################################
#  Time
# ---------------------------------------------------
#

t = np.arange(u_hat_ctsd.size)
plt.plot(t, u_hat_ctsd, label="CTSD")
plt.plot(t, u_hat_leap_frog, label="LF")
plt.xlabel("$t / T$")
plt.ylabel("$\hat{u}(t)$")
plt.title("Estimated input signal")
plt.grid()
plt.xlim((0, 1500))
plt.ylim((-1, 1))
plt.tight_layout()
