"""
The Mid Point Setting
======================

In this tutorial we investigate the effect of a symmetric versus non symmetric
FIR filter.
"""
###############################################################################
# Analog System
# -------------
#
# We will commit to a leap-frog control-bounded analog system throughtout
# this tutorial.
from cbadc.analog_system import LeapFrog
from cbadc.digital_control import DigitalControl
import numpy as np

# Determine system parameters
N = 6
M = N
beta = 6250
# Set control period
T = 1.0 / (2.0 * beta)
# Adjust the feedback to achieve a bandwidth corresponding to OSR.
OSR = 128
omega_3dB = 2 * np.pi / (T * OSR)

# Instantiate analog system.
beta_vec = beta * np.ones(N)
rho_vec = - omega_3dB ** 2 / beta * np.ones(N)
Gamma = np.diag(-beta_vec)
analog_system = LeapFrog(beta_vec, rho_vec, Gamma)

print(analog_system, "\n")

###############################################################################
# Analog Signal
# -------------
#
# We will also need an analog signal for conversion.
# In this tutorial we will use a Sinusoidal signal.
from cbadc.analog_signal import Sinusoidal

# Set the peak amplitude.
amplitude = 1.0
# Choose the sinusoidal frequency via an oversampling ratio (OSR).
frequency = 1.0 / (T * OSR * (1 << 0))

# We also specify a phase an offset these are hovewer optional.
phase = np.pi / 3
offset = 0.0

# Instantiate the analog signal
analog_signal = Sinusoidal(amplitude, frequency, phase, offset)

print(analog_signal)


###############################################################################
# Simulating
# ----------
#
# Each estimator will require an independent stream of control signals.
# Therefore, we will next instantiate several digital controls and simulators.
from cbadc.simulator import StateSpaceSimulator

# Set simulation precision parameters
atol = 1e-6
rtol = 1e-12
max_step= T / 10.

# Instantiate digital controls
digital_control1 = DigitalControl(T, M)
digital_control2 = DigitalControl(T, M)
print(digital_control1)

# Instantiate simulators.
simulator1 = StateSpaceSimulator(
    analog_system,
    digital_control1,
    [analog_signal],
    atol = atol,
    rtol = rtol,
    max_step = max_step
)
simulator2 = StateSpaceSimulator(
    analog_system,
    digital_control2,
    [analog_signal],
    atol = atol,
    rtol = rtol,
    max_step = max_step
)

print(simulator1)

###############################################################################
# Default and Mid point FIR Filter
# --------------------------------
#
# Next we instantiate the quadratic and default estimator
from cbadc.digital_estimator import FIRFilter

# Set the bandwidth of the estimator
G_at_omega = np.linalg.norm(
    analog_system.transfer_function_matrix(np.array([omega_3dB])))
eta2 = G_at_omega**2
print(f"eta2 = {eta2}, {20 * np.log10(eta2)} [dB]")

# Set the batch size
K1 = 1 << 10
K2 = 1 << 10

# Instantiate the default filter
fir_default = FIRFilter(simulator1, analog_system,
                        digital_control1, eta2, K1, K2, mid_point=False)
fir_mid_point = FIRFilter(simulator2, analog_system,
                          digital_control2, eta2, K1, K2, mid_point=True)
print(fir_default, "\n")
print(fir_mid_point, "\n")


###############################################################################
# Visualize Estimator's Transfer Function (Same for Both)
# -------------------------------------------------------
#
import matplotlib.pyplot as plt

# Logspace frequencies
frequencies = np.logspace(-3, 0, 100)
omega = 4 * np.pi * beta * frequencies

# Compute NTF
ntf = fir_mid_point.noise_transfer_function(omega)
ntf_dB = 20 * np.log10(np.abs(ntf))

# Compute STF
stf = fir_mid_point.signal_transfer_function(omega)
stf_dB = 20 * np.log10(np.abs(stf.flatten()))

# Signal attenuation at the input signal frequency
stf_at_omega = fir_mid_point.signal_transfer_function(
    np.array([2 * np.pi * frequency]))[0]

# Plot
plt.figure()
plt.semilogx(frequencies, stf_dB, label='$STF(\omega)$')
for n in range(N):
    plt.semilogx(frequencies, ntf_dB[0, n, :], label=f"$|NTF_{n+1}(\omega)|$")
plt.semilogx(frequencies, 20 * np.log10(np.linalg.norm(
    ntf[0, :, :], axis=0)), '--', label="$ || NTF(\omega) ||_2 $")

# Add labels and legends to figure
plt.legend()
plt.grid(which='both')
plt.title("Signal and noise transfer functions")
plt.xlabel("$\omega / (4 \pi \\beta ) $")
plt.ylabel("dB")
plt.xlim((frequencies[1], frequencies[-1]))
plt.gcf().tight_layout()

###############################################################################
# Impulse Responses
# -----------------
#


# Next visualize the decay of the resulting filter coefficients.
h_index = np.arange(-K1, K2)

impulse_response_default = np.linalg.norm(
    np.array(fir_default.h[:, 0, :]), axis=1) ** 2
impulse_response_default_dB = 10 * np.log10(impulse_response_default)

impulse_response_mid_point = np.linalg.norm(
    np.array(fir_mid_point.h[:, 0, :]), axis=1) ** 2
impulse_response_mid_point_dB = 10 * np.log10(impulse_response_mid_point)

fig, ax = plt.subplots(2)

ax[0].plot(h_index, impulse_response_default, label=f"Default")
ax[1].plot(h_index, impulse_response_default_dB, label=f"Default")
ax[0].plot(h_index, impulse_response_mid_point, label=f"Mid point")
ax[1].plot(h_index, impulse_response_mid_point_dB, label=f"Mid point")
ax[0].legend()
fig.suptitle(f"For $\eta^2 = {20 * np.log10(eta2)}$ [dB]")
ax[1].set_xlabel("filter taps k")
ax[0].set_ylabel("$\| \mathbf{h} [k]\|^2_2$")
ax[1].set_ylabel("$\| \mathbf{h} [k]\|^2_2$ [dB]")
ax[0].set_xlim((-50, 50))
ax[0].grid(which='both')
ax[1].set_xlim((-50, 500))
ax[1].set_ylim((-200, 0))
ax[1].grid(which='both')

###############################################################################
# Estimating (Filtering)
# ----------------------
#
# Next we execute all simulation and estimation tasks by iterating over the
# estimators. Note that since no stop criteria is set for either the analog
# signal, the simulator, or the digital estimator this iteration could
# potentially continue until the default stop criteria of 2^63 iterations.

# Set simulation length
size = K2 << 4
u_hat_default = np.zeros(size)
u_hat_mid_point = np.zeros(size)
for index in range(size):
    u_hat_default[index] = next(fir_default)
    u_hat_mid_point[index] = next(fir_mid_point)

###############################################################################
# Visualizing Results
# -------------------
#
# Finally, we summarize the comparision by visualizing the resulting estimate
# in both time and frequency domain.
from cbadc.utilities import compute_power_spectral_density

t = np.arange(size)
# compensate the built in K1 delay of FIR filter.
t_fir = np.arange(-K1 + 1, size - K1 + 1)
u = np.zeros_like(u_hat_mid_point)
u_mid_point = np.zeros_like(u)
for index, tt in enumerate(t):
    u[index] = analog_signal.evaluate(tt * T)
    u_mid_point[index] = analog_signal.evaluate(tt * T - T / 2.0)
plt.plot(t, stf_at_omega * u, label="$\mathrm{STF}(2 \pi f_u) * u(t)$")
plt.plot(t_fir, u_hat_default, label="$\hat{u}(t)$ Default")
plt.plot(t_fir - 0.5, u_hat_mid_point, label="$\hat{u}(t - T/2)$ Mid point")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimated input signal")
plt.grid(which='both')
plt.xlim((-100, 500))
plt.tight_layout()

plt.figure()
plt.plot(t, stf_at_omega * u, label="$\mathrm{STF}(2 \pi f_u) * u(t)$")
plt.plot(t_fir, u_hat_default, label="$\hat{u}(t)$ Default")
plt.plot(t_fir - 0.5, u_hat_mid_point, label="$\hat{u}(t - T/2)$ Mid point")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimated input signal")
plt.grid(which='both')
plt.xlim((t_fir[-1] + 50, t_fir[-1]))
plt.tight_layout()

plt.figure()
plt.plot(t, stf_at_omega * u, label="$\mathrm{STF}(2 \pi f_u) * u(t)$")
plt.plot(t_fir, u_hat_default, label="$\hat{u}(t)$ Default")
plt.plot(t_fir - 0.5, u_hat_mid_point, label="$\hat{u}(t - T/2)$ Mid point")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimated input signal")
plt.grid(which='both')
plt.xlim((t_fir[0], t[-1]))
plt.tight_layout()

mid_point_error = stf_at_omega * \
    u_mid_point[:(u.size - K1 + 1)] - u_hat_mid_point[(K1 -1):]
default_error = stf_at_omega * u[:(u.size - K1 + 1)] - u_hat_default[(K1 - 1):]
plt.figure()
plt.plot(t[:(u.size - K1 + 1)], mid_point_error,
         label="$|\mathrm{STF}(2 \pi f_u) * u(t) - \hat{u}(t)|$ Mid point")
plt.plot(t[:(u.size - K1 + 1)], default_error,
         label="$|\mathrm{STF}(2 \pi f_u) * u(t) - \hat{u}(t)|$ Default")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimation error")
plt.grid(which='both')
plt.tight_layout()

print(f"Average Mid point error: {np.linalg.norm(mid_point_error) / mid_point_error.size} \nAverage Default error: {np.linalg.norm(default_error) / default_error.size}")

plt.figure()
u_hat_mid_point_clipped = u_hat_mid_point[(K1 + K2):]
u_hat_default_clipped = u_hat_default[(K1 + K2):]
u_clipped = stf_at_omega * u
f_mid_point, psd_mid_point = compute_power_spectral_density(
    u_hat_mid_point_clipped, nperseg=1 << 12)
f_default, psd_default = compute_power_spectral_density(
    u_hat_default_clipped, nperseg=1 << 12)
f_ref, psd_ref = compute_power_spectral_density(u_clipped, nperseg=1 << 12)
plt.semilogx(f_ref, 10 * np.log10(psd_ref),
             label="$\mathrm{STF}(2 \pi f_u) * U(f)$")
plt.semilogx(f_mid_point, 10 * np.log10(psd_mid_point),
             label="$\hat{U}(f)$ Mid point")
plt.semilogx(f_default, 10 * np.log10(psd_default),
             label="$\hat{U}(f)$ Default")
plt.legend()
plt.ylim((-200, 100))
plt.xlim((f_default[1], f_default[-1]))
plt.xlabel('frequency [Hz]')
plt.ylabel('$ \mathrm{V}^2 \, / \, (1 \mathrm{Hz})$')
plt.grid(which='both')
plt.show()
