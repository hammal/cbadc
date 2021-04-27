"""
=============
Downsampling
=============

In this tutorial we demonstrate how to configure the digital estimator
for downsampling.
"""

###############################################################################
# Analog System
# -------------
#
# For this tutorial we will commit to a leap-frog control-bounded analog
# system.
from cbadc.analog_system import LeapFrog
from cbadc.digital_control import DigitalControl
import numpy as np


# Determine system parameters
N = 4
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
# In this tutorial we will use a Sinusodial signal.
from cbadc.analog_signal import Sinusodial

# Set the peak amplitude.
amplitude = 1.0
# Choose the sinusodial frequency via an oversampling ratio (OSR).
frequency = 1.0 / (T * OSR * (1 << 0))

# We also specify a phase an offset these are hovewer optional.
phase = 0.0
offset = 0.0

# Instantiate the analog signal
analog_signal = Sinusodial(amplitude, frequency, phase, offset)

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

# Instantiate digital controls. We will need four of them as we will compare
# four different estimators.
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
# Oversampling = 1
# ----------------------------------------
#
# First we initialize our default estimator without a downsampling parameter
# which then defaults to 1, i.e., no downsampling.
from cbadc.digital_estimator import FIRFilter

# Set the bandwidth of the estimator
G_at_omega = np.linalg.norm(
    analog_system.transfer_function_matrix(np.array([omega_3dB])))
eta2 = G_at_omega**2
print(f"eta2 = {eta2}, {20 * np.log10(eta2)} [dB]")

# Set the filter size
L1 = 1 << 13
L2 = L1

# Instantiate the digital estimator.
digital_estimator_ref = FIRFilter(
    simulator1, analog_system, digital_control1, eta2, L1, L2)

print(digital_estimator_ref, "\n")


###############################################################################
# Visualize Estimator's Transfer Function
# ---------------------------------------
#
import matplotlib.pyplot as plt

# Logspace frequencies
frequencies = np.logspace(-3, 0, 100)
omega = 4 * np.pi * beta * frequencies

# Compute NTF
ntf = digital_estimator_ref.noise_transfer_function(omega)
ntf_dB = 20 * np.log10(np.abs(ntf))

# Compute STF
stf = digital_estimator_ref.signal_transfer_function(omega)
stf_dB = 20 * np.log10(np.abs(stf.flatten()))

# Signal attenuation at the input signal frequency
stf_at_omega = digital_estimator_ref.signal_transfer_function(
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
# FIR Filter With Downsampling
# ----------------------------
#
# Next we repeat the initalization steps above but for a downsampled estimator

digital_estimator_dow = FIRFilter(
    simulator2,
    analog_system,
    digital_control2,
    eta2,
    L1,
    L2,
    downsample = OSR)

print(digital_estimator_dow, "\n")

###############################################################################
# Estimating (Filtering)
# ----------------------
#

# Set simulation length
size = L2 << 4
u_hat_ref = np.zeros(size)
u_hat_dow = np.zeros(size // OSR)
for index in range(size):
    u_hat_ref[index] = next(digital_estimator_ref)
for index in range(size // OSR):
    u_hat_dow[index] = next(digital_estimator_dow)

###############################################################################
# Visualizing Results
# -------------------
#
# Finally, we summarize the comparision by visualizing the resulting estimate
# in both time and frequency domain.
from cbadc.utilities import compute_power_spectral_density

# compensate the built in L1 delay of FIR filter.
t = np.arange(-L1 + 1, size - L1 + 1)
t_down = np.arange(-L1//OSR + 1, (size - L1) // OSR + 1)
u = np.zeros_like(u_hat_ref)
for index, tt in enumerate(t):
    u[index] = analog_signal.evaluate( tt * T)
plt.plot(t, u_hat_ref, label="$\hat{u}(t)$ Reference")
plt.plot(t_down, u_hat_dow, label="$\hat{u}(t)$ Downsampled")
plt.plot(t, stf_at_omega * u, label="$\mathrm{STF}(2 \pi f_u) * u(t)$")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimated input signal")
plt.grid(which='both')
# plt.xlim((-100, 500))
plt.tight_layout()

plt.figure()
u_hat_ref_clipped = u_hat_ref[(L1 + L2):]
u_hat_dow_clipped = u_hat_dow[(L1 + L2) // OSR:]
f_ref, psd_ref = compute_power_spectral_density(
  u_hat_ref_clipped, nperseg=1 << 12)
f_dow, psd_dow = compute_power_spectral_density(
    u_hat_dow_clipped, nperseg=1 << 12, fs=1.0/OSR)
plt.semilogx(f_ref, 10 * np.log10(psd_ref), label="$\hat{U}(f)$ Referefence")
plt.semilogx(f_dow, 10 * np.log10(psd_dow), label="$\hat{U}(f)$ Downsampled")
plt.legend()
plt.ylim((-200, 50))
plt.xlim((f_ref[1], f_ref[-1]))
plt.xlabel('frequency [Hz]')
plt.ylabel('$ \mathrm{V}^2 \, / \, (1 \mathrm{Hz})$')
plt.grid(which='both')
plt.show()
