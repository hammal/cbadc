"""
Compare Estimators
==================

"""
###############################################################################
# System
# ---------
#
from cbadc.analog_system import LeapFrog
from cbadc.digital_control import DigitalControl
import numpy as np

N = 6
M = N
beta = 6250
T = 1.0 / (2.0 * beta)
OSR = 64
omega_3dB = 2 * np.pi / (T * OSR)

beta_vec = beta * np.ones(N)
rho_vec = - omega_3dB ** 2 / beta * np.ones(N)
Gamma = np.diag(-beta_vec)

analog_system = LeapFrog(beta_vec, rho_vec, Gamma)
digital_control1 = DigitalControl(T, M)
digital_control2 = DigitalControl(T, M)

print(analog_system, "\n")
print(digital_control1)

###############################################################################
# Signal
# ---------
#
from cbadc.analog_signal import Sinusodial, ConstantSignal

# Set the peak amplitude.
amplitude = 1.0
# Choose the sinusodial frequency via an oversampling ratio (OSR).
frequency = 1.0 / (T * OSR * (1 << 0))

# We also specify a phase an offset these are hovewer optional.
phase = np.pi / 3
offset = 0.0

# Instantiate the analog signal
analog_signal = Sinusodial(amplitude, frequency, phase, offset)
# analog_signal = ConstantSignal(offset)
# print to ensure correct parametrization.
print(analog_signal)


###############################################################################
# Simulating
# ----------
#
from cbadc.simulator import StateSpaceSimulator
atol = 1e-6
rtol = 1e-3
max_step= T / 10.

# Instantiate two simulators.
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



###############################################################################
# Setup Filter
# ------------
#
from cbadc.digital_estimator import DigitalEstimator, FIRFilter

# Set the bandwidth of the estimator
G_at_omega = np.linalg.norm(analog_system.transfer_function(np.array([omega_3dB])))
eta2 = G_at_omega**2
print(f"eta2 = {eta2}, {10 * np.log10(eta2)} [dB]")

# Set the batch size
K1 = 1 << 10
K2 = 1 << 12

# Instantiate the digital estimator (this is where the filter coefficients are computed).
digital_estimator_batch = DigitalEstimator(simulator1, analog_system, digital_control1, eta2, K1, K2)
digital_estimator_fir = FIRFilter(simulator2, analog_system, digital_control2, eta2, K2, K2 + 1)

print(digital_estimator_batch, "\n")
print(digital_estimator_fir, "\n")

###############################################################################
# Visualize Estimator's Transfer Function (Same for Both)
# -------------------------------------------------------
#
import matplotlib.pyplot as plt

# Logspace frequencies
frequencies = np.logspace(-3, 0, 100)
omega = 4 * np.pi * beta * frequencies

# Compute NTF
ntf = digital_estimator_batch.noise_transfer_function(omega)
ntf_dB = 20 * np.log10(np.abs(ntf))

# Compute STF
stf = digital_estimator_batch.signal_transfer_function(omega)
stf_dB = 20 * np.log10(np.abs(stf.flatten()))

# Signal attenuation at the input signal frequency
stf_at_omega = digital_estimator_batch.signal_transfer_function(np.array([2 * np.pi * frequency]))[0]

# Plot
plt.figure()
plt.semilogx(frequencies, stf_dB, label='$STF(\omega)$')
for n in range(N):
    plt.semilogx(frequencies, ntf_dB[0, n, :], label=f"$|NTF_{n+1}(\omega)|$")
plt.semilogx(frequencies, 20 * np.log10(np.linalg.norm(ntf[0,:,:], axis=0)), '--', label="$ || NTF(\omega) ||_2 $")

# Add labels and legends to figure
plt.legend()
plt.grid(which='both')
plt.title("Signal and noise transfer functions")
plt.xlabel("$\omega / (4 \pi \\beta ) $")
plt.ylabel("dB")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()

###############################################################################
# Estimating (Filtering)
# ----------------------
#

# Set simulation length
size = 1 << 15
u_hat_batch = np.zeros(size)
u_hat_fir = np.zeros(size)
for index in range(size):
    u_hat_batch[index] = np.array(next(digital_estimator_batch))
    u_hat_fir[index] = np.array(next(digital_estimator_fir))

    

###############################################################################
# Visualizing Results
# -------------------
#
from cbadc.utilities import compute_power_spectral_density

t = np.arange(size)
u = np.zeros_like(u_hat_batch)
for index, tt in enumerate(t):
    u[index] = analog_signal.evaluate( tt * T)
plt.plot(t, stf_at_omega * u, label="$\mathrm{STF}(2 \pi f_u) * u(t)$")
plt.plot(t, u_hat_batch, label="$\hat{u}(t)$ Batch")
plt.plot(t, u_hat_fir, label="$\hat{u}(t)$ FIR")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimated input signal")
plt.grid(which='both')
# plt.xlim((0, 500))
plt.tight_layout()

plt.figure()
u_hat_batch_clipped = u_hat_batch[K1 + K2 + 1:]
u_hat_fir_clipped = u_hat_fir[K1 + K2 + 1:]
u_clipped = stf_at_omega * u[K1 + K2 + 1:]
f, psd_batch = compute_power_spectral_density(u_hat_batch_clipped)
f, psd_fir = compute_power_spectral_density(u_hat_fir_clipped)
_, psd_ref = compute_power_spectral_density(u_clipped)
plt.semilogx(f, 10 * np.log10(psd_ref), label="$\mathrm{STF}(2 \pi f_u) * U(f)$")
plt.semilogx(f, 10 * np.log10(psd_batch), label="$\hat{U}(f)$ Batch")
plt.semilogx(f, 10 * np.log10(psd_fir), label="$\hat{U}(f)$ FIR")
plt.legend()
plt.ylim((-200, 40))
plt.xlim((f[1], f[-1]))
plt.xlabel('frequency [Hz]')
plt.ylabel('$ \mathrm{V}^2 \, / \, (1 \mathrm{Hz})$')
plt.grid(which='both')

###############################################################################
# Plot FIR Filter Attenuation
# ---------------------------
#
h_index = np.arange(-K2, K2 + 1)

impulse_response = 20 * np.log10(np.abs(np.array(digital_estimator_fir.h[:,0,:])))

plt.figure()
for index in range(N):
    plt.plot(h_index, impulse_response[:, index], label=f"$h_{index + 1}[k]$")
plt.legend()
plt.title(f"For $\eta^2 = {10 * np.log10(eta2)}$ [dB]")
plt.xlabel("filter taps k")
plt.ylabel("$| h_\ell [k]|^2_2$ [dB]")
plt.xlim((-K2, K2 + 1))
plt.grid(which='both')