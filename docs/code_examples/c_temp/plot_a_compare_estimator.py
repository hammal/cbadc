"""
Compare Estimators
==================

In this tutorial we investigate different estimator implementation techniques
and compare their performance.
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
# In this tutorial we will use a Sinusodial signal.
from cbadc.analog_signal import Sinusodial

# Set the peak amplitude.
amplitude = 1.0
# Choose the sinusodial frequency via an oversampling ratio (OSR).
frequency = 1.0 / (T * OSR * (1 << 0))

# We also specify a phase an offset these are hovewer optional.
phase = np.pi / 3
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
# Quadratic Complexity Estimator
# ------------------------------
#
# Next we instantiate the quadratic and default estimator
# :py:class:`cbadc.digital_estimator.DigitalEstimator`. Note that during its
# construction, the corresponding filter coefficients of the system will be
# computed. Therefore, this procedure could be computationally intense for a
# analog system with a large analog state order or equivalently for large
# number of independent digital controls.
from cbadc.digital_estimator import DigitalEstimator

# Set the bandwidth of the estimator
G_at_omega = np.linalg.norm(
    analog_system.transfer_function_matrix(np.array([omega_3dB])))
eta2 = G_at_omega**2
print(f"eta2 = {eta2}, {20 * np.log10(eta2)} [dB]")

# Set the batch size
K1 = 1 << 10
K2 = 1 << 10

# Instantiate the digital estimator (this is where the filter coefficients are computed).
digital_estimator_batch = DigitalEstimator(
    simulator1, analog_system, digital_control1, eta2, K1, K2)

print(digital_estimator_batch, "\n")


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
stf_at_omega = digital_estimator_batch.signal_transfer_function(
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
# FIR Filter Estimator
# --------------------
#
# Similarly as for the previous estimator the
# :py:class:`cbadc.digital_estimator.FIRFilter` is initalized. Additionally,
# we visualize the decay of the :math:`\|\cdot\|_2` norm of the corresponding
# fiter coefficients. This is an aid to determine if the lookahead and lookback
# sizes L1 and L2 are set sufficiently large.
from cbadc.digital_estimator import FIRFilter

# Determine lookback
L1 = K2
# Determine lookahead
L2 = K2
digital_estimator_fir = FIRFilter(
    simulator2, analog_system, digital_control2, eta2, L1, L2)

print(digital_estimator_fir, "\n")


# Next visualize the decay of the resulting filter coefficients.
h_index = np.arange(-L1, L2)

impulse_response = np.abs(np.array(digital_estimator_fir.h[:, 0, :])) ** 2
impulse_response_dB = 10 * np.log10(impulse_response)

fig, ax = plt.subplots(2)
for index in range(N):
    ax[0].plot(h_index, impulse_response[:, index],
               label=f"$h_{index + 1}[k]$")
    ax[1].plot(h_index, impulse_response_dB[:, index],
               label=f"$h_{index + 1}[k]$")
ax[0].legend()
fig.suptitle(f"For $\eta^2 = {20 * np.log10(eta2)}$ [dB]")
ax[1].set_xlabel("filter taps k")
ax[0].set_ylabel("$| h_\ell [k]|^2_2$")
ax[1].set_ylabel("$| h_\ell [k]|^2_2$ [dB]")
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
u_hat_batch = np.zeros(size)
u_hat_fir = np.zeros(size)
for index in range(size):
    u_hat_batch[index] = next(digital_estimator_batch)
    u_hat_fir[index] = next(digital_estimator_fir)

###############################################################################
# Visualizing Results
# -------------------
#
# Finally, we summarize the comparision by visualizing the resulting estimate
# in both time and frequency domain.
from cbadc.utilities import compute_power_spectral_density

t = np.arange(size)
# compensate the built in L1 delay of FIR filter.
t_fir = np.arange(-L1 + 1, size - L1 + 1)
u = np.zeros_like(u_hat_batch)
for index, tt in enumerate(t):
    u[index] = analog_signal.evaluate( tt * T)
plt.plot(t, stf_at_omega * u, label="$\mathrm{STF}(2 \pi f_u) * u(t)$")
plt.plot(t_fir, u_hat_fir, label="$\hat{u}(t)$ FIR")
plt.plot(t, u_hat_batch, label="$\hat{u}(t)$ Batch")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimated input signal")
plt.grid(which='both')
plt.xlim((-100, 500))
plt.tight_layout()

plt.figure()
plt.plot(t, stf_at_omega * u, label="$\mathrm{STF}(2 \pi f_u) * u(t)$")
plt.plot(t_fir, u_hat_fir, label="$\hat{u}(t)$ FIR")
plt.plot(t, u_hat_batch, label="$\hat{u}(t)$ Batch")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimated input signal")
plt.grid(which='both')
plt.xlim((t_fir[-1] - 50, t_fir[-1]))
plt.tight_layout()

plt.figure()
plt.plot(t, stf_at_omega * u, label="$\mathrm{STF}(2 \pi f_u) * u(t)$")
plt.plot(t, u_hat_batch, label="$\hat{u}(t)$ Batch")
plt.plot(t_fir, u_hat_fir, label="$\hat{u}(t)$ FIR")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimated input signal")
plt.grid(which='both')
plt.xlim((t_fir[0], t[-1]))
plt.tight_layout()

batch_error = stf_at_omega * u - u_hat_batch
fir_error = stf_at_omega * u[:(u.size - L1 + 1)] - u_hat_fir[(L1 - 1):]
plt.figure()
plt.plot(t, batch_error,
         label="$|\mathrm{STF}(2 \pi f_u) * u(t) - \hat{u}(t)|$ Batch")
plt.plot(t[:(u.size - L1 + 1)], fir_error,
         label="$|\mathrm{STF}(2 \pi f_u) * u(t) - \hat{u}(t)|$ FIR")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimation error")
plt.grid(which='both')
plt.tight_layout()

print(
    f"Average Batch Error: {np.linalg.norm(batch_error) / batch_error.size} \nAverage FIR Error: {np.linalg.norm(fir_error) / fir_error.size}")

plt.figure()
u_hat_batch_clipped = u_hat_batch[(K1 + K2):]
u_hat_fir_clipped = u_hat_fir[(L1 + L2):]
u_clipped = stf_at_omega * u
f_batch, psd_batch = compute_power_spectral_density(
    u_hat_batch_clipped, nperseg=1 << 12)
f_fir, psd_fir = compute_power_spectral_density(
    u_hat_fir_clipped, nperseg=1 << 12)
f_ref, psd_ref = compute_power_spectral_density(u_clipped, nperseg=1 << 12)
plt.semilogx(f_ref, 10 * np.log10(psd_ref),
             label="$\mathrm{STF}(2 \pi f_u) * U(f)$")
plt.semilogx(f_batch, 10 * np.log10(psd_batch), label="$\hat{U}(f)$ Batch")
plt.semilogx(f_fir, 10 * np.log10(psd_fir), label="$\hat{U}(f)$ FIR")
plt.legend()
plt.ylim((-200, 100))
plt.xlim((f_fir[1], f_fir[-1]))
plt.xlabel('frequency [Hz]')
plt.ylabel('$ \mathrm{V}^2 \, / \, (1 \mathrm{Hz})$')
plt.grid(which='both')
plt.show()
