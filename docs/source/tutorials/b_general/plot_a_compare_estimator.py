"""
Compare Estimators
==================

In this tutorial we investigate different estimator implementation techniques
and compare their performance.
"""
import timeit
import matplotlib.pyplot as plt
import numpy as np
import cbadc

###############################################################################
# Analog System
# -------------
#
# We will commit to a leap-frog control-bounded analog system throughtout
# this tutorial.

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
rho_vec = -(omega_3dB ** 2) / beta * np.ones(N)
Gamma = np.diag(-beta_vec)
analog_system = cbadc.analog_system.LeapFrog(beta_vec, rho_vec, Gamma)

print(analog_system, "\n")

###############################################################################
# Analog Signal
# -------------
#
# We will also need an analog signal for conversion.
# In this tutorial we will use a Sinusoidal signal.

# Set the peak amplitude.
amplitude = 1.0
# Choose the sinusoidal frequency via an oversampling ratio (OSR).
frequency = 1.0 / (T * OSR * (1 << 0))

# We also specify a phase an offset these are hovewer optional.
phase = 0.0
offset = 0.0

# Instantiate the analog signal
analog_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)

print(analog_signal)


###############################################################################
# Simulating
# ----------
#
# Each estimator will require an independent stream of control signals.
# Therefore, we will next instantiate several digital controls and simulators.

# Set simulation precision parameters
atol = 1e-6
rtol = 1e-12

# Instantiate digital controls. We will need four of them as we will compare
# four different estimators.
digital_control1 = cbadc.digital_control.DigitalControl(T, M)
digital_control2 = cbadc.digital_control.DigitalControl(T, M)
digital_control3 = cbadc.digital_control.DigitalControl(T, M)
digital_control4 = cbadc.digital_control.DigitalControl(T, M)
print(digital_control1)

# Instantiate simulators.
simulator1 = cbadc.simulator.StateSpaceSimulator(
    analog_system,
    digital_control1,
    [analog_signal],
    atol=atol,
    rtol=rtol,
)
simulator2 = cbadc.simulator.StateSpaceSimulator(
    analog_system,
    digital_control2,
    [analog_signal],
    atol=atol,
    rtol=rtol,
)
simulator3 = cbadc.simulator.StateSpaceSimulator(
    analog_system,
    digital_control3,
    [analog_signal],
    atol=atol,
    rtol=rtol,
)
simulator4 = cbadc.simulator.StateSpaceSimulator(
    analog_system,
    digital_control4,
    [analog_signal],
    atol=atol,
    rtol=rtol,
)
print(simulator1)

###############################################################################
# Default, Quadratic Complexity, Estimator
# ----------------------------------------
#
# Next we instantiate the quadratic and default estimator
# :py:class:`cbadc.digital_estimator.DigitalEstimator`. Note that during its
# construction, the corresponding filter coefficients of the system will be
# computed. Therefore, this procedure could be computationally intense for a
# analog system with a large analog state order or equivalently for large
# number of independent digital controls.

# Set the bandwidth of the estimator
G_at_omega = np.linalg.norm(
    analog_system.transfer_function_matrix(np.array([omega_3dB]))
)
eta2 = G_at_omega ** 2
print(f"eta2 = {eta2}, {10 * np.log10(eta2)} [dB]")

# Set the batch size
K1 = 1 << 14
K2 = 1 << 14

# Instantiate the digital estimator (this is where the filter coefficients are
# computed).
digital_estimator_batch = cbadc.digital_estimator.DigitalEstimator(
    analog_system, digital_control1, eta2, K1, K2
)
digital_estimator_batch(simulator1)

print(digital_estimator_batch, "\n")


###############################################################################
# Visualize Estimator's Transfer Function (Same for Both)
# -------------------------------------------------------
#

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
    np.array([2 * np.pi * frequency])
)[0]

# Plot
plt.figure()
plt.semilogx(frequencies, stf_dB, label="$STF(\omega)$")
for n in range(N):
    plt.semilogx(frequencies, ntf_dB[0, n, :], label=f"$|NTF_{n+1}(\omega)|$")
plt.semilogx(
    frequencies,
    20 * np.log10(np.linalg.norm(ntf[0, :, :], axis=0)),
    "--",
    label="$ || NTF(\omega) ||_2 $",
)

# Add labels and legends to figure
plt.legend()
plt.grid(which="both")
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
# filter coefficients. This is an aid to determine if the lookahead and
# lookback sizes L1 and L2 are set sufficiently large.

# Determine lookback
L1 = K2
# Determine lookahead
L2 = K2
digital_estimator_fir = cbadc.digital_estimator.FIRFilter(
    analog_system, digital_control2, eta2, L1, L2
)

print(digital_estimator_fir, "\n")

digital_estimator_fir(simulator2)

# Next visualize the decay of the resulting filter coefficients.
h_index = np.arange(-L1, L2)

impulse_response = np.abs(np.array(digital_estimator_fir.h[0, :, :])) ** 2
impulse_response_dB = 10 * np.log10(impulse_response)

fig, ax = plt.subplots(2)
for index in range(N):
    ax[0].plot(h_index, impulse_response[:, index], label=f"$h_{index + 1}[k]$")
    ax[1].plot(h_index, impulse_response_dB[:, index], label=f"$h_{index + 1}[k]$")
ax[0].legend()
fig.suptitle(f"For $\eta^2 = {10 * np.log10(eta2)}$ [dB]")
ax[1].set_xlabel("filter taps k")
ax[0].set_ylabel("$| h_\ell [k]|^2_2$")
ax[1].set_ylabel("$| h_\ell [k]|^2_2$ [dB]")
ax[0].set_xlim((-50, 50))
ax[0].grid(which="both")
ax[1].set_xlim((-50, 500))
ax[1].set_ylim((-200, 0))
ax[1].grid(which="both")


###############################################################################
# IIR Filter Estimator
# --------------------
#
# The IIR filter is closely related to the FIR filter with the exception
# of an moving average computation.
# See :py:class:`cbadc.digital_estimator.IIRFilter` for more information.

# Determine lookahead
L2 = K2

digital_estimator_iir = cbadc.digital_estimator.IIRFilter(
    analog_system, digital_control3, eta2, L2
)

print(digital_estimator_iir, "\n")

digital_estimator_iir(simulator3)

###############################################################################
# Parallel Estimator
# ------------------------------
#
# Next we instantiate the parallel estimator
# :py:class:`cbadc.digital_estimator.ParallelEstimator`. The parallel estimator
# resembles the default estimator but diagonalizes the filter coefficients
# resulting in a more computationally more efficient filter that can be
# parallelized into independent filter operations.

# Instantiate the digital estimator (this is where the filter coefficients are
# computed).
digital_estimator_parallel = cbadc.digital_estimator.ParallelEstimator(
    analog_system, digital_control4, eta2, K1, K2
)

digital_estimator_parallel(simulator4)
print(digital_estimator_parallel, "\n")


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
u_hat_iir = np.zeros(size)
u_hat_parallel = np.zeros(size)
for index in range(size):
    u_hat_batch[index] = next(digital_estimator_batch)
    u_hat_fir[index] = next(digital_estimator_fir)
    u_hat_iir[index] = next(digital_estimator_iir)
    u_hat_parallel[index] = next(digital_estimator_parallel)

###############################################################################
# Visualizing Results
# -------------------
#
# Finally, we summarize the comparision by visualizing the resulting estimate
# in both time and frequency domain.

t = np.arange(size)
# compensate the built in L1 delay of FIR filter.
t_fir = np.arange(-L1 + 1, size - L1 + 1)
t_iir = np.arange(-L1 + 1, size - L1 + 1)
u = np.zeros_like(u_hat_batch)
for index, tt in enumerate(t):
    u[index] = analog_signal.evaluate(tt * T)
plt.plot(t, u_hat_batch, label="$\hat{u}(t)$ Batch")
plt.plot(t_fir, u_hat_fir, label="$\hat{u}(t)$ FIR")
plt.plot(t_iir, u_hat_iir, label="$\hat{u}(t)$ IIR")
plt.plot(t, u_hat_parallel, label="$\hat{u}(t)$ Parallel")
plt.plot(t, stf_at_omega * u, label="$\mathrm{STF}(2 \pi f_u) * u(t)$")
plt.xlabel("$t / T$")
plt.legend()
plt.title("Estimated input signal")
plt.grid(which="both")
plt.xlim((-100, 500))
plt.tight_layout()

plt.figure()
plt.plot(t, u_hat_batch, label="$\hat{u}(t)$ Batch")
plt.plot(t_fir, u_hat_fir, label="$\hat{u}(t)$ FIR")
plt.plot(t_iir, u_hat_iir, label="$\hat{u}(t)$ IIR")
plt.plot(t, u_hat_parallel, label="$\hat{u}(t)$ Parallel")
plt.plot(t, stf_at_omega * u, label="$\mathrm{STF}(2 \pi f_u) * u(t)$")
plt.xlabel("$t / T$")
plt.legend()
plt.title("Estimated input signal")
plt.grid(which="both")
plt.xlim((t_fir[-1] - 50, t_fir[-1]))
plt.tight_layout()

plt.figure()
plt.plot(t, u_hat_batch, label="$\hat{u}(t)$ Batch")
plt.plot(t_fir, u_hat_fir, label="$\hat{u}(t)$ FIR")
plt.plot(t_iir, u_hat_iir, label="$\hat{u}(t)$ IIR")
plt.plot(t, u_hat_parallel, label="$\hat{u}(t)$ Parallel")
plt.plot(t, stf_at_omega * u, label="$\mathrm{STF}(2 \pi f_u) * u(t)$")
plt.xlabel("$t / T$")
plt.legend()
plt.title("Estimated input signal")
plt.grid(which="both")
# plt.xlim((t_fir[0], t[-1]))
plt.xlim(((1 << 14) - 100, (1 << 14) + 100))
plt.tight_layout()

batch_error = stf_at_omega * u - u_hat_batch
fir_error = stf_at_omega * u[: (u.size - L1 + 1)] - u_hat_fir[(L1 - 1) :]
iir_error = stf_at_omega * u[: (u.size - L1 + 1)] - u_hat_iir[(L1 - 1) :]
parallel_error = stf_at_omega * u - u_hat_parallel
plt.figure()
plt.plot(t, batch_error, label="$|\mathrm{STF}(2 \pi f_u) * u(t) - \hat{u}(t)|$ Batch")
plt.plot(
    t[: (u.size - L1 + 1)],
    fir_error,
    label="$|\mathrm{STF}(2 \pi f_u) * u(t) - \hat{u}(t)|$ FIR",
)
plt.plot(
    t[: (u.size - L1 + 1)],
    iir_error,
    label="$|\mathrm{STF}(2 \pi f_u) * u(t) - \hat{u}(t)|$ IIR",
)
plt.plot(
    t, parallel_error, label="$|\mathrm{STF}(2 \pi f_u) * u(t) - \hat{u}(t)|$ Parallel"
)
plt.xlabel("$t / T$")
plt.xlim(((1 << 14) - 100, (1 << 14) + 100))
plt.ylim((-0.00001, 0.00001))
plt.legend()
plt.title("Estimation error")
plt.grid(which="both")
plt.tight_layout()


print(f"Average Batch Error: {np.linalg.norm(batch_error) / batch_error.size}")
print(f"Average FIR Error: {np.linalg.norm(fir_error) / fir_error.size}")
print(f"Average IIR Error: {np.linalg.norm(iir_error) / iir_error.size}")
print(
    f"""Average Parallel Error: { np.linalg.norm(parallel_error)/
    parallel_error.size}"""
)

plt.figure()
u_hat_batch_clipped = u_hat_batch[(K1 + K2) : -K2]
u_hat_fir_clipped = u_hat_fir[(L1 + L2) :]
u_hat_iir_clipped = u_hat_iir[(K1 + K2) : -K2]
u_hat_parallel_clipped = u_hat_parallel[(K1 + K2) : -K2]
u_clipped = stf_at_omega * u
f_batch, psd_batch = cbadc.utilities.compute_power_spectral_density(u_hat_batch_clipped)
f_fir, psd_fir = cbadc.utilities.compute_power_spectral_density(u_hat_fir_clipped)
f_iir, psd_iir = cbadc.utilities.compute_power_spectral_density(u_hat_iir_clipped)
f_parallel, psd_parallel = cbadc.utilities.compute_power_spectral_density(
    u_hat_parallel_clipped
)
f_ref, psd_ref = cbadc.utilities.compute_power_spectral_density(u_clipped)
plt.semilogx(f_ref, 10 * np.log10(psd_ref), label="$\mathrm{STF}(2 \pi f_u) * U(f)$")
plt.semilogx(f_batch, 10 * np.log10(psd_batch), label="$\hat{U}(f)$ Batch")
plt.semilogx(f_fir, 10 * np.log10(psd_fir), label="$\hat{U}(f)$ FIR")
plt.semilogx(f_iir, 10 * np.log10(psd_iir), label="$\hat{U}(f)$ IIR")
plt.semilogx(f_parallel, 10 * np.log10(psd_parallel), label="$\hat{U}(f)$ Parallel")
plt.legend()
plt.ylim((-200, 50))
plt.xlim((f_fir[1], f_fir[-1]))
plt.xlabel("frequency [Hz]")
plt.ylabel("$ \mathrm{V}^2 \, / \, (1 \mathrm{Hz})$")
plt.grid(which="both")
plt.show()


###############################################################################
# Compute Time
# ------------
#
# Compare the execution time of each estimator


def dummy_input_control_signal():
    while True:
        yield np.zeros(M, dtype=np.int8)


def iterate_number_of_times(iterator, number_of_times):
    for _ in range(number_of_times):
        _ = next(iterator)


digital_estimator_batch = cbadc.digital_estimator.DigitalEstimator(
    analog_system, digital_control1, eta2, K1, K2
)
digital_estimator_fir = cbadc.digital_estimator.FIRFilter(
    analog_system, digital_control2, eta2, L1, L2
)
digital_estimator_parallel = cbadc.digital_estimator.ParallelEstimator(
    analog_system, digital_control4, eta2, K1, K2
)
digital_estimator_iir = cbadc.digital_estimator.IIRFilter(
    analog_system, digital_control3, eta2, L2
)

digital_estimator_batch(dummy_input_control_signal())
digital_estimator_fir(dummy_input_control_signal())
digital_estimator_parallel(dummy_input_control_signal())
digital_estimator_iir(dummy_input_control_signal())

length = 1 << 14
repetitions = 10

print("Digital Estimator:")
print(
    timeit.timeit(
        lambda: iterate_number_of_times(digital_estimator_batch, length),
        number=repetitions,
    ),
    "sec \n",
)

print("FIR Estimator:")
print(
    timeit.timeit(
        lambda: iterate_number_of_times(digital_estimator_fir, length),
        number=repetitions,
    ),
    "sec \n",
)

print("IIR Estimator:")
print(
    timeit.timeit(
        lambda: iterate_number_of_times(digital_estimator_iir, length),
        number=repetitions,
    ),
    "sec \n",
)

print("Parallel Estimator:")
print(
    timeit.timeit(
        lambda: iterate_number_of_times(digital_estimator_parallel, length),
        number=repetitions,
    ),
    "sec \n",
)

# sphinx_gallery_thumbnail_number = 7
