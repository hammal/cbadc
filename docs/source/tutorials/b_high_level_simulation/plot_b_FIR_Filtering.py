"""
=================================
FIR Filter
=================================

We demonstrate how to set up the FIR filter implementation.

.. seealso::
    :doc:`./plot_c_downsample.rst` for a FIR decimation example.
"""
import matplotlib.pyplot as plt
import numpy as np
import cbadc

###############################################################################
# ---------------------------------------
# Analog System and Digital Control Setup
# ---------------------------------------
#
# To initialize a digital estimator, we need to specify which analog system and
# digital control are used. Here we default to the chain-of-integrators
# example.

N = 6
M = N
beta = 6250.0
rho = -1e-2
kappa = -1.0
A = [
    [beta * rho, 0, 0, 0, 0, 0],
    [beta, beta * rho, 0, 0, 0, 0],
    [0, beta, beta * rho, 0, 0, 0],
    [0, 0, beta, beta * rho, 0, 0],
    [0, 0, 0, beta, beta * rho, 0],
    [0, 0, 0, 0, beta, beta * rho],
]
B = [[beta], [0], [0], [0], [0], [0]]
CT = np.eye(N)
Gamma = [
    [kappa * beta, 0, 0, 0, 0, 0],
    [0, kappa * beta, 0, 0, 0, 0],
    [0, 0, kappa * beta, 0, 0, 0],
    [0, 0, 0, kappa * beta, 0, 0],
    [0, 0, 0, 0, kappa * beta, 0],
    [0, 0, 0, 0, 0, kappa * beta],
]
Gamma_tildeT = CT
T = 1.0 / (2 * beta)

analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
digital_control = cbadc.digital_control.DigitalControl(cbadc.analog_signal.Clock(T), M)

# Summarize the analog system, digital control, and digital estimator.
print(analog_system, "\n")
print(digital_control)

###############################################################################
# ----------------
# Impulse Response
# ----------------
#
# Next we instantiate a :py:class:`cbadc.digital_estimator.FIRFilter` and
# visualize its impulse responses.
#

# Choose an arbitrary eta2
eta2 = 1e6

# Instantiate digital estimator
K1 = 250
K2 = 250
digital_estimator = cbadc.digital_estimator.FIRFilter(
    analog_system, digital_control, eta2, K1, K2
)

# extract impulse response
impulse_response = np.abs(np.array(digital_estimator.h[0, :, :]))

# Visualize the impulse response
h_index = np.arange(-K1, K2)
fig, ax = plt.subplots(2)
for index in range(N):
    ax[0].plot(h_index, impulse_response[:, index], label=f"$h_{index + 1}[k]$")
    ax[1].semilogy(h_index, impulse_response[:, index], label=f"$h_{index + 1}[k]$")
ax[0].legend()
fig.suptitle(f"For $\eta^2 = {10 * np.log10(eta2)}$ [dB]")
ax[1].set_xlabel("filter tap k")
ax[0].set_ylabel("$| h_\ell [k]|$")
ax[1].set_ylabel("$| h_\ell [k]|$")
ax[0].set_xlim((-50, 50))
ax[0].grid(which="both")
ax[1].set_xlim((-K1, K2))
ax[1].grid(which="both")


##############################################################################
# -----------------------------------
# Impulse Response and :math:`\eta^2`
# -----------------------------------
#
# The rate at which the estimator's impulse response decays depends on our
# bandwidth parameter :math:`\eta^2`. Specifically, as we increase
# :math:`\eta^2` we typically seek a higher resolution at a smaller bandwidth.
# As expected, a more narrowband filter requires a longer impulse responses,
# or equivalently, has a slower decaying impulse response. Therefore,
# we require more filter taps for a given precision as we increase
# :math:`\eta^2`.
#
# Note that we plot only the first and largest filter coefficient
# :math:`h_1[k]`. The fact that :math:`h_1[k]` has the largest filter
# coefficients follows from the choice of analog system and digital control
# and does not necessarily generalize.
#
# We additionally plot the corresponding digital estimator transfer functions
# as a function of the bandwidth parameter :math:`\eta^2`.

Eta2 = np.logspace(0, 7, 8)
K1 = 1 << 8
K2 = 1 << 8
h_index = np.arange(-K1, K2)


plt.figure()
for eta2 in Eta2:
    digital_estimator = cbadc.digital_estimator.FIRFilter(
        analog_system, digital_control, eta2, K1, K2
    )
    plt.semilogy(
        np.arange(0, K2),
        np.abs(np.array(digital_estimator.h[0, :, 0]))[K2:],
        label=f"$\eta^2 \approx {10 * np.log10(eta2):0.1e}$ [dB]",
    )
plt.legend()
plt.xlabel("filter tap k")
plt.ylabel("$| h_1 [k] \|$")
plt.xlim((0, K2))
plt.grid(which="both")


# Plot corresponding transfer functions of estimator

# Logspace frequencies
frequencies = np.logspace(-3, 0, 100)
omega = 4 * np.pi * beta * frequencies

plt.figure()
for eta2 in Eta2:
    # Compute NTF
    digital_estimator = cbadc.digital_estimator.FIRFilter(
        analog_system, digital_control, eta2, K1, K2
    )

    ntf = digital_estimator.noise_transfer_function(omega)
    ntf_dB = 20 * np.log10(np.abs(ntf))

    # Compute STF
    stf = digital_estimator.signal_transfer_function(omega)
    stf_dB = 20 * np.log10(np.abs(stf.flatten()))

    plt.semilogx(frequencies, stf_dB, "--")
    color = plt.gca().lines[-1].get_color()
    plt.semilogx(
        frequencies,
        20 * np.log10(np.linalg.norm(ntf[0, :, :], axis=0)),
        color=color,
        label=f"$\eta^2 = {10 * np.log10(eta2)}$ [dB]",
    )

# Add labels and legends to figure
plt.legend(loc=4)
plt.grid(which="both")
plt.title("Signal (dashed) and noise (solid) transfer functions")
plt.xlabel("$\omega / (4 \pi \\beta ) $")
plt.ylabel("dB")
plt.xlim((1e-2, 0.5))
plt.ylim((-150, 3))
plt.gcf().tight_layout()

###############################################################################
# Filter length
# -------------
#
# We can confirm the intuition from the previous section by considering
# different number of filter taps for a given control signal sequence. Firstly,
# we once more plot the decay of the filter coefficients and recognize
# that an SNR of around 100 dB (normalized spectrum), would need around
# K1=K2=180 filter taps or more. This is confirmed by simulation as the
# harmonics in the estimated spectrum disappear for larger number of
# filter taps. Note also the reference used in the spectral plots which
# corresponds to the default implementation
# :py:class:`cbadc.digital_estimator.BatchEstimator` using a much
# longer lookahead than corresponding FIR filters implementations.
#
# The simulation is often a robust way of determining a minimum
# number of filter taps for a specific analog system, digital control,
# and :math:`\eta^2` combination.
#
# As is clear from the filter coefficient the different dimensions of the
# control signals :math:`\mathbf{s}[k]` can be filtered with FIR filters
# of different lengths as their decay varies.
#

filter_lengths = [10, 20, 40, 80, 120, 160, 180, 200, 220]

eta2 = 1e6

control_signal_sequences = [
    cbadc.utilities.byte_stream_2_control_signal(
        cbadc.utilities.read_byte_stream_from_file("sinusoidal_simulation.dat", M),
        M,
    )
    for _ in filter_lengths
]

stop_after_number_of_iterations = 1 << 16
u_hat = np.zeros(stop_after_number_of_iterations)


digital_estimators = [
    cbadc.digital_estimator.FIRFilter(
        analog_system,
        digital_control,
        eta2,
        filter_lengths[index],
        filter_lengths[index],
        stop_after_number_of_iterations=stop_after_number_of_iterations,
    )
    for index in range(len(filter_lengths))
]

for index, de in enumerate(digital_estimators):
    de(control_signal_sequences[index])


plt.figure()
for index in range(N):
    plt.semilogy(
        np.arange(0, filter_lengths[-1]),
        np.abs(np.array(digital_estimators[-1].h[0, :, :]))[
            filter_lengths[-1] :, index
        ],
        label=f"$h_{index + 1}[k]$",
    )
plt.legend()
plt.xlabel("filter tap k")
plt.ylabel("$| h_\ell [k]|$")
plt.xlim((0, filter_lengths[-1]))
plt.grid(which="both")

digital_estimators_ref = cbadc.digital_estimator.BatchEstimator(
    analog_system,
    digital_control,
    eta2,
    stop_after_number_of_iterations >> 2,
    1 << 14,
    stop_after_number_of_iterations=stop_after_number_of_iterations,
)

digital_estimators_ref(
    cbadc.utilities.byte_stream_2_control_signal(
        cbadc.utilities.read_byte_stream_from_file("sinusoidal_simulation.dat", M),
        M,
    )
)

for index, estimate in enumerate(digital_estimators_ref):
    u_hat[index] = estimate
f_ref, psd_ref = cbadc.utilities.compute_power_spectral_density(u_hat)

u_hats = []
plt.rcParams["figure.figsize"] = [6.40, 6.40 * 4]
fig, ax = plt.subplots(len(filter_lengths), 1)
for index_de in range(len(filter_lengths)):
    # Compute estimates for each estimator
    for index, estimate in enumerate(digital_estimators[index_de]):
        u_hat[index] = estimate
    u_hats.append(np.copy(u_hat))

    # Compute power spectral density
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat[filter_lengths[index_de] :]
    )

    # Plot the FIR filters
    color = next(ax[index_de]._get_lines.prop_cycler)["color"]

    ax[index_de].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[index_de].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)

    ax[index_de].semilogx(f_ref, 10 * np.log10(psd_ref), label="Reference", color="k")

    ax[index_de].semilogx(
        f, 10 * np.log10(psd), label=f"K1=K2={filter_lengths[index_de]}", color=color
    )

    ax[index_de].set_ylabel("$ \mathrm{V}^2 \, / \, \mathrm{Hz}$")

    ax[index_de].legend()
    ax[index_de].set_xlim((0.0002, 0.5))

ax[-1].set_xlabel("frequency [Hz]")
fig.tight_layout()

# Plot snapshot in time domain
plt.rcParams["figure.figsize"] = [6.40, 6.40]
plt.figure()
plt.title("Estimates in time domain")
for index in range(len(filter_lengths)):
    t_fir = np.arange(
        -filter_lengths[index] + 1,
        stop_after_number_of_iterations - filter_lengths[index] + 1,
    )
    plt.plot(t_fir, u_hats[index], label=f"K1=K2={filter_lengths[index]}")
plt.ylabel("$\hat{u}(t)$")
plt.xlim((64000, 64600))
plt.ylim((-0.6, 0.6))
plt.xlabel("$t / T$")
_ = plt.legend()

# sphinx_gallery_thumbnail_number = 4
