"""
===============================
Digital Estimator as FIR Filter
===============================

We demonstrate how to set up the FIR filter implementation.
"""

###############################################################################
# ---------------------------------------
# Analog System and Digital Control Setup
# ---------------------------------------
#
# To initialize a digital estimator, we need to specify which analog system and
# digital control are used. Here we default to the chain-of-integrators
# example.

from cbadc.analog_system import AnalogSystem
from cbadc.digital_control import DigitalControl
from cbadc.digital_estimator import DigitalEstimator
import numpy as np
N = 6
M = N
beta = 6250.
rho = - beta * 1e-2
A = [[rho, 0, 0, 0, 0, 0],
     [beta, rho, 0, 0, 0, 0],
     [0, beta, rho, 0, 0, 0],
     [0, 0, beta, rho, 0, 0],
     [0, 0, 0, beta, rho, 0],
     [0, 0, 0, 0, beta, rho]]
B = [[beta], [0], [0], [0], [0], [0]]
CT = np.eye(N)
Gamma = [[-beta, 0, 0, 0, 0, 0],
         [0, -beta, 0, 0, 0, 0],
         [0, 0, -beta, 0, 0, 0],
         [0, 0, 0, -beta, 0, 0],
         [0, 0, 0, 0, -beta, 0],
         [0, 0, 0, 0, 0, -beta]]
Gamma_tildeT = CT
T = 1.0/(2 * beta)

analog_system = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
digital_control = DigitalControl(T, M)

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
import matplotlib.pyplot as plt
from cbadc.utilities import read_byte_stream_from_file, byte_stream_2_control_signal

eta2 = 1e5

from cbadc.digital_estimator import FIRFilter
byte_stream = read_byte_stream_from_file('sinusodial_simulation.adc', M)
control_signal_sequences = byte_stream_2_control_signal(byte_stream, M)

K1 = 250
K2 = 250
h_index = np.arange(-K1, K2)

digital_estimator = FIRFilter(
    control_signal_sequences, analog_system, digital_control, eta2, K1, K2)
impulse_response = np.abs(np.array(digital_estimator.h[:, 0, :])) ** 2
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
ax[1].set_xlim((-K1, K2))
ax[1].grid(which='both')

###############################################################################
# Transfer Function
# -----------------
#
# Additionally, we plot the corresponding transfer functions of the estimator.
#

# Logspace frequencies
frequencies = np.logspace(-3, 0, 100)
omega = 4 * np.pi * beta * frequencies

# Compute NTF
ntf = digital_estimator.noise_transfer_function(omega)
ntf_dB = 20 * np.log10(np.abs(ntf))

# Compute STF
stf = digital_estimator.signal_transfer_function(omega)
stf_dB = 20 * np.log10(np.abs(stf.flatten()))

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
plt.xlim((frequencies[10], frequencies[-1]))
plt.ylim((-300, 10))
plt.gcf().tight_layout()


##############################################################################
# -----------------------------------
# Impulse Response and :math:`\eta^2`
# -----------------------------------
#
# The rate at which the estimator's impulse response decays depends on our
# bandwidth parameter :math:`\eta^2`. Specifically, as we increase
# :math:`\eta^2` we seek a higher resolution at a smaller bandwidth. As
# expected, decreasing the bandwidth requires 'more' filtering and, therefore,
# a slower decaying impulse response. Equivalently, we require more filter taps
# for a given precision as we increase :math:`\eta^2`.
Eta2 = np.logspace(0, 7, 8)
K1 = 250
K2 = 250
h_index = np.arange(-K1, K2)

plt.figure()
for eta2 in Eta2:
    digital_estimator = FIRFilter(
        control_signal_sequences, analog_system, digital_control, eta2, K1, K2)
    impulse_response = 20 * \
        np.log10(np.linalg.norm(
            np.array(digital_estimator.h[:, 0, :]), axis=-1))
    plt.plot(h_index, impulse_response,
             label=f"$\eta^2 = {20 * np.log10(eta2)}$ [dB]")
plt.legend()
plt.xlabel("filter taps k")
plt.ylabel("$\| \mathbf{h} [k] \|^2_2$ [dB]")
plt.xlim((-K1, K2))
plt.grid(which="both")

###############################################################################
# Filter length
# -------------
#
# We can confirm the intuition from the previous section by considering
# a different number of filter taps for a given simulation. Specifically, we
# FIRFilter parametrizations.
#
# Conducting such a simulation is a good way of determining a minimum number
# of filter taps for a specific analog system, digital control, and :math:`\eta^2`
# combination.
from cbadc.utilities import compute_power_spectral_density

filter_lengths = [1, 1 << 4, 1 << 6, 1 << 8]
print(f"filter_lengths: {filter_lengths}")

eta2 = 1e6

control_signal_sequences = [byte_stream_2_control_signal(read_byte_stream_from_file(
    '../a_getting_started/sinusodial_simulation.adc', M), M) for _ in filter_lengths]

stop_after_number_of_iterations = 1 << 16
u_hat = np.zeros(stop_after_number_of_iterations)
digital_estimators = [FIRFilter(
    cs,
    analog_system,
    digital_control,
    eta2,
    filter_lengths[index],
    filter_lengths[index],
    stop_after_number_of_iterations=stop_after_number_of_iterations
) for index, cs in enumerate(control_signal_sequences)]

fig_frequency_spectrum = 4
fig_time_domain = 5
for index_de, de in enumerate(digital_estimators):
    # Print the estimator configuration
    print(de)
    for index, estimate in enumerate(de):
        u_hat[index] = estimate
    f, psd = compute_power_spectral_density(u_hat[filter_lengths[index_de]:])
    plt.figure(fig_frequency_spectrum)
    plt.semilogx(f, 10 * np.log10(psd),
                 label=f'K1=K2={filter_lengths[index_de]}')
    plt.figure(fig_time_domain)
    t_fir = np.arange(-filter_lengths[index_de] + 1,
                      stop_after_number_of_iterations - filter_lengths[index_de] + 1)
    plt.plot(t_fir, u_hat, label=f'K1=K2={filter_lengths[index_de]}')

plt.xlabel('$t / T$')
plt.ylabel('$\hat{u}(t)$')
plt.legend()
plt.title("Estimated input signal")
plt.grid(which='both')
plt.xlim(stop_after_number_of_iterations - 501,
         stop_after_number_of_iterations - 1)
plt.tight_layout()

plt.figure(fig_frequency_spectrum)
plt.legend()
plt.xlabel('frequency [Hz]')
plt.ylabel('$ \mathrm{V}^2 \, / \, \mathrm{Hz}$')
plt.xlim((f[1], f[-1]))
plt.grid(which="both")
