"""
=============
Downsampling
=============

In this tutorial we demonstrate how to configure the digital estimator
for downsampling.
"""
import numpy as np
from cbadc.digital_control import DigitalControl
from cbadc.analog_system import AnalogSystem
from cbadc.utilities import compute_power_spectral_density
import matplotlib.pyplot as plt
from cbadc.digital_estimator import FIRFilter
from cbadc.utilities import read_byte_stream_from_file, \
    byte_stream_2_control_signal

###############################################################################
# Setting up the Analog System and Digital Control
# ------------------------------------------------
#
# In this example, we assume that we have access to a control signal
# s[k] generated by the interactions of an analog system and digital control.
# Furthermore, we a chain-of-integrators converter with corresponding
# analog system and digital control.
#
# .. image:: /images/chainOfIntegratorsGeneral.svg
#    :width: 500
#    :align: center
#    :alt: The chain of integrators ADC.

# Setup analog system and digital control

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
Gamma_tildeT = np.eye(N)
T = 1.0/(2 * beta)

analog_system = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
digital_control = DigitalControl(T, M)

# Summarize the analog system, digital control, and digital estimator.
print(analog_system, "\n")
print(digital_control)


###############################################################################
# Loading Control Signal from File
# --------------------------------
#
# Next, we will load an actual control signal to demonstrate the digital
# estimator's capabilities. To this end, we will use the
# `sinusodial_simulation.adc` file that was produced in
# :doc:`./plot_b_simulate_a_control_bounded_adc`.
#
# The control signal file is encoded as raw binary data so to unpack it
# correctly we will use the :func:`cbadc.utilities.read_byte_stream_from_file`
# and :func:`cbadc.utilities.byte_stream_2_control_signal` functions.

byte_stream = read_byte_stream_from_file(
    '../a_getting_started/sinusodial_simulation.adc', M)
control_signal_sequences1 = byte_stream_2_control_signal(byte_stream, M)

byte_stream = read_byte_stream_from_file(
    '../a_getting_started/sinusodial_simulation.adc', M)
control_signal_sequences2 = byte_stream_2_control_signal(byte_stream, M)

###############################################################################
# Oversampling
# -------------
#

OSR = 64

omega_3dB = 2 * np.pi / (2 * T * OSR)


###############################################################################
# Oversampling = 1
# ----------------------------------------
#
# First we initialize our default estimator without a downsampling parameter
# which then defaults to 1, i.e., no downsampling.

# Set the bandwidth of the estimator
G_at_omega = np.linalg.norm(
    analog_system.transfer_function_matrix(np.array([omega_3dB])))
eta2 = G_at_omega**2
print(f"eta2 = {eta2}, {10 * np.log10(eta2)} [dB]")

# Set the filter size
L1 = 1 << 13
L2 = L1

# Instantiate the digital estimator.
digital_estimator_ref = FIRFilter(
    control_signal_sequences1, analog_system, digital_control, eta2, L1, L2)

print(digital_estimator_ref, "\n")


###############################################################################
# Visualize Estimator's Transfer Function
# ---------------------------------------
#

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
    np.array([omega_3dB]))[0]

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
# Next we repeat the initialization steps above but for a downsampled estimator

digital_estimator_dow = FIRFilter(
    control_signal_sequences2,
    analog_system,
    digital_control,
    eta2,
    L1,
    L2,
    downsample=OSR)

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

# compensate the built in L1 delay of FIR filter.
t = np.arange(-L1 + 1, size - L1 + 1)
t_down = np.arange(-(L1) // OSR, (size - L1) // OSR) * OSR + 1
plt.plot(t, u_hat_ref, label="$\hat{u}(t)$ Reference")
plt.plot(t_down, u_hat_dow, label="$\hat{u}(t)$ Downsampled")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimated input signal")
plt.grid(which='both')
plt.xlim((-50, 1000))
plt.tight_layout()

plt.figure()
u_hat_ref_clipped = u_hat_ref[(L1 + L2):]
u_hat_dow_clipped = u_hat_dow[(L1 + L2) // OSR:]
f_ref, psd_ref = compute_power_spectral_density(
    u_hat_ref_clipped)
f_dow, psd_dow = compute_power_spectral_density(
    u_hat_dow_clipped, fs=1.0/OSR)
plt.semilogx(f_ref, 10 * np.log10(psd_ref), label="$\hat{U}(f)$ Referefence")
plt.semilogx(f_dow, 10 * np.log10(psd_dow), label="$\hat{U}(f)$ Downsampled")
plt.legend()
plt.ylim((-200, 50))
plt.xlim((f_ref[1], f_ref[-1]))
plt.xlabel('frequency [Hz]')
plt.ylabel('$ \mathrm{V}^2 \, / \, (1 \mathrm{Hz})$')
plt.grid(which='both')
plt.show()

# sphinx_gallery_thumbnail_number = 2