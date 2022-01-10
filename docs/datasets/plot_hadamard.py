"""
Loading a Hadamard Ramp Simulation
==================================

Builds on...
"""
import cbadc
import cbadc.datasets.hadamard
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Create a Simulation Wrapper
# ----------------------------
#
# We load the PCB A prototype by instantiating
# the wrapper class as

simulation_wrapper = cbadc.datasets.hadamard.HadamardPCB('B')

###############################################################################
# Load a specific simulation
# ---------------------------
#
# In this case we load
# :py:func:`cbadc.datasets.hadamard.HadamardPCB.simulation_ramp_1_B`
# simulation by invoking

(
    control_signal,
    ideal_control_signal,
    simulator,
    size,
) = simulation_wrapper.simulation_ramp_1_B()

size = 1 << 12
###############################################################################
# Configure a Digital Estimator
# -----------------------------
#
eta2 = 1e5
L1 = 1 << 10
L2 = 1 << 10
OSR = 1 << 5


digital_estimator = cbadc.digital_estimator.FIRFilter(
    simulator.analog_system, simulator.digital_control, eta2, L1, L2, downsample=OSR
)

print(digital_estimator, "\n")

digital_estimator(control_signal)

###############################################################################
# Post filtering with FIR
# ------------------------------------
#

numtaps = 1 << 10
f_cutoff = 1.0 / OSR
fir_filter = scipy.signal.firwin(numtaps, f_cutoff)

digital_estimator.convolve((fir_filter))

###############################################################################
# Filtering Estimate
# --------------------
#

u_hat = np.zeros(size // OSR)
for index in cbadc.utilities.show_status(range(size // OSR)):
    u_hat[index] = next(digital_estimator)

###############################################################################
# Visualize Estimate
# --------------------
#

t = np.arange(size // OSR) * OSR
plt.plot(t, u_hat, label="$\hat{u}(t)$")
plt.xlabel('$t / T$')
plt.legend()
plt.title("Estimated input signal")
plt.grid(which='both')
# offset = (L1 + L2) * 4
# plt.xlim((offset, offset + 1000))
plt.ylim((-0.6, 0.6))
plt.tight_layout()

###############################################################################
# Visualize Estimate Spectrum
# ---------------------------
#

plt.figure()
u_hat_clipped = u_hat[(L1 + L2) // OSR :]
freq, psd = cbadc.utilities.compute_power_spectral_density(
    u_hat_clipped, fs=1.0 / (simulator.digital_control.clock.T * OSR)
)
plt.semilogx(freq, 10 * np.log10(psd), label="$\hat{U}(f)$")
plt.legend()
plt.ylim((-300, 50))
# plt.xlim((f_ref[1], f_ref[-1]))
plt.xlabel('$f$ [Hz]')
plt.ylabel('$ \mathrm{V}^2 \, / \, (1 \mathrm{Hz})$')
plt.grid(which='both')
plt.show()
