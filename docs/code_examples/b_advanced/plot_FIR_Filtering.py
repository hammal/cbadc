"""
Digital Estimator as FIR Filter
===================================

"""

###############################################################################
# AS and DC Setup
# ------------------------------------------------
#

# Setup analog system and digital control
from cbadc.analog_system import AnalogSystem
from cbadc.digital_control import DigitalControl
from cbadc.digital_estimator import DigitalEstimator
import numpy as np
N = 6
M = N
A = [[0, 0, 0, 0, 0, 0], 
     [6250, 0, 0, 0, 0, 0], 
     [0, 6250, 0, 0, 0, 0], 
     [0, 0, 6250, 0, 0, 0],
     [0, 0, 0, 6250, 0, 0],
     [0, 0, 0, 0, 6250, 0]]
B = [[6250], [0], [0], [0], [0], [0]]
CT = [[0, 0, 0, 0, 0, 1]]
Gamma = [[-6250, 0, 0, 0, 0, 0], 
         [0, -6250, 0, 0, 0, 0], 
         [0, 0, -6250, 0, 0, 0],
         [0, 0, 0, -6250, 0, 0],
         [0, 0, 0, 0, -6250, 0],
         [0, 0, 0, 0, 0, -6250]]
Gamma_tildeT = [[1, 0, 0, 0, 0, 0], 
                [0, 1, 0, 0, 0, 0], 
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]]
T = 1.0/(2 * 6250)

analog_system = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
digital_control = DigitalControl(T, M)

# Summarize the analog system, digital control, and digital estimator.
print(analog_system, "\n")
print(digital_control)


###############################################################################
# FIR
# ----------------
#
import matplotlib.pyplot as plt
from cbadc.utilities import read_byte_stream_from_file, byte_stream_2_control_signal

eta2 = 1e4

from cbadc.digital_estimator import FIRFilter
byte_stream = read_byte_stream_from_file('sinusodial_simulation.adc', M)
control_signal_sequences = byte_stream_2_control_signal(byte_stream, M)

K1 = 150
K2 = 150
h_index = np.arange(-K1, K2)

digital_estimator = FIRFilter(control_signal_sequences, analog_system, digital_control, eta2, K1, K2)

impulse_response = 20 * np.log10(np.abs(np.array(digital_estimator.h[:,0,:])))

plt.figure()
for index in range(N):
    plt.plot(h_index, impulse_response[:, index], label=f"$h_{index + 1}[k]$")
plt.legend()
plt.title(f"For $\eta^2 = {10 * np.log10(eta2)}$ [dB]")
plt.xlabel("filter taps k")
plt.ylabel("$| h_\ell [k]|^2_2$ [dB]")
plt.xlim((-K1, K2))
plt.grid()


##############################################################################
# FIR eta
# ----------------
#
Eta2 = np.logspace(0, 7, 8)
K1 = 1000
K2 = 1000
h_index = np.arange(-K1, K2)

plt.figure()
for eta2 in Eta2:
    digital_estimator = FIRFilter(control_signal_sequences, analog_system, digital_control, eta2, K1, K2)
    impulse_response = 20 * np.log10(np.linalg.norm(np.array(digital_estimator.h[:,0,:]), axis=-1))
    plt.plot(h_index, impulse_response, label=f"$\eta^2 = {10 * np.log10(eta2)}$ [dB]")
plt.legend()
plt.xlabel("filter taps k")
plt.ylabel("$\| \mathbf{h} [k] \|^2_2$ [dB]")
plt.xlim((-K1, K2))
plt.grid()



###############################################################################
# FIR
# ----------------
#
byte_stream = read_byte_stream_from_file('sinusodial_simulation.adc', M)
control_signal_sequences = byte_stream_2_control_signal(byte_stream, M)

stop_after_number_of_iterations = 90000
u_hat = np.zeros(stop_after_number_of_iterations)
K1 = 1000
K2 = 1000 + 1
digital_estimator = FIRFilter(control_signal_sequences, analog_system, digital_control, eta2, K1, K2)
plt.figure()
plt.plot(np.arange(-K1, K2), 20 * np.log10(np.linalg.norm(np.array(digital_estimator.h), axis=-1).flatten()))
plt.title("Impulse response")
plt.xlabel("k")
plt.ylabel("$\| \mathbf{h} [k] \|^2_2$ [dB]")
plt.grid()

for index, u_hat_temp in enumerate(digital_estimator):
    u_hat[index] = u_hat_temp
   
plt.figure()
t = np.arange(u_hat.size)
plt.plot(t, u_hat)
plt.xlabel('$t / T$')
plt.ylabel('$\hat{u}(t)$')
plt.title("Estimated input signal")
plt.grid()
plt.xlim((0, 10000))
plt.tight_layout()

###############################################################################
# Plotting the PSD
# ----------------
#
# As is typical for delta-sigma modulators we often visualize the performance 
# of the estimate by plotting the power spectral density (PSD).
from cbadc.utilities import compute_power_spectral_density

f, psd = compute_power_spectral_density(u_hat, nperseg = u_hat.size )
plt.figure()
plt.semilogx(f, 10 * np.log10(psd))
plt.xlabel('frequency [Hz]')
plt.ylabel('$ \mathrm{V}^2 \, / \, \mathrm{Hz}$')