"""
Designing for a Target Specification
====================================

In this tutorial we investigate the
:py:func:`cbadc.specification.get_chain_of_integrator` and
:py:func:`cbadc.specification.get_leap_frog` convenience function
to quickly get initalized analog systems and digital control
for a given target specification.
"""
import cbadc
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Specifying a Target Performance
# -------------------------------
#
# Our target specification requires three things to be specified
#

ENOB = 12
N = 4
BW = 1e6

# Then a corresponding chain-of-integrators system is created as
as_coi, digital_control = cbadc.specification.get_chain_of_integrator(
    ENOB=ENOB, N=N, BW=BW, xi=2e-3 / np.pi
)
# where xi is a tuning parameter.

# Similarly, the leap-frog analog system is created as
analog_system, digital_control = cbadc.specification.get_leap_frog(
    ENOB=ENOB, N=N, BW=BW, xi=7e-2 / np.pi
)

# Comparing the transfer functions
BW_log = np.log10(BW)
frequencies = np.logspace(BW_log - 2, BW_log + 1, 500)
omegas = 2 * np.pi * frequencies


# Compute transfer functions for each frequency in frequencies
transfer_function_coi = as_coi.transfer_function_matrix(omegas)
transfer_function_lf = analog_system.transfer_function_matrix(omegas)

plt.semilogx(
    frequencies,
    20 * np.log10(np.linalg.norm(transfer_function_coi[:, 0, :], axis=0)),
    label="chain-of-integrators $ ||\mathbf{G}(\omega)||_2 $",
)
plt.semilogx(
    frequencies,
    20 * np.log10(np.linalg.norm(transfer_function_lf[:, 0, :], axis=0)),
    label="leap-frog $ ||\mathbf{G}(\omega)||_2 $",
)

# Add labels and legends to figure
plt.legend()
plt.grid(which="both")
plt.title("Analog system transfer function")
plt.xlabel("$f$ [Hz]")
plt.ylabel("dB")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()


###############################################################################
# -------------------------
# Comparing System Orders
# -------------------------
#
# We demonstrate how the filters evolve for different filter orders N
#
ENOB = 16
BW = 1e6
N = [2, 4, 8, 10, 12]

# Chain-of-integrators
plt.figure()
for n in N:
    analog_system, digital_control = cbadc.specification.get_chain_of_integrator(
        ENOB=ENOB, N=n, BW=BW, xi=2e-3 / np.pi
    )
    transfer_function = analog_system.transfer_function_matrix(omegas)

    plt.semilogx(
        frequencies,
        20 * np.log10(np.linalg.norm(transfer_function[:, 0, :], axis=0)),
        label=f"chain-of-integrators, N={n}, " + "$ ||\mathbf{G}(\omega)||_2 $",
    )
# Add labels and legends to figure
plt.legend()
plt.grid(which="both")
plt.title("Analog system transfer function")
plt.xlabel("$f$ [Hz]")
plt.ylabel("dB")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()

###############################################################################
# --------------------------------------------
# Performance Validation Chain-of-Integrators
# --------------------------------------------
#
# We confirm the results above by full system simulations
#

eta2 = (
    np.linalg.norm(analog_system.transfer_function_matrix(np.array([2 * np.pi * BW])))
    ** 2
)
K1 = 1 << 10
K2 = K1
amplitude = 1e0
phase = 0.0
offset = 0.0
size = 1 << 15
u_hat = np.zeros(size)

plt.figure()
for n in N[1:][::-1]:
    analog_system, digital_control = cbadc.specification.get_chain_of_integrator(
        ENOB=ENOB, N=n, BW=BW, xi=2e-3 / np.pi
    )
    digital_estimator = cbadc.digital_estimator.BatchEstimator(
        analog_system, digital_control, eta2, K1, K2
    )
    frequency = 1.0 / digital_control.clock.T
    while frequency > BW:
        frequency /= 2
    input_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)
    simulator = cbadc.simulator.get_simulator(
        analog_system, digital_control, [input_signal]
    )
    digital_estimator(simulator)
    for index in range(size):
        u_hat[index] = next(digital_estimator)
    u_hat_cut = u_hat[K1 + K2 :]
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_cut[:], fs=1 / digital_control.clock.T, nperseg=u_hat_cut.size
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / digital_control.clock.T
    )
    est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
    est_ENOB = cbadc.fom.snr_to_enob(est_SNR)
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"N={n}, OSR={1/(2 * digital_control.clock.T * BW):.0f}, est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB",
    )

plt.title("Power spectral density of input estimate")
plt.xlabel('Hz')
plt.ylabel('$V^2$ / Hz dB')
plt.legend()
plt.grid(which="both")
plt.gcf().tight_layout()


###############################################################################
# --------------------------------------------
# Performance Validation Leap-Frog
# --------------------------------------------
#

# Leap-frog
plt.figure()
for n in N:
    analog_system, digital_control = cbadc.specification.get_leap_frog(
        ENOB=ENOB, N=n, BW=BW, xi=7e-2 / np.pi
    )
    transfer_function = analog_system.transfer_function_matrix(omegas)

    plt.semilogx(
        frequencies,
        20 * np.log10(np.linalg.norm(transfer_function[:, 0, :], axis=0)),
        label=f"leap-frog, N={n}, " + "$ ||\mathbf{G}(\omega)||_2 $",
    )

# Add labels and legends to figure
plt.legend()
plt.grid(which="both")
plt.title("Analog system transfer function")
plt.xlabel("$f$ [Hz]")
plt.ylabel("dB")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()


plt.figure()
for n in N[1:][::-1]:
    analog_system, digital_control = cbadc.specification.get_leap_frog(
        ENOB=ENOB, N=n, BW=BW
    )
    digital_estimator = cbadc.digital_estimator.BatchEstimator(
        analog_system, digital_control, eta2, K1, K2
    )
    frequency = 1.0 / digital_control.clock.T
    while frequency > BW:
        frequency /= 2
    input_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)
    simulator = cbadc.simulator.get_simulator(
        analog_system, digital_control, [input_signal]
    )
    digital_estimator(simulator)
    for index in range(size):
        u_hat[index] = next(digital_estimator)
    u_hat_cut = u_hat[K1 + K2 :]
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_cut[:], fs=1 / digital_control.clock.T, nperseg=u_hat_cut.size
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / digital_control.clock.T
    )
    est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
    est_ENOB = cbadc.fom.snr_to_enob(est_SNR)
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=f"N={n}, OSR={1/(2 * digital_control.clock.T * BW):.0f}, est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB",
    )

plt.title("Power spectral density of input estimate")
plt.xlabel('Hz')
plt.ylabel('$V^2$ / Hz dB')
plt.legend()
plt.grid(which="both")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()

# sphinx_gallery_thumbnail_number = 5
