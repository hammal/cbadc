"""
Calibrating Digital Estimator for Mismatch
==========================================

"""
import cbadc
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Design a nominal and mismatched analog system
# ---------------------------------------------
#

# Fix system parameters such as effective number of bits
ENOB = 13
# System order
N = 4
# Bandwidth
BW = 1e6

random_control_scale = 1e-1

# Instantiate leap-frog analog system is created as
analog_system, digital_control = cbadc.specification.get_leap_frog(
    ENOB=ENOB, N=N, BW=BW
)
analog_system_ref, digital_control_ref = cbadc.specification.get_leap_frog(
    ENOB=ENOB, N=N, BW=BW
)

ref_vector = np.zeros((N, 1))
ref_vector[0] = analog_system.Gamma[0, 0] * random_control_scale

analog_system = cbadc.analog_system.AnalogSystem(
    analog_system.A,
    analog_system.B,
    analog_system.CT,
    np.hstack((ref_vector, analog_system.Gamma)),
    analog_system.Gamma_tildeT,
)

mismatch = 1.1

analog_system_distorted = cbadc.analog_system.AnalogSystem(
    analog_system.A * mismatch,
    analog_system.B * mismatch,
    analog_system.CT,
    analog_system.Gamma * mismatch,
    analog_system.Gamma_tildeT,
)

###############################################################################
# Setup digital control and nominal estimator
# ---------------------------------------------
#

eta2 = (
    np.linalg.norm(analog_system.transfer_function_matrix(np.array([2 * np.pi * BW])))
    ** 2
)
K1 = 1 << 10
K2 = K1

calibration_control = cbadc.digital_control.DitherControl(1, digital_control)

fir_filter = cbadc.digital_estimator.FIRFilter(
    analog_system, calibration_control, eta2, K1, K2
)


###############################################################################
# Setup testing simulations for verification
# ---------------------------------------------
#

# Ref
amplitude = 5e-1
frequency = 1.0 / digital_control.clock.T
while frequency > BW:
    frequency /= 2
input_signal_ref = cbadc.analog_signal.Sinusoidal(amplitude, frequency)

## A version for uncalibrated
uncalibrated_control = cbadc.digital_control.DitherControl(
    1, cbadc.digital_control.DigitalControl(digital_control.clock, N)
)
uncalibrated_sim = cbadc.simulator.get_simulator(
    analog_system_distorted,
    uncalibrated_control,
    [input_signal_ref],
)
uncalibrated_filter = cbadc.digital_estimator.FIRFilter(
    analog_system, uncalibrated_control, eta2, K1, K2
)
uncalibrated_filter(uncalibrated_sim)

## A version assuming perfect system knowledge
simulator_ver_ref = cbadc.simulator.get_simulator(
    analog_system_ref,
    digital_control_ref,
    [input_signal_ref],
)
ref_filter = cbadc.digital_estimator.FIRFilter(
    analog_system_ref, digital_control_ref, eta2, K1, K2
)
ref_filter(simulator_ver_ref)


# Generate verification estimates
size = 1 << 14
u_hat_ref = np.zeros(size)
u_hat_uncalibrated = np.zeros(size)
for index in range(size):
    u_hat_ref[index] = next(ref_filter)
    u_hat_uncalibrated[index] = next(uncalibrated_filter)
u_hat_cut_ref = u_hat_ref[K1 + K2 :]
u_hat_cut_uncalibrated = u_hat_uncalibrated[K1 + K2 :]


###############################################################################
# Generate a testing simulator for calibration
# ---------------------------------------------
#
analog_signal_cal = cbadc.analog_signal.ConstantSignal(0)
simulator_cal = cbadc.simulator.get_simulator(
    analog_system_distorted,
    calibration_control,
    [analog_signal_cal],
)
simulator_test = cbadc.simulator.get_simulator(
    analog_system_distorted,
    calibration_control,
    [input_signal_ref],
)

###############################################################################
# Instantiate adaptive filter and set buffer training data size
# -------------------------------------------------------------
#
training_data_size = 1 << 15
adaptive_filter = cbadc.digital_estimator.AdaptiveFilter(
    fir_filter, 0, training_data_size
)

###############################################################################
# Instantiate the training instance
# ----------------------------------
#
calibrator = cbadc.digital_calibration.Calibration(
    adaptive_filter, simulator_cal, simulator_test
)

###############################################################################
# Train adaptive filter
# ----------------------------------
#
# this step could potentially be repeated many times
#
epochs = 1 << 16


def step_size(x):
    return 1e-1 / ((1 + x ** (0.01)))


batch_size = 1 << 6

calibrator.compute_step_size_template()
calibrator.train(epochs, step_size, batch_size, stochastic_delay=0)

###############################################################################
# Print training statistics and plot training error
# -------------------------------------------------
#
# this step could potentially be repeated many times
#
print(calibrator.stats())
calibrator.plot_test_accuracy()

###############################################################################
# Extract testing estimate
# -------------------------
#
u_hat = calibrator.test(size)

###############################################################################
# Visualise PSD of testing data
# -----------------------------
#

# uncalibrated
f, psd = cbadc.utilities.compute_power_spectral_density(
    u_hat_cut_uncalibrated[:],
    fs=1 / uncalibrated_control.clock.T,
    nperseg=u_hat_cut_uncalibrated.size,
)
signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
noise_index = np.ones(psd.size, dtype=bool)
noise_index[signal_index] = False
noise_index[f < (BW * 1e-2)] = False
noise_index[f > BW] = False
fom = cbadc.utilities.snr_spectrum_computation_extended(
    psd, signal_index, noise_index, fs=1 / uncalibrated_control.clock.T
)
est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
est_ENOB = cbadc.fom.snr_to_enob(est_SNR)
plt.semilogx(
    f,
    10 * np.log10(np.abs(psd)),
    label=f"Uncalibrated, est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB",
)

# Ref
f, psd = cbadc.utilities.compute_power_spectral_density(
    u_hat_cut_ref[:], fs=1 / digital_control.clock.T, nperseg=u_hat_cut_ref.size
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
    label=f"Ref, est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB",
)

# Calibrated Est
f, psd = cbadc.utilities.compute_power_spectral_density(
    u_hat[:], fs=1 / digital_control.clock.T, nperseg=u_hat.size
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
    label=f"Calibrated, est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB",
)

plt.title("Power spectral density of input estimate")
plt.xlabel('Hz')
plt.ylabel('$V^2$ / Hz dB')
plt.legend()
plt.grid(which="both")
plt.gcf().tight_layout()

# sphinx_gallery_thumbnail_number = 1
