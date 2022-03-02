"""
Calibrating an Unknown Op-Amp Pole from Data
============================================

"""
import cbadc
import numpy as np
import matplotlib.pyplot as plt
import copy

###############################################################################
# Setup proxy simulators from numpy data
# --------------------------------------
#
specification_false = cbadc.utilities.pickle_load('op-amp_data/AS_PARAMS_FALSE.dict')
specification_ref = cbadc.utilities.pickle_load('op-amp_data/AS_PARAMS_TRUE.dict')
T = specification_ref['T']
BW = specification_ref['fc']
N = specification_false['N']
M = N + 1

ctrl_bits_test_uncalibrated = iter(np.load('op-amp_data/s_test.npy'))
ctrl_bits_test_ref = iter(np.load('op-amp_data/s_test.npy'))
ctrl_bits_test_cal = cbadc.simulator.NumpySimulator('op-amp_data/s_test.npy')
ctrl_bits_train = cbadc.simulator.NumpySimulator('op-amp_data/s_train.npy')

###############################################################################
# Instantiate Analog Systems, Digital Controls, and Nominal FIR Filters
# ----------------------------------------------------------------------
#

analog_system = cbadc.analog_system.AnalogSystem(
    specification_false['A'],
    specification_false['B'],
    np.eye(specification_false['N']),
    specification_false['Gamma'],
    specification_false['Gamma'].transpose(),
)
analog_system_ref = cbadc.analog_system.AnalogSystem(
    specification_ref['A'],
    specification_ref['B'],
    np.eye(specification_ref['N']),
    specification_ref['Gamma'],
    specification_ref['Gamma'].transpose(),
)

digital_control = cbadc.digital_control.DigitalControl(
    cbadc.analog_signal.Clock(specification_false['T']), M
)
digital_control_ref = cbadc.digital_control.DigitalControl(
    cbadc.analog_signal.Clock(specification_ref['T']), M
)

eta2 = (
    np.linalg.norm(
        analog_system_ref.transfer_function_matrix(
            np.array([2 * np.pi * specification_ref['fc']])
        )
    )
    ** 2
)
K1 = 1 << 8
K2 = K1

fir_filter = cbadc.digital_estimator.FIRFilter(
    analog_system, digital_control, eta2, K1, K2
)

fir_filter_ref = cbadc.digital_estimator.FIRFilter(
    analog_system_ref, digital_control_ref, eta2, K1, K2
)

###############################################################################
# Setup testing simulations for verification
# ---------------------------------------------
#

# Ref
uncalibrated_filter = copy.deepcopy(fir_filter)
uncalibrated_filter(ctrl_bits_test_uncalibrated)


fir_filter_ref(ctrl_bits_test_ref)


size = 1 << 14
u_hat_ref = np.zeros(size)
u_hat_uncalibrated = np.zeros(size)
for index in range(size):
    u_hat_ref[index] = next(fir_filter_ref)
    u_hat_uncalibrated[index] = next(uncalibrated_filter)
u_hat_cut_ref = u_hat_ref[K1 + K2 :]
u_hat_cut_uncalibrated = u_hat_uncalibrated[K1 + K2 :]

###############################################################################
# Instantiate adaptive filter and set buffer training data size
# -------------------------------------------------------------
#
training_data_size = 1 << 15 - 1
adaptive_filter = cbadc.digital_estimator.AdaptiveFilter(
    fir_filter, 0, training_data_size
)

###############################################################################
# Instantiate the training instance
# ----------------------------------
#
calibrator = cbadc.digital_calibration.Calibration(
    adaptive_filter, ctrl_bits_train, ctrl_bits_test_cal
)

###############################################################################
# Train adaptive filter
# ----------------------------------
#
# this step could potentially be repeated many times
#
epochs = 1 << 16
step_size = lambda x: 1e-1 / ((1 + x ** (0.01)))
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
    fs=1 / digital_control.clock.T,
    nperseg=u_hat_cut_uncalibrated.size,
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

# sphinx_gallery_thumbnail_number = 2
