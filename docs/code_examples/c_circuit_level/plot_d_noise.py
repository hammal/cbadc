"""
=========================================
Thermal Noise Simulations
=========================================

In this tutorial we demonstrate how to account for non-idealities
in the design process.
"""

import cbadc
import numpy as np
import matplotlib.pyplot as plt
import copy

np.set_printoptions(precision=1)

###############################################################################
# Estimating the Noise Sensitivity of an Analog Frontend
# ------------------------------------------------------

N = 5
ENOB = 16
BW = 1e7
SNR_dB = cbadc.fom.enob_to_snr(ENOB)
snr = cbadc.fom.snr_from_dB(SNR_dB)
print(f"for an SNR: {SNR_dB:0.1f} dB")
input_signal_power = 1 / 2.0


###############################################################################
# Setting up the Analog Frontend and Estimators
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
analog_frontend = cbadc.synthesis.get_leap_frog(N=N, ENOB=ENOB, BW=BW)
analog_system = analog_frontend.analog_system
digital_control = analog_frontend.digital_control

analog_frontend_ref = cbadc.synthesis.get_leap_frog(N=N, ENOB=ENOB, BW=BW)
analog_system_ref = analog_frontend_ref.analog_system
digital_control_ref = analog_frontend_ref.digital_control

eta2 = (
    np.linalg.norm(analog_system.transfer_function_matrix(np.array([2 * np.pi * BW])))
    ** 2
)

K1 = 1 << 10
K2 = 1 << 10
digital_estimator = cbadc.digital_estimator.BatchEstimator(
    analog_system, digital_control, eta2, K1, K2
)
digital_estimator_ref = cbadc.digital_estimator.BatchEstimator(
    analog_system_ref, digital_control_ref, eta2, K1, K2
)

white_noise_sensitivies = digital_estimator.white_noise_sensitivities(
    (BW * 1e-5, BW), snr, input_power=input_signal_power, spectrum=True
)

print(
    f"These are the permissable white noise PSDs:\n{white_noise_sensitivies[0,:]} V^2/Hz\n{np.sqrt(white_noise_sensitivies[0,:])} V/sqrt(Hz)"
)

###############################################################################
# White Noise Limited Simulations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
plt.figure()
size = 1 << 14
u_hat = np.zeros(size)
u_hat_ref = np.zeros(size)

# input_signals = [cbadc.analog_signal.ConstantSignal(0.1)]
input_signals = [
    cbadc.analog_signal.Sinusoidal(1, 1 / (1024 * digital_control.clock.T))
]

for index, thermal_snr_limit in enumerate(
    np.array(
        [
            1e0,
            1e2,
            1e4,
        ]
    )
):
    noise_covariance_matrix = np.diag(
        white_noise_sensitivies[0, :] * BW * thermal_snr_limit
    )

    digital_control.reset()
    simulator = cbadc.simulator.FullSimulator(
        analog_system, digital_control, input_signals, cov_x=noise_covariance_matrix
    )
    digital_estimator(simulator)
    digital_estimator.warm_up(K1 + K2)
    for index in cbadc.utilities.show_status(range(size)):
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
        label=f"W_N_Limit={10 * np.log10(snr / thermal_snr_limit):.1f} dB, est_SNR={est_SNR:.1f} dB",
    )

# Reference Simulation

simulator_ref = cbadc.simulator.FullSimulator(
    analog_system_ref, digital_control_ref, input_signals
)
digital_estimator_ref(simulator_ref)
for index in cbadc.utilities.show_status(range(size)):
    u_hat_ref[index] = next(digital_estimator_ref)

u_hat_cut = u_hat_ref[K1 + K2 :]
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
    label=f"Ref, est_SNR={est_SNR:.1f} dB",
)
plt.title("Power spectral density of input estimate")
plt.xlabel('Hz')
plt.ylabel('$V^2$ / Hz dB')
plt.ylim((-210, -40))
plt.legend()
plt.grid(which="both")
plt.gcf().tight_layout()

###############################################################################
# Circuit Level Thermal Noise Sizing
# ----------------------------------
#
analog_frontend = cbadc.synthesis.get_leap_frog(N=N, ENOB=ENOB, BW=BW)
analog_system = analog_frontend.analog_system
digital_control = analog_frontend.digital_control

A_DC = 1e4
omega_p = 2 * np.pi * BW

digital_estimator = cbadc.digital_estimator.BatchEstimator(
    analog_system, digital_control, eta2, K1, K2
)

verilog_digital_control = cbadc.circuit_level.DigitalControl(
    copy.deepcopy(digital_control)
)

first_order_pole_op_amp_analog_system = (
    cbadc.circuit_level.AnalogSystemFirstOrderPoleOpAmp(
        BW=(BW * 1e-5, BW),
        target_snr=snr,
        digital_estimator=digital_estimator,
        A_DC=A_DC,
        omega_p=omega_p,
    )
)

first_order_pole_op_amp_analog_frontend = cbadc.circuit_level.AnalogFrontend(
    first_order_pole_op_amp_analog_system, verilog_digital_control
)

t_stop = digital_control.clock.T * (size + K1 + K2)
simulation_clock = cbadc.analog_signal.Clock(digital_control.clock.T)

first_order_pole_op_amp_testbench = cbadc.circuit_level.TestBench(
    first_order_pole_op_amp_analog_frontend,
    input_signals[0],
    simulation_clock,
    t_stop,
)

print(f"Capacitor values:\n{first_order_pole_op_amp_analog_system.C_diag}")
print(f"\n\nResistor network A: {first_order_pole_op_amp_analog_system._A_G_matrix}")
print(f"\n\nResistor network B: {first_order_pole_op_amp_analog_system._B_G_matrix}")
print(
    f"\n\nResistor network Gamma: {first_order_pole_op_amp_analog_system._Gamma_G_matrix}"
)

###############################################################################
# Plotting the Power Spectral Densities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

plt.figure()
digital_estimator = first_order_pole_op_amp_analog_frontend.get_estimator(
    cbadc.digital_estimator.FIRFilter, eta2, K1, K2
)
white_noise_sensitivies = digital_estimator.white_noise_sensitivities(
    (BW * 1e-5, BW), snr, input_power=input_signal_power, spectrum=True
)
print(
    f"These are the permissable white noise PSDs:\n{white_noise_sensitivies[0,:]} V^2/Hz\n{np.sqrt(white_noise_sensitivies[0,:])} V/sqrt(Hz)"
)
noise_covariance_matrix = np.diag(white_noise_sensitivies[0, :] * BW)
simulator = first_order_pole_op_amp_testbench.get_simulator(
    cbadc.simulator.SimulatorType.full_numerical, cov_x=noise_covariance_matrix
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
    label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB",
)


plt.title("Power spectral density of input estimate")
plt.xlabel('Hz')
plt.ylabel('$V^2$ / Hz dB')
plt.legend()
plt.grid(which="both")
# plt.xlim((f[0], f[-1]))
plt.gcf().tight_layout()

# sphinx_gallery_thumbnail_number = 1
