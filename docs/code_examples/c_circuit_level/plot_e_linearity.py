"""
=========================================
Linearity Simulations
=========================================

In this tutorial we demonstrate how to account for non-idealities
in the design process.
"""
import cbadc
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=1)

###############################################################################
# Simulate with non-linearities
# ------------------------------------------------------

N = 5
ENOB = 12
BW = 1e7
SNR_dB = cbadc.fom.enob_to_snr(ENOB)

size = 1 << 14
u_hat = np.zeros(size)
u_hat_ref = np.zeros(size)

analog_frontend = cbadc.synthesis.get_leap_frog(N=N, ENOB=ENOB, BW=BW)
analog_frontend_ref = cbadc.synthesis.get_leap_frog(N=N, ENOB=ENOB, BW=BW)

eta2 = (
    np.linalg.norm(
        analog_frontend.analog_system.transfer_function_matrix(
            np.array([2 * np.pi * BW])
        )
    )
    ** 2
)
K1 = 1 << 10
K2 = 1 << 10

digital_estimator = cbadc.digital_estimator.BatchEstimator(
    analog_frontend.analog_system, analog_frontend.digital_control, eta2, K1, K2
)
digital_estimator_ref = cbadc.digital_estimator.BatchEstimator(
    analog_frontend_ref.analog_system, analog_frontend_ref.digital_control, eta2, K1, K2
)

###############################################################################
# Specify non-linearities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The non-linearities are specified via the Taylor expansion as
#
# :math:`\dot{\mathbf{x}}(t) = f(\mathbf{x}, t) + \frac{f'(\mathbf{x}_0,t)}{1!}(\mathbf{x} - \mathbf{x}_0) + \frac{f''(\mathbf{x}_0,t)}{2!}(\mathbf{x} - \mathbf{x}_0)^2 + ...`
#
# where
#
# :math:`f(\mathbf{x}, t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`
#
# :math:`f^{\ell}(\mathbf{x}, t)` referres to the :math:`\ell`-th derivative of :math:`f(\mathbf{x}, t)`
# with respect to :math:`\mathbf{x}(t)` and :math:`\mathbf{x}_0` is an offset vector.
#
# in other words by specifying the :math:`f^{\ell}(\mathbf{x}_0,t)` column vectors we can simulate non-linearities.

beta = 1 / (2 * analog_frontend.digital_control.clock.T)
function_derivatives = np.array(
    [
        [1e-2 / beta, 1e-2 / (beta**2), 1e-2 / (beta**3), 1e-2 / (beta**4)],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

###############################################################################
# Simulating
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

input_signals = [
    cbadc.analog_signal.Sinusoidal(
        1, 1 / (1024 * analog_frontend.digital_control.clock.T)
    )
]
simulator = cbadc.simulator.NonLinearSimulator(
    analog_frontend.analog_system,
    analog_frontend.digital_control,
    input_signals,
    function_expansion=function_derivatives,
)
simulator_ref = cbadc.simulator.FullSimulator(
    analog_frontend_ref.analog_system,
    analog_frontend_ref.digital_control,
    input_signals,
)

digital_estimator(simulator)
digital_estimator_ref(simulator_ref)
for index in range(size):
    u_hat[index] = next(digital_estimator)
    u_hat_ref[index] = next(digital_estimator_ref)


###############################################################################
# Visualizing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
plt.figure()

# Ref

u_hat_cut = u_hat_ref[K1 + K2 :]
f, psd = cbadc.utilities.compute_power_spectral_density(
    u_hat_cut[:], fs=1 / analog_frontend.digital_control.clock.T, nperseg=u_hat_cut.size
)
signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
noise_index = np.ones(psd.size, dtype=bool)
noise_index[signal_index] = False
noise_index[f < (BW * 1e-2)] = False
noise_index[f > BW] = False
fom = cbadc.utilities.snr_spectrum_computation_extended(
    psd, signal_index, noise_index, fs=1 / analog_frontend.digital_control.clock.T
)
est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
est_ENOB = cbadc.fom.snr_to_enob(est_SNR)
plt.semilogx(
    f,
    10 * np.log10(np.abs(psd)),
    label=f"Ref, OSR={1/(2 * analog_frontend.digital_control.clock.T * BW):.0f}, est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB",
)


# Est
u_hat_cut = u_hat[K1 + K2 :]
f, psd = cbadc.utilities.compute_power_spectral_density(
    u_hat_cut[:], fs=1 / analog_frontend.digital_control.clock.T, nperseg=u_hat_cut.size
)
signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
noise_index = np.ones(psd.size, dtype=bool)
noise_index[signal_index] = False
noise_index[f < (BW * 1e-2)] = False
noise_index[f > BW] = False
fom = cbadc.utilities.snr_spectrum_computation_extended(
    psd, signal_index, noise_index, fs=1 / analog_frontend.digital_control.clock.T
)
est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
est_ENOB = cbadc.fom.snr_to_enob(est_SNR)
plt.semilogx(
    f,
    10 * np.log10(np.abs(psd)),
    label=f"Est, OSR={1/(2 * analog_frontend.digital_control.clock.T * BW):.0f}, est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB",
)

plt.title("Power spectral density of input estimate")
plt.xlabel('Hz')
plt.ylabel('$V^2$ / Hz dB')
plt.legend()
plt.grid(which="both")
# plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()
