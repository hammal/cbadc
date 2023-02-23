# import cbadc
# import shlib
# from pade.utils import init_logger
# import numpy as np
# import matplotlib.pyplot as plt
# from pade.psf_parser import PSFParser
# logger = init_logger()

# work_dir = shlib.to_path(__file__).parent
# netlist_dir = shlib.to_path(work_dir, 'netlist')
# observation_filename = shlib.to_path(work_dir, 'observations.csv')
# spectre_raw_data_dir = shlib.to_path(work_dir, 'spectre_raw_data')
# spectre_log_file = shlib.to_path(work_dir, 'spectre_sim.log')
# shlib.mkdir(spectre_raw_data_dir)
# shlib.mkdir(netlist_dir)

# def get_eta2(AF, fc):
#     eta2 = (
#         np.linalg.norm(
#             AF.analog_system.transfer_function_matrix(
#                 np.array([2 * np.pi * fc])
#             )) ** 2 )
#     return eta2

# def get_leapfrog_dither(N, OSR, BW, dither_gain=0.1, kappa_scale=1, **kwargs):
#     T = 1.0 / (2 * OSR * BW)
#     omega_BW = 2.0 * np.pi * BW
#     beta = 1 / (2 * T)
#     alpha = -((omega_BW / 2) ** 2) / beta

#     # Apply scaling
#     beta_scale = kwargs.get('beta_scale', np.ones((N)))
#     alpha_scale = np.array([1/beta_scale[i+1] for i in range(0, N-1)])
#     beta_vec = beta*beta_scale
#     kappa_scale = kappa_scale*np.array([np.prod(beta_scale[:i+1]) for i in range(N)])
#     kappa_vec = -beta*kappa_scale
#     alpha_vec = alpha*alpha_scale
#     rho_vec = np.zeros((N))

#     Gamma = np.zeros((N, N+1))
#     Gamma[0,0] = kappa_vec.flatten()[0] * dither_gain
#     Gamma[:, 1:] = np.diag(kappa_vec)

#     A = np.diag(alpha_vec, k=1) + np.diag(beta_vec[1:], k=-1) + np.diag(rho_vec)
#     B = np.zeros((N, 1))
#     B[0] = beta_vec[0]
#     CT = np.eye(N)
#     Gamma_tildeT = -np.eye(N)

#     digital_control = cbadc.digital_control.DigitalControl(cbadc.analog_signal.Clock(T), N)
#     digital_control = cbadc.digital_control.DitherControl(1, digital_control,
#             impulse_response=digital_control._impulse_response[
#                 0
#             ]
#             )
#     analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
#     return cbadc.analog_frontend.AnalogFrontend(analog_system, digital_control)


# def test_testbench_to_file():
#     testbench = cbadc.old_circuit.get_testbench(
#         analog_frontend_target,
#         [input_signal],
#         save_all_variables=True,
#         save_to_filename=observation_filename,
#     )
#     testbench.to_file(netlist_filename, path=netlist_dir)


# def test_testbench_sim_1():

#     testbench = cbadc.old_circuit.get_testbench(
#         analog_frontend_target,
#         [input_signal],
#         save_all_variables=True,
#         save_to_filename=observation_filename,
#     )
#     testbench.run_spectre_simulation(
#         netlist_filename,
#         path=netlist_dir,
#         raw_data_dir=spectre_raw_data_dir,
#         log_file=spectre_log_file,
#     )


# def test_testbench_sim_2():
#     testbench = cbadc.old_circuit.get_opamp_testbench(
#         analog_frontend_target,
#         [input_signal],
#         C=1e-12,
#         save_all_variables=True,
#         save_to_filename=observation_filename,
#     )
#     testbench.run_spectre_simulation(
#         netlist_filename,
#         path=netlist_dir,
#         raw_data_dir=spectre_raw_data_dir,
#         log_file=spectre_log_file,
#     )


# def test_testbench_sim_3():
#     testbench = cbadc.old_circuit.get_opamp_testbench(
#         analog_frontend_target,
#         [input_signal],
#         C=1e-12,
#         GBWP=1e3 * BW,
#         A_DC=1e3,
#         save_all_variables=True,
#         save_to_filename=observation_filename,
#     )
#     testbench.run_spectre_simulation(
#         netlist_filename,
#         path=netlist_dir,
#         raw_data_dir=spectre_raw_data_dir,
#         log_file=spectre_log_file,
#     )


# def test_get_spectre_simulator_default():
#     testbench = cbadc.old_circuit.get_opamp_testbench(
#         analog_frontend_target,
#         [input_signal],
#         C=1e-12,
#         GBWP=1e3 * BW,
#         A_DC=1e3,
#         save_to_filename=observation_filename,
#     )
#     simulator = testbench.get_spectre_simulator(observation_filename)
#     for i in range(10):
#         print(next(simulator))


# def test_get_spectre_simulator_all_variables():
#     testbench = cbadc.old_circuit.get_opamp_testbench(
#         analog_frontend_target,
#         [input_signal],
#         C=1e-12,
#         GBWP=1e3 * BW,
#         A_DC=1e3,
#         save_all_variables=True,
#         save_to_filename=observation_filename,
#     )
#     simulator = testbench.get_spectre_simulator(observation_filename)
#     for i in range(10):
#         print(next(simulator))


# def test_with_estimator():
#     testbench = cbadc.old_circuit.get_testbench(
#         analog_frontend_target,
#         [input_signal],
#         save_all_variables=False,
#         save_to_filename=observation_filename,
#     )
#     simulator = testbench.get_spectre_simulator(observation_filename)
#     eta2 = testbench.analog_frontend.analog_system.analog_system.eta2(BW)
#     K1 = 1 << 8
#     K2 = K1
#     estimator = testbench.analog_frontend.get_estimator(
#         cbadc.digital_estimator.FIRFilter, eta2, K1, K2
#     )
#     estimator(simulator)

#     size = 1 << 12
#     u_hat = np.zeros(size)
#     for i in range(size):
#         u_hat[i] = next(estimator)
#     u_hat_cut = u_hat[K1 + K2 :]
#     fs = 1 / testbench.analog_frontend.digital_control.digital_control.clock.T
#     f, psd = cbadc.utilities.compute_power_spectral_density(
#         u_hat_cut[:], fs=fs, nperseg=u_hat_cut.size
#     )
#     signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
#     noise_index = np.ones(psd.size, dtype=bool)
#     noise_index[signal_index] = False
#     noise_index[f < (BW * 1e-2)] = False
#     noise_index[f > BW] = False
#     fom = cbadc.utilities.snr_spectrum_computation_extended(
#         psd, signal_index, noise_index, fs=fs
#     )
#     est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
#     est_ENOB = cbadc.fom.snr_to_enob(est_SNR)

#     plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
#     plt.semilogx(
#         f,
#         10 * np.log10(np.abs(psd)),
#         label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}",
#     )
#     plt.xlabel('Hz')
#     plt.ylabel('V^2 / Hz dB')
#     plt.legend()
#     plt.savefig("test_with_estimator_psd.png")
#     assert est_ENOB >= ENOB


# def test_full():
#     testbench = cbadc.old_circuit.get_opamp_testbench(
#         analog_frontend_target,
#         [input_signal],
#         C=1e-12,
#         GBWP=1e6 * BW,
#         A_DC=1e6,
#         save_all_variables=False,
#         save_to_filename=observation_filename,
#     )
#     # testbench = cbadc.circuit.get_testbench(
#     #     analog_frontend_target,
#     #     [input_signal],
#     #     save_all_variables=False,
#     #     save_to_filename=observation_filename,
#     # )
#     testbench.run_spectre_simulation(
#         netlist_filename,
#         path=netlist_dir,
#         raw_data_dir=spectre_raw_data_dir,
#         log_file=spectre_log_file,
#     )
#     simulator = testbench.get_spectre_simulator(observation_filename)

#     K1 = 1 << 8
#     K2 = K1
#     estimator = testbench.analog_frontend.get_estimator(
#         cbadc.digital_estimator.FIRFilter, eta2, K1, K2
#     )
#     estimator(simulator)

#     size = 1 << 12
#     u_hat = np.zeros(size)
#     for i in range(size):
#         u_hat[i] = next(estimator)
#     u_hat_cut = u_hat[K1 + K2 :]
#     fs = 1 / testbench.analog_frontend.digital_control.digital_control.clock.T
#     f, psd = cbadc.utilities.compute_power_spectral_density(
#         u_hat_cut[:], fs=fs, nperseg=u_hat_cut.size
#     )
#     signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
#     noise_index = np.ones(psd.size, dtype=bool)
#     noise_index[signal_index] = False
#     noise_index[f < (BW * 1e-2)] = False
#     noise_index[f > BW] = False
#     fom = cbadc.utilities.snr_spectrum_computation_extended(
#         psd, signal_index, noise_index, fs=fs
#     )
#     est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
#     est_ENOB = cbadc.fom.snr_to_enob(est_SNR)

#     plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
#     plt.semilogx(
#         f,
#         10 * np.log10(np.abs(psd)),
#         label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}",
#     )
#     plt.xlabel('Hz')
#     plt.ylabel('V^2 / Hz dB')
#     plt.legend()
#     plt.savefig("test_full_psd.png")
#     assert est_ENOB >= ENOB - 0.5


# def test_full_dither(analog_frontend_target):

#     testbench = cbadc.old_circuit.get_opamp_testbench(
#         analog_frontend_target,
#         [input_signal],
#         C=1e-12,
#         # GBWP=1e6 * BW,
#         # A_DC=1e6,
#         save_all_variables=False,
#         save_to_filename=observation_filename,
#     )
#     # testbench = cbadc.circuit.get_testbench(
#     #     analog_frontend_target,
#     #     [input_signal],
#     #     save_all_variables=False,
#     #     save_to_filename=observation_filename,
#     # )
#     testbench.run_spectre_simulation(
#         netlist_filename,
#         path=netlist_dir,
#         raw_data_dir=spectre_raw_data_dir,
#         log_file=spectre_log_file,
#     )
#     simulator = testbench.get_spectre_simulator(observation_filename)

#     K1 = 1 << 8
#     K2 = K1
#     estimator = testbench.analog_frontend.get_estimator(
#         cbadc.digital_estimator.FIRFilter, eta2, K1, K2
#     )
#     estimator(simulator)

#     size = 1 << 12
#     u_hat = np.zeros(size)
#     for i in range(size):
#         u_hat[i] = next(estimator)
#     u_hat_cut = u_hat[K1 + K2 :]
#     fs = 1 / testbench.analog_frontend.digital_control.digital_control.clock.T
#     f, psd = cbadc.utilities.compute_power_spectral_density(
#         u_hat_cut[:], fs=fs, nperseg=u_hat_cut.size
#     )
#     signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
#     noise_index = np.ones(psd.size, dtype=bool)
#     noise_index[signal_index] = False
#     noise_index[f < (BW * 1e-2)] = False
#     noise_index[f > BW] = False
#     fom = cbadc.utilities.snr_spectrum_computation_extended(
#         psd, signal_index, noise_index, fs=fs
#     )
#     est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
#     est_ENOB = cbadc.fom.snr_to_enob(est_SNR)

#     plt.title(f"Power spectral density:\nN={N},ENOB={ENOB}")
#     plt.semilogx(
#         f,
#         10 * np.log10(np.abs(psd)),
#         label=f"est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB, BW={BW:.0e}",
#     )
#     plt.xlabel('Hz')
#     plt.ylabel('V^2 / Hz dB')
#     plt.legend()
#     plt.savefig("test_full_psd.png")
#     assert est_ENOB >= ENOB - 0.5


# if __name__ == '__main__':
#     RERUN_SIM = 1

#     N = 4
#     BW = 1e6
#     OSR = 25
#     ENOB = 15 #

#     SIM_NAME = f'LF_N{N}_B{ENOB}'
#     netlist_filename = f"{SIM_NAME}.scs"

#     # analog_frontend_target = cbadc.synthesis.get_leap_frog(ENOB=ENOB, N=N, BW=BW)
#     analog_frontend_target = get_leapfrog_dither(N, OSR, BW)

#     T = analog_frontend_target.digital_control.clock.T
#     amplitude = 0.3
#     frequency = 1.0 / T
#     while frequency > BW:
#         frequency /= 2
#     input_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency, offset=0.5)

#     eta2 = get_eta2(analog_frontend_target, BW)

#     test_full_dither(analog_frontend_target)
