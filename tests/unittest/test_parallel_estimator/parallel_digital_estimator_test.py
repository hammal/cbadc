# from cbadc import StateSpaceSimulator
# from cbadc.parallel_digital_estimator.digital_estimator import DigitalEstimator
# from cbadc import AnalogSignal, Sinusoidal
# from cbadc import AnalogSystem
# from cbadc import DigitalControl
# import numpy as np
# from tests.test_analog_system.chain_of_integrators import
# chain_of_integrators
# beta = 6250.0
# rho = -62.5
# N = 5
# M = 5
# A = np.eye(N) * rho + np.eye(N, k=-1) * beta
# B = np.zeros((N, 1))
# B[0, 0] = beta
# # B[0, 1] = -beta
# C = np.eye(N)
# Gamma_tilde = np.eye(M)
# Gamma = Gamma_tilde * (-beta)
# Ts = 1/(2 * beta)


# def controlSequence():
#     while True:
#         yield np.ones(M, dtype=np.uint8)


# def test_initialization(chain_of_integrators):
#     digitalControl = DigitalControl(Ts, M)
#     eta2 = 1.0
#     K1 = 100
#     K2 = 0
#     DigitalEstimator(controlSequence(
#     ), chain_of_integrators['system'], digitalControl, eta2, K1, K2)
#     # assert(False)


# def test_estimation_empty():
#     digitalControl = DigitalControl(Ts, M)
#     eta2 = 100.0
#     K1 = 100
#     K2 = 10

#     analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)

#     estimator = DigitalEstimator(
#         controlSequence(), analogSystem, digitalControl, eta2, K1, K2,
# stop_after_number_of_iterations=25)
#     assert(estimator.empty() == True)


# def test_estimation_full():
#     digitalControl = DigitalControl(Ts, M)
#     eta2 = 100.0
#     K1 = 100
#     K2 = 10

#     analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)

#     estimator = DigitalEstimator(
#         controlSequence(), analogSystem, digitalControl, eta2, K1, K2,
# stop_after_number_of_iterations=25)
#     assert(estimator.full() == False)


# def test_estimation():
#     digitalControl = DigitalControl(Ts, M)
#     eta2 = 100.0
#     K1 = 100
#     K2 = 10

#     analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)

#     estimator = DigitalEstimator(
#         controlSequence(), analogSystem, digitalControl, eta2, K1, K2,
# stop_after_number_of_iterations=25)
#     for est in estimator:
#         print(np.array(est))


# def test_batch_iterations():
#     digitalControl = DigitalControl(Ts, M)
#     eta2 = 100.0
#     K1 = 25
#     K2 = 1000

#     analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
#     estimator = DigitalEstimator(controlSequence(
#     ), analogSystem, digitalControl, eta2, K1, K2=K2,
# stop_after_number_of_iterations=200)
#     for est in estimator:
#         print(np.array(est))
#     # raise "temp"


# def test_estimation_with_circuit_simulator():
#     eta2 = 1e12
#     K1 = 1000
#     K2 = 0

#     analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
#     # analogSignals = [Sinusoidal(0.5, 10)]
#     analogSignals = [AnalogSignal(0.)]
#     digitalControl = DigitalControl(Ts, M)
#     circuitSimulator = StateSpaceSimulator(
#         analogSystem, digitalControl, analogSignals, t_stop=Ts * 1001)
#     estimator = DigitalEstimator(
#         circuitSimulator, analogSystem, digitalControl, eta2, K1, K2)
#     for est in estimator:
#         print('Est from python: ', est[0])


# def test_performance():
#     digitalControl = DigitalControl(Ts, M)
#     eta2 = 100.0
#     K1 = 25
#     K2 = 1000

#     analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
#     estimator = DigitalEstimator(controlSequence(
#     ), analogSystem, digitalControl, eta2, K1, K2=K2,
# stop_after_number_of_iterations=10000)
#     for est in estimator:
#         pass


# def test_batch_computation():
#     eta2 = 1e12
#     K1 = 3
#     K2 = 100

#     analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
#     analogSignals = [Sinusoidal(0.5, 10)]
#     # analogSignals = [AnalogSignal(0.)]
#     digitalControl = DigitalControl(Ts, M)
#     circuitSimulator = StateSpaceSimulator(
#         analogSystem, digitalControl, analogSignals, t_stop=Ts * 1000)
#     estimator = DigitalEstimator(
#         circuitSimulator, analogSystem, digitalControl, eta2, K1, K2)
#     for est in estimator:
#         print('Est from python: ', est[0])
#     # assert(False)
