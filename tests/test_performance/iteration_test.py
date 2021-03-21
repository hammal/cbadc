# from cbc import StateSpaceSimulator
# from cbc.digital_estimator import DigitalEstimator
# from cbc.parallel_digital_estimator.digital_estimator import
# DigitalEstimator as ParallelDigitalEstimator
# from cbc import AnalogSignal, Sinusodial
# from cbc import AnalogSystem
# from cbc import DigitalControl
# from ..AnalogToDigital import Sin, System, Control, Simulator, WienerFilter
# import numpy as np

# beta = 6250.0
# rho = -62.5
# N = 10
# A = np.eye(N) * rho + np.eye(N, k=-1) * beta
# B = np.zeros((N, 1))
# B[0, 0] = beta
# # B[0, 1] = -beta
# C = np.eye(N)
# Gamma_tilde = np.eye(N)
# Gamma = Gamma_tilde * (-beta)
# Ts = 1/(2 * beta)

# amplitude = 1.0
# frequency = 10.
# phase = 0.

# eta2 = 1e12
# K1 = 100
# K2 = 0
# size = K1 + K2
# analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
# digitalControl = DigitalControl(Ts, N)
# analogSignals = [Sinusodial(amplitude, frequency, phase)]

# # Old parameters
# input = Sin(Ts, amplitude, frequency, phase, B.flatten())
# system = System(A, C)
# t = np.linspace(0, Ts * (size - 1), size)


# def controlSequence():
#     while True:
#         yield np.ones(N, dtype=np.uint8)


# def iterate_through(iterator):
#     count = 0
#     for _ in range(size):
#         iterator.__next__()
#         count = count + 1
#     return count


# def test_benchmark_parallel_linear_algorithm(benchmark):
#     est = ParallelDigitalEstimator(
#         controlSequence(), analogSystem, digitalControl, eta2, K1, K2)
#     result = benchmark(iterate_through, est)
#     assert(result == size)


# def test_benchmark_quadratic_algorithm(benchmark):
#     est = DigitalEstimator(
#         controlSequence(), analogSystem, digitalControl, eta2, K1, K2)
#     result = benchmark(iterate_through, est)
#     assert(result == size)


# def test_benchmark_circuit_simulation_algorithm(benchmark):
#     est = StateSpaceSimulator(
#         analogSystem, digitalControl, analogSignals)
#     result = benchmark(iterate_through, est)
#     assert(result == size)


# def test_benchmark_old_python_simulation(benchmark):
#     def temp():
#         ctrl = Control(Gamma, size)
#         simulator = Simulator(system, control=ctrl,
#                               initalState=np.ones(N), options={})
#         s = simulator.simulate(t, (input,))['output']
#         return s.shape[0]
#     result = benchmark(temp)
#     assert(result == size)


# def test_benchmark_old_python_filtering(benchmark):
#     ctrl = Control(Gamma, size)
#     simulator = Simulator(system, control=ctrl,
#                           initalState=np.ones(N), options={})
#     t = np.linspace(0, Ts * (size - 1), size)
#     simulator = simulator.simulate(t, (input,))
#     recon = WienerFilter(t, system, (input,))

#     def temp():
#         t = recon.filter(ctrl)[0]
#         return t.size
#     result = benchmark(temp)
#     assert(result == size)
