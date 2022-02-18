# from cbadc import StateSpaceSimulator
# from cbadc.digital_estimator import DigitalEstimator
# from cbadc.parallel_digital_estimator.digital_estimator import
# DigitalEstimator as ParallelDigitalEstimator
# from cbadc import AnalogSignal, Sinusoidal
# from cbadc import AnalogSystem
# from cbadc import DigitalControl
# from ..AnalogToDigital import Sin, System, Control, Simulator, WienerFilter
# import numpy as np

# beta = 6250.0
# rho = -62.5
# N = 5
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
# K1 = 10
# K2 = 1000
# size = 10000
# analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
# digitalControl = DigitalControl(Ts, N)
# analogSignals = [Sinusoidal(0.5, 1)]


# def test_old_library():
#     input = Sin(Ts, amplitude, frequency, phase, B.flatten())
#     system = System(A, C)
#     ctrl = Control(Gamma, size)
#     simulator = Simulator(system, control=ctrl,
#                           initalState=np.ones(N), options={})
#     t = np.linspace(0, Ts * (size - 1), size)
#     result = simulator.simulate(t, (input,))
#     filter = WienerFilter(t, system, (input,))
#     u_hat = filter.filter(ctrl)
