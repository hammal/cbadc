from AnalogToDigital import Sin, System, Control, Simulator, WienerFilter
import numpy as np
import matplotlib.pyplot as plt

size = 10000
eta2 = 1e12

amplitude = .76123514
frequency = 11.232
phase = np.pi/3*2.

beta = 6250.0
rho = -6.25 * 0
N = 6
M = N

A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0, 0] = beta
C = np.eye(N)
Gamma_tilde = np.eye(M)
Gamma = Gamma_tilde * (-beta)
Ts = 1/(2 * beta)

input = Sin(Ts, amplitude, frequency, phase, B.flatten())
system = System(A, C)
ctrl = Control(Gamma, size)
simulator = Simulator(system, control=ctrl,
                      initalState=np.ones(N), options={})
t = np.linspace(0, Ts * (size - 1), size)
result = simulator.simulate(t, (input,))
filter = WienerFilter(t, system, (input,), options={
    "eta2": np.ones(N) * eta2})
e = filter.filter(ctrl)[0]

plt.plot(t, e, label='est')
plt.plot(t, amplitude * np.sin(2 * np.pi * frequency * t + phase), label='ref')
plt.legend()
plt.show()
