from cbc.circuit_simulator import CircuitSimulator
from cbc.digital_estimator.digital_estimator import DigitalEstimator
from cbc.parallel_digital_estimator.digital_estimator import DigitalEstimator as ParallelDigitalEstimator
from cbc.analog_signal import AnalogSignal, Sinusodial
from cbc.analog_system import AnalogSystem
from cbc.digital_control import DigitalControl
from ..AnalogToDigital import Sin, System, Control, Simulator, WienerFilter
from matplotlib import pyplot as plt
import numpy as np

beta = 6250.0
rho = -6.25 * 0
N = 6
M = N
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0, 0] = beta
# B[0, 1] = -beta
C = np.eye(N)
Gamma_tilde = np.eye(M)
Gamma = Gamma_tilde * (-beta)
Ts = 1/(2 * beta)

amplitude = .76123514
frequency = 11.232
phase = np.pi/3*2.


def test_estimation_with_circuit_simulator():
    eta2 = 1e12
    K1 = 10000
    K2 = 0
    size = 10000
    window = 1000
    size_2 = size // 2
    window_2 = window // 2
    left_w = size_2 - window_2
    right_w = size_2 + window_2

    analogSystem = AnalogSystem(A, B, C, Gamma, Gamma_tilde)
    analogSignals = [Sinusodial(amplitude, frequency, phase)]
    # analogSignals = [AnalogSignal(0.25)]
    digitalControl1 = DigitalControl(Ts, M)

    tf_abs = np.abs(analogSystem.transfer_function(
        np.array([2 * np.pi * frequency])))
    print(tf_abs, tf_abs.shape)

    circuitSimulator1 = CircuitSimulator(
        analogSystem, digitalControl1, analogSignals)
    estimator1 = DigitalEstimator(
        circuitSimulator1, analogSystem, digitalControl1, eta2, K1, K2)
    digitalControl2 = DigitalControl(Ts, M)
    circuitSimulator2 = CircuitSimulator(
        analogSystem, digitalControl2, analogSignals)
    estimator2 = ParallelDigitalEstimator(
        circuitSimulator2, analogSystem, digitalControl1, eta2, K1, K2)
    digitalControl4 = DigitalControl(Ts, M)
    circuitSimulator4 = CircuitSimulator(
        analogSystem, digitalControl4, analogSignals)
    estimator4 = DigitalEstimator(
        circuitSimulator4, analogSystem, digitalControl4, eta2, K1, K2, midPoint=True)

    tf_1 = estimator1.transfer_function(np.array([2 * np.pi * frequency]))[0]
    tf_2 = estimator2.transfer_function(np.array([2 * np.pi * frequency]))[0]
    tf_4 = estimator4.transfer_function(np.array([2 * np.pi * frequency]))[0]

    # Old Python Framework
    input = Sin(Ts, amplitude, frequency, phase, B.flatten())
    system = System(A, C)
    ctrl = Control(Gamma, size)
    simulator = Simulator(system, control=ctrl,
                          initalState=np.zeros(N), options={})
    t = np.linspace(0, Ts * (size - 1), size)
    result = simulator.simulate(t, (input,))
    filter = WienerFilter(t, system, (input,), options={
                          "eta2": np.ones(N) * eta2})
    e3 = filter.filter(ctrl)[0]
    print(f"Olf Bf: {filter.Bf}")
    print(f"Olf Bb: {filter.Bb}")
    #

    e1_array = np.zeros(size)
    e2_array = np.zeros(size)
    e4_array = np.zeros(size)
    model_error = 0
    e1_error = 0
    e2_error = 0
    e3_error = 0
    e4_error = 0

    for index in range(size):
        e1 = estimator1.__next__()
        e2 = estimator2.__next__()
        e4 = estimator4.__next__()
        e1_array[index] = e1
        e2_array[index] = e2
        e4_array[index] = e4
        t = index * Ts
        u = analogSignals[0].evaluate(t)
        if (index > left_w and index < right_w):
            print(
                f"Time: {t:0.2f}, Input Signal: {u * tf_2}, e1: {e1}, e2: {e2}, e3: {e3[index]}, e4: {e4}")
            e1_error += np.abs(e1 - u * tf_1)**2
            e2_error += np.abs(e2 - u * tf_2)**2
            e3_error += np.abs(e3[index] - u * tf_1)**2
            e4_error += np.abs(e4 - u * tf_2)**2
    e1_error /= window
    e2_error /= window
    e3_error /= window
    e4_error /= window
    print(
        f"Quadratic estimator error:    {e1_error}, {10 * np.log10(e1_error)} dB")
    print(
        f"Linear estimator error:       {e2_error}, {10 * np.log10(e2_error)} dB")
    print(
        f"Python estimator error:       {e3_error}, {10 * np.log10(e3_error)} dB")
    print(
        f"MidPoint estimator error:     {e4_error}, {10 * np.log10(e4_error)} dB")

    assert(np.allclose(e1_error, 0, rtol=1e-3, atol=1e-3))
    assert(np.allclose(e2_error, 0, rtol=1e-3, atol=1e-3))
    assert(np.allclose(e3_error, 0, rtol=1e-3, atol=1e-3))
    assert(np.allclose(e4_error, 0, rtol=1e-3, atol=1e-3))
