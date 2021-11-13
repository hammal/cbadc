import numpy as np
from cbadc.digital_control import DigitalControl
from cbadc.analog_system import AnalogSystem
from cbadc.analog_signal import Sinusoidal
from cbadc.simulator import StateSpaceSimulator
from cbadc.digital_estimator import (
    DigitalEstimator,
    ParallelEstimator,
    FIRFilter,
    IIRFilter,
)

beta = 6250.0
rho = -6.25 * 0
N = 6
M = N
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0, 0] = beta
# B[0, 1] = -beta
CT = np.eye(N)
Gamma_tildeT = np.eye(M)
Gamma = Gamma_tildeT * (-beta)
Ts = 1 / (2 * beta)

amplitude = 0.76123514
frequency = 11.232
phase = np.pi / 3 * 2.0


def test_estimation_with_circuit_simulator():
    eta2 = 1e12

    K1 = 1 << 10
    K2 = 1 << 10
    size = K2 << 2
    window = 1000
    size_2 = size // 2
    window_2 = window // 2
    left_w = size_2 - window_2
    right_w = size_2 + window_2

    analogSystem = AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    analogSignals = [Sinusoidal(amplitude, frequency, phase)]
    digitalControl1 = DigitalControl(Ts, M)
    digitalControl2 = DigitalControl(Ts, M)
    digitalControl3 = DigitalControl(Ts, M)
    digitalControl4 = DigitalControl(Ts, M)

    tf_abs = np.abs(
        analogSystem.transfer_function_matrix(np.array([2 * np.pi * frequency]))
    )
    print(tf_abs, tf_abs.shape)

    simulator1 = StateSpaceSimulator(analogSystem, digitalControl1, analogSignals)
    simulator2 = StateSpaceSimulator(analogSystem, digitalControl2, analogSignals)
    simulator3 = StateSpaceSimulator(analogSystem, digitalControl3, analogSignals)
    simulator4 = StateSpaceSimulator(analogSystem, digitalControl4, analogSignals)
    estimator1 = DigitalEstimator(analogSystem, digitalControl1, eta2, K1, K2)
    estimator2 = ParallelEstimator(analogSystem, digitalControl2, eta2, K1, K2)
    estimator3 = FIRFilter(analogSystem, digitalControl3, eta2, K1, K2)
    estimator4 = IIRFilter(analogSystem, digitalControl4, eta2, K2)
    estimator1(simulator1)
    estimator2(simulator2)
    estimator3(simulator3)
    estimator4(simulator4)

    tf_1 = estimator1.signal_transfer_function(np.array([2 * np.pi * frequency]))[0]

    e1_array = np.zeros(size)
    e2_array = np.zeros(size)
    e3_array = np.zeros(size)
    e4_array = np.zeros(size)
    e1_error = 0
    e2_error = 0
    e3_error = 0
    e4_error = 0

    estimator1.warm_up()
    estimator2.warm_up()
    estimator3.warm_up()
    estimator4.warm_up()

    for index in range(size):
        e1 = estimator1.__next__()
        e2 = estimator2.__next__()
        e3 = estimator3.__next__()
        e4 = estimator4.__next__()
        e1_array[index] = e1
        e2_array[index] = e2
        e3_array[index] = e3
        e4_array[index] = e4
        t = index * Ts
        u = analogSignals[0].evaluate(t)
        if index > left_w and index < right_w:
            print(
                f"Time: {t: 0.2f}, Input Signal: {u * tf_1}, e1: {e1}, e2: {e2}, e3: {e3}, e4: {e4}"
            )
            e1_error += (
                np.abs(
                    e1
                    - analogSignals[0].evaluate(t - Ts * estimator1.filter_lag()) * tf_1
                )
                ** 2
            )
            e2_error += (
                np.abs(
                    e2
                    - analogSignals[0].evaluate(t - Ts * estimator2.filter_lag()) * tf_1
                )
                ** 2
            )
            e3_error += (
                np.abs(
                    e3
                    - analogSignals[0].evaluate(t - Ts * estimator3.filter_lag()) * tf_1
                )
                ** 2
            )
            e4_error += (
                np.abs(
                    e4
                    - analogSignals[0].evaluate(t - Ts * estimator4.filter_lag()) * tf_1
                )
                ** 2
            )
    e1_error /= window
    e2_error /= window
    e3_error /= window
    e4_error /= window
    print(
        f"""Digital estimator error:        {e1_error}, {10 *
        np.log10(e1_error)} dB"""
    )
    print(
        f"""Parallel estimator error:       {e2_error}, {10 *
        np.log10(e2_error)} dB"""
    )
    print(
        f"""FIR filter estimator error:     {e3_error}, {10 *
        np.log10(e3_error)} dB"""
    )
    print(
        f"""IIR filter estimator error:     {e4_error}, {10 *
        np.log10(e4_error)} dB"""
    )

    assert np.allclose(e1_error, 0, rtol=1e-6, atol=1e-6)
    assert np.allclose(e2_error, 0, rtol=1e-6, atol=1e-6)
    assert np.allclose(e3_error, 0, rtol=1e-6, atol=1e-6)
    assert np.allclose(e4_error, 0, rtol=1e-6, atol=1e-6)
