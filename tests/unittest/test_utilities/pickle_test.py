import cbadc
import pytest
import math
import numpy as np

filename = './dummy.pickle'


def pickle_unpickle(test_object):
    cbadc.utilities.pickle_dump(test_object, filename)
    unpickled_object = cbadc.utilities.pickle_load(filename)
    assert isinstance(unpickled_object, test_object.__class__)


def test_analog_signals():
    pickle_unpickle(cbadc.analog_signal.ConstantSignal())
    T = 1e-3
    tt = 1e-6
    td = 1e-12
    duty_cycle = 0.4
    pickle_unpickle(cbadc.analog_signal.Clock(T, tt, td, duty_cycle))
    t0 = 0.1
    pickle_unpickle(cbadc.analog_signal.StepResponse(t0))
    t0 = 0.5
    tau = 1e-3
    pickle_unpickle(cbadc.analog_signal.RCImpulseResponse(tau, t0))
    amplitude = 1.2
    bandwidth = 42.0
    delay = math.pi
    pickle_unpickle(cbadc.analog_signal.SincPulse(amplitude, bandwidth, delay))
    amplitude = 1.0
    frequency = 42.0
    pickle_unpickle(cbadc.analog_signal.Sinusoidal(amplitude, frequency))


def test_analog_system():
    beta = 6250.0
    rho = -62.5
    N = 5
    A = np.eye(N) * rho + np.eye(N, k=-1) * beta
    B = np.zeros((N, 1))
    B[0] = beta
    CT = np.zeros((N, 1)).transpose()
    CT[-1] = 1.0
    Gamma_tildeT = np.eye(N)
    Gamma = Gamma_tildeT * (-beta)
    analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    for item in analog_system.__dict__:
        print(item)
        pickle_unpickle(item)
    pickle_unpickle(analog_system)


def test_digital_control():
    Ts = 1e-3
    M = 4
    clock = cbadc.analog_signal.Clock(Ts)
    pickle_unpickle(cbadc.digital_control.DigitalControl(clock, M))


def test_digital_estimator():
    beta = 6250.0
    rho = -62.5
    N = 2
    M = 2
    A = np.eye(N) * rho + np.eye(N, k=-1) * beta
    B = np.zeros((N, 1))
    B[0, 0] = beta
    # B[0, 1] = -beta
    CT = np.eye(N)
    Gamma_tildeT = np.eye(M)
    Gamma = Gamma_tildeT * (-beta)
    Ts = 1 / (2 * beta)
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    eta2 = 1.0
    K1 = 100
    K2 = 0
    pickle_unpickle(
        cbadc.digital_estimator.BatchEstimator(
            analog_system, digitalControl, eta2, K1, K2
        )
    )


def test_simulator():
    beta = 6250.0
    rho = -62.5
    N = 5
    M = 5
    A = np.eye(N) * rho + np.eye(N, k=-1) * beta
    B = np.zeros((N, 1))
    B[0] = beta
    CT = np.zeros((1, N))
    CT[-1] = 1.0
    Gamma_tildeT = np.eye(M)
    Gamma = Gamma_tildeT * (-beta)
    Ts = 1 / (2 * beta)
    analogSignals = [cbadc.analog_signal.ConstantSignal(0.1)]
    clock = cbadc.analog_signal.Clock(Ts)
    digitalControl = cbadc.digital_control.DigitalControl(clock, M)
    analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    pickle_unpickle(
        cbadc.simulator.get_simulator(
            analog_system, digitalControl, analogSignals, clock
        )
    )
