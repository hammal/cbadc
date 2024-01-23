import cbadc
import pytest
import math
import numpy as np
import uuid
import os


@pytest.fixture()
def pickle_unpickle():
    filename = f"{str(uuid.uuid4())}.pkl"

    def pickle_unpickle(test_object):
        cbadc.utilities.pickle_dump(test_object, filename)
        unpickled_object = cbadc.utilities.pickle_load(filename)
        assert isinstance(unpickled_object, test_object.__class__)

    yield pickle_unpickle
    if os.path.exists(filename):
        print(f"removing file: {filename}")
        os.remove(filename)


def test_analog_signals(pickle_unpickle):
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


def test_analog_system(pickle_unpickle):
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
    pickle_unpickle(analog_system)


def test_chain_of_integrator(pickle_unpickle):
    beta = 6250.0
    rho = -62.5
    N = 5
    Gamma_tildeT = np.eye(N)
    Gamma = Gamma_tildeT * (-beta)
    analog_system = cbadc.analog_system.ChainOfIntegrators(
        beta * np.ones(N), rho * np.ones(N), Gamma
    )
    pickle_unpickle(analog_system)


def test_leapfrog(pickle_unpickle):
    beta = 6250.0
    alpha = -beta / 10
    rho = -62.5
    N = 5
    Gamma_tildeT = np.eye(N)
    Gamma = Gamma_tildeT * (-beta)
    analog_system = cbadc.analog_system.LeapFrog(
        beta * np.ones(N), alpha * np.ones(N - 1), rho * np.ones(N), Gamma
    )
    pickle_unpickle(analog_system)


def test_filters(pickle_unpickle):
    N = 6
    Wn = (1e2, 1e3)
    rp = 3e0
    rs = 1.0
    analog_system = cbadc.analog_system.ButterWorth(N, Wn[0])
    pickle_unpickle(analog_system)
    analog_system = cbadc.analog_system.ChebyshevI(N, Wn[1], rp)
    pickle_unpickle(analog_system)
    analog_system = cbadc.analog_system.ChebyshevII(N, Wn[0], rs)
    pickle_unpickle(analog_system)
    # analog_system = cbadc.analog_system.Cauer(N, Wn[0], rp, rs)
    # pickle_unpickle(analog_system)
    wp = 0.2
    ws = 0.3
    gpass = 0.1
    gstop = 2.0
    for ftype in ["butter", "cheby1", "cheby2", "ellip"]:
        analog_system = cbadc.analog_system.IIRDesign(wp, ws, gpass, gstop, ftype=ftype)
        pickle_unpickle(analog_system)


def test_digital_control(pickle_unpickle):
    Ts = 1e-3
    M = 4
    clock = cbadc.analog_signal.Clock(Ts)
    digital_control = cbadc.digital_control.DigitalControl(clock, M)
    pickle_unpickle(digital_control)
    pickle_unpickle(cbadc.digital_control.DitherControl(1, digital_control))
    pickle_unpickle(
        cbadc.digital_control.MultiLevelDigitalControl(clock, M, [5 for _ in range(M)])
    )
    # pickle_unpickle(
    #     cbadc.digital_control.MultiPhaseDigitalControl(clock, np.arange(M) / M * Ts)
    # )
    # pickle_unpickle(
    #     cbadc.digital_control.SwitchedCapacitorControl(clock, np.arange(M) / Ts)
    # )


@pytest.mark.parametrize(
    "reconstruction_method",
    [
        pytest.param(cbadc.digital_estimator.BatchEstimator, id="batch_de"),
        pytest.param(cbadc.digital_estimator.ParallelEstimator, id="par-batch-de"),
        pytest.param(cbadc.digital_estimator.IIRFilter, id="IIR_de"),
        pytest.param(cbadc.digital_estimator.FIRFilter, id="FIR_de"),
    ],
)
def test_digital_estimator(reconstruction_method, pickle_unpickle):
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
    K2 = K1
    pickle_unpickle(reconstruction_method(analog_system, digitalControl, eta2, K1, K2))


@pytest.mark.parametrize(
    "simulation_method",
    [
        pytest.param(cbadc.simulator.FullSimulator, id="full_num_sim"),
        pytest.param(
            cbadc.simulator.PreComputedControlSignalsSimulator, id="pre_num_sim"
        ),
    ],
)
def test_simulator(simulation_method, pickle_unpickle):
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
    pickle_unpickle(simulation_method(analog_system, digitalControl, analogSignals))


def test_analog_frontend(pickle_unpickle):
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
    pickle_unpickle(cbadc.analog_frontend.AnalogFrontend(analog_system, digitalControl))


@pytest.fixture
def chain_of_integrators():
    beta = 6250.0
    rho = -62.5
    N = 5
    A = np.eye(N) * rho + np.eye(N, k=-1) * beta
    B = np.zeros((N, 1))
    B[0] = beta
    CT = np.zeros((N, 1)).transpose()
    CT[0, -1] = 1.0
    Gamma_tildeT = np.eye(N)
    Gamma = Gamma_tildeT * (-beta)
    return {
        "N": N,
        "A": A,
        "B": B,
        "CT": CT,
        "M": N,
        "Gamma": Gamma,
        "Gamma_tildeT": Gamma_tildeT,
        "system": cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT),
        "beta": beta,
        "rho": rho,
    }
