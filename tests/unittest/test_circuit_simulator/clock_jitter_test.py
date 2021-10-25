import cbadc
import numpy as np
import pytest


def test_simulator_verify_with_estimator():
    N = 1
    M = N

    C_x = 1e-9
    C_Gamma = C_x / 2
    R_s = 100.0
    R_beta = 16e4

    beta = 1 / (R_beta * C_x)
    T = 1 / (2 * beta)
    std = T * 0.1  # 10% clock jitter
    clock_jitter = lambda: np.random.random() * std

    A = beta * np.eye(N, k=-1) + np.random.randn(N, N) * beta * 1e-3
    B = np.zeros(N)
    B[0] = beta
    CT = np.eye(N)
    impulse_response = cbadc.digital_control.RCImpulseResponse(R_s * C_Gamma)
    digital_control_sc = cbadc.digital_control.DigitalControl(
        T, M, impulse_response=impulse_response
    )

    Gamma = -1 / (R_s * C_x) * np.eye(M)
    Gamma_tildeT = np.eye(M)

    analog_system_sc = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)

    print(digital_control_sc)
    print(analog_system_sc)

    amplitude = 1e-2
    analog_signal = cbadc.analog_signal.Sinusodial(amplitude, 1 / T / 64)

    size = 1 << 10

    simulator_sc = cbadc.simulator.StateSpaceSimulator(
        analog_system_sc,
        digital_control_sc,
        [analog_signal],
        Ts=T,
        clock_jitter=clock_jitter,
        initial_state_vector=np.array([0.5]),
    )

    digital_control_ref = cbadc.digital_control.DigitalControl(
        T, M, impulse_response=impulse_response
    )
    simulator_ref = cbadc.simulator.StateSpaceSimulator(
        analog_system_sc,
        digital_control_ref,
        [analog_signal],
        Ts=T,
        initial_state_vector=np.array([0.5]),
    )

    # Digital Estimators
    eta2 = 1e3
    K1 = 1 << 8
    K2 = K1
    # prepending an anti-aliasing filter
    # omega_3dB = 2 * np.pi / T / 16
    # wp = omega_3dB / 2.0
    # ws = omega_3dB
    # gpass = 1.0
    # gstop = 60
    # filter = cbadc.analog_system.IIRDesign(wp, ws, gpass, gstop, ftype="ellip")
    digital_estimator_sc = cbadc.digital_estimator.FIRFilter(
        # cbadc.analog_system.chain([filter, analog_system_sc]),
        analog_system_sc,
        digital_control_sc,
        eta2,
        K1,
        K2,
    )
    digital_estimator_sc(simulator_sc)
    digital_estimator_ref = cbadc.digital_estimator.FIRFilter(
        # cbadc.analog_system.chain([filter, analog_system_sc]),
        analog_system_sc,
        digital_control_ref,
        eta2,
        K1,
        K2,
    )
    digital_estimator_ref(simulator_ref)
    # Simulations
    for _ in cbadc.utilities.show_status(range(size)):
        u_hat = next(digital_estimator_sc)
        u_hat_ref = next(digital_estimator_ref)
        print(np.round(simulator_sc.t, 4), u_hat, u_hat_ref, u_hat - u_hat_ref)
        np.testing.assert_allclose(u_hat, u_hat_ref, atol=1e-2)
