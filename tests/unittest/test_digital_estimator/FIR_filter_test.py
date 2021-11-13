import cbadc
import numpy as np
from tests.fixture.chain_of_integrators import chain_of_integrators

beta = 6250.0
rho = -62.5
N = 5
M = 5
A = np.eye(N) * rho + np.eye(N, k=-1) * beta
B = np.zeros((N, 1))
B[0, 0] = beta
# B[0, 1] = -beta
CT = np.eye(N)
Gamma_tildeT = np.eye(M)
Gamma = Gamma_tildeT * (-beta)
Ts = 1 / (2 * beta)


def test_write_C_header(chain_of_integrators):
    eta2 = 1.0
    K1 = 4
    K2 = 3
    digital_control = cbadc.digital_control.DigitalControl(
        1.0 / (2 * chain_of_integrators["beta"]), 5
    )
    filter = cbadc.digital_estimator.FIRFilter(
        chain_of_integrators["system"], digital_control, eta2, K1, K2
    )
    filter.write_C_header("FIR_filter_C_header")


def test_fixed_point():
    eta2 = 1e3
    K1 = 1 << 11
    K2 = 1 << 11
    size = 1 << 14

    analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
    analog_signals = [cbadc.analog_signal.Sinusoidal(0.5, 10)]
    # analog_signals = [cbadc.analog_signal.ConstantSignal(0.25)]
    digital_control_floating_point = cbadc.digital_control.DigitalControl(Ts, M)
    circuitSimulator_floating_point = cbadc.simulator.StateSpaceSimulator(
        analog_system, digital_control_floating_point, analog_signals
    )
    estimator_floating_point = cbadc.digital_estimator.FIRFilter(
        analog_system, digital_control_floating_point, eta2, K1, K2
    )
    estimator_floating_point(circuitSimulator_floating_point)

    digital_control_fixed_point = cbadc.digital_control.DigitalControl(Ts, M)
    circuitSimulator_fixed_point = cbadc.simulator.StateSpaceSimulator(
        analog_system, digital_control_fixed_point, analog_signals
    )

    fixed_point = cbadc.utilities.FixedPoint(64, 5.0)
    estimator_fixed_point = cbadc.digital_estimator.FIRFilter(
        analog_system,
        digital_control_fixed_point,
        eta2,
        K1,
        K2,
        fixed_point=fixed_point,
    )
    estimator_fixed_point(circuitSimulator_fixed_point)

    print(estimator_fixed_point.h)
    estimator_floating_point.warm_up()
    estimator_fixed_point.warm_up()
    for index in range(size - K2):
        print(index)
        est_floating = next(estimator_floating_point)
        est_fixed = next(estimator_fixed_point)
        np.testing.assert_allclose(est_floating, est_fixed, rtol=1e-5)


def test_write_C_header_with_fixed_point(chain_of_integrators):
    eta2 = 1.0
    K1 = 4
    K2 = 3
    fixed_point = cbadc.utilities.FixedPoint(8, 1.0)
    digital_control = cbadc.digital_control.DigitalControl(
        1.0 / (2 * chain_of_integrators["beta"]), 5
    )
    filter = cbadc.digital_estimator.FIRFilter(
        chain_of_integrators["system"],
        digital_control,
        eta2,
        K1,
        K2,
        fixed_point=fixed_point,
    )
    filter.write_C_header("FIR_filter_C_header_with_fixed_point")
