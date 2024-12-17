import os
import cbadc
import numpy as np
from tests.fixture.chain_of_integrators import chain_of_integrators
from cbadc.analog_signal import Clock

_ = chain_of_integrators

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
    Ts = 1.0 / (2 * chain_of_integrators["beta"])
    clock = Clock(Ts)
    digital_control = cbadc.digital_control.DigitalControl(clock, 5)
    filter = cbadc.digital_estimator.FIRFilter(
        chain_of_integrators["system"], digital_control, eta2, K1, K2
    )
    filename = "FIR_filter_C_header"
    filter.write_C_header(filename)
    os.remove(f"{filename}.h")


# TODO fix scaling factor
# def test_fixed_point():
#     eta2 = 1e3
#     K1 = 1 << 10
#     K2 = 1 << 10
#     size = 1 << 11

#     analog_filter = cbadc.analog_filter.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
#     analog_signals = [cbadc.analog_signal.Sinusoidal(0.5, 10)]
#     # analog_signals = [cbadc.analog_signal.ConstantSignal(0.25)]
#     clock = cbadc.analog_signal.Clock(Ts)
#     digital_control_floating_point = cbadc.digital_control.DigitalControl(clock, M)
#     estimator_floating_point = cbadc.digital_estimator.FIRFilter(
#         analog_filter, digital_control_floating_point, eta2, K1, K2
#     )
#     digital_control_fixed_point = cbadc.digital_control.DigitalControl(clock, M)
#     fixed_point = cbadc.utilities.FixedPoint(64, 5.0)
#     estimator_fixed_point = cbadc.digital_estimator.FIRFilter(
#         analog_filter,
#         digital_control_fixed_point,
#         eta2,
#         K1,
#         K2,
#         fixed_point=fixed_point,
#     )

#     print(estimator_fixed_point.h)
#     np.testing.assert_allclose(
#         estimator_fixed_point.h, estimator_floating_point.h, rtol=1e-10
#     )


def test_write_C_header_with_fixed_point(chain_of_integrators):
    eta2 = 1.0
    K1 = 4
    K2 = 3
    Ts = 1.0 / (2 * chain_of_integrators["beta"])
    clock = Clock(Ts)
    fixed_point = cbadc.utilities.FixedPoint(8, 1.0)
    digital_control = cbadc.digital_control.DigitalControl(clock, 5)
    filter = cbadc.digital_estimator.FIRFilter(
        chain_of_integrators["system"],
        digital_control,
        eta2,
        K1,
        K2,
        fixed_point=fixed_point,
    )
    filename = "FIR_filter_C_header_with_fixed_point"
    filter.write_C_header(filename)
    os.remove(f"{filename}.h")


def test_impulse_response(chain_of_integrators):
    eta2 = 1.0
    K1 = 4
    K2 = 3
    Ts = 1.0 / (2 * chain_of_integrators["beta"])
    clock = Clock(Ts)
    fixed_point = cbadc.utilities.FixedPoint(8, 1.0)
    digital_control = cbadc.digital_control.DigitalControl(clock, 5)
    filter = cbadc.digital_estimator.FIRFilter(
        chain_of_integrators["system"],
        digital_control,
        eta2,
        K1,
        K2,
        fixed_point=fixed_point,
    )
    np.testing.assert_almost_equal(filter.h[:, ::-1, :], filter.impulse_response())
