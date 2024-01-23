from cbadc.digital_control import DigitalControl
from cbadc.analog_signal import Clock
import numpy as np


def test_initialization():
    Ts = 1e-3
    M = 4
    clock = Clock(Ts)
    DigitalControl(clock, M)


def test_evaluate():
    Ts = 1e-3
    M = 4
    clock = Clock(Ts)
    digitalControl = DigitalControl(clock, M)
    x = np.random.randn(4)
    t = 0.0
    digitalControl.control_update(t, x)
    res = digitalControl.control_contribution(t)
    for value in res:
        assert value == 1 or value == -1


def test_control_signal():
    Ts = 1e-3
    M = 4
    clock = Clock(Ts)
    digitalControl = DigitalControl(clock, M)
    x = np.ones(M)
    t = 0.0
    np.testing.assert_allclose(np.zeros(M), digitalControl.control_signal())
    print("Initial control signal: ", np.asarray(digitalControl.control_signal()))
    digitalControl.control_update(t, x)
    res = digitalControl.control_contribution(t)
    print(
        "Control signal after first control contribution call: ",
        np.asarray(digitalControl.control_signal()),
    )
    print("control contribution response: ", np.asarray(res))
    np.testing.assert_allclose(np.ones(M), digitalControl.control_signal())
    np.testing.assert_allclose(np.ones(M), res)
    digitalControl.control_update(t, -x)
    res = digitalControl.control_contribution(t)
    print(
        "control signal after second control contribution update: ",
        np.asarray(digitalControl.control_signal()),
    )
    print("control contribution response: ", np.asarray(res))
    np.testing.assert_allclose(np.zeros(M), digitalControl.control_signal())
    np.testing.assert_allclose(-np.ones(M), res)
    digitalControl.control_update(t, -x)
    res = digitalControl.control_contribution(t)
    print(
        "control signal after third control contribution update: ",
        np.asarray(digitalControl.control_signal()),
    )
    print("control contribution response: ", np.asarray(res))
    np.testing.assert_allclose(np.zeros(M), digitalControl.control_signal())
    np.testing.assert_allclose(-np.ones(M), res)
