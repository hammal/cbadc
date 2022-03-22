from cbadc.digital_control import MultiLevelDigitalControl
from cbadc.analog_signal import Clock
import numpy as np


def test_initialization():
    Ts = 1e-3
    M = 4
    clock = Clock(Ts)
    MultiLevelDigitalControl(clock, M, [2] * M)


def test_evaluate():
    Ts = 1e-3
    M = 4
    clock = Clock(Ts)
    digitalControl = MultiLevelDigitalControl(clock, M, [1] * M)
    x = np.random.randn(4)
    t = 0.1
    digitalControl.control_update(t, x)
    res = digitalControl.control_contribution(t)
    for value in res:
        assert value == 1 or value == -1


def test_evaluate_2_levels():
    Ts = 1e-3
    M = 4
    clock = Clock(Ts)
    digitalControl = MultiLevelDigitalControl(clock, M, [2] * M)
    x = np.array([-1, -0.45, 0.1, 0.55])
    t = 0.1
    digitalControl.control_update(t, x)
    res = digitalControl.control_contribution(t)
    print(res)
    for value in res:
        assert value == -1 or value == 0 or value == 1


def test_evaluate_3_levels():
    Ts = 1e-3
    M = 4
    clock = Clock(Ts)
    digitalControl = MultiLevelDigitalControl(clock, M, [3] * M)
    x = np.array([-1, -0.45, 0.1, 0.55])
    t = 0.1
    digitalControl.control_update(t, x)
    res = digitalControl.control_contribution(t)
    print(res)
    for value in res:
        assert (
            value == -1
            or np.allclose(value, 2 / 3 - 1)
            or np.allclose(value, 1 - 2 / 3)
            or value == 1
        )


def test_evaluate_4_levels():
    Ts = 1e-3
    M = 4
    clock = Clock(Ts)
    digitalControl = MultiLevelDigitalControl(clock, M, [4] * M)
    x = np.array([-1, -0.45, 0.1, 0.55])
    t = 0.1
    digitalControl.control_update(t, x)
    res = digitalControl.control_contribution(t)
    print(res)
    for value in res:
        assert (
            value == -1
            or np.allclose(value, 2 / 4 - 1)
            or np.allclose(value, 0)
            or np.allclose(value, 1 - 2 / 4)
            or value == 1
        )
