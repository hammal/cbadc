from cbadc import DigitalControl
import numpy as np


def test_initialization():
    Ts = 1e-3
    M = 4
    digitalControl = DigitalControl(Ts, M)


def test_evaluate():
    Ts = 1e-3
    M = 4
    digitalControl = DigitalControl(Ts, M)
    x = np.random.randn(4)
    t = 0.1
    res = digitalControl.evaluate(t, x)
    for value in res:
        assert value == 1 or value == -1


def test_control_signal():
    Ts = 1e-3
    M = 4
    digitalControl = DigitalControl(Ts, M)
    x = np.ones(M)
    t = Ts
    np.testing.assert_allclose(np.zeros(M), digitalControl.control_signal())
    print(np.asarray(digitalControl.control_signal()))
    res = digitalControl.evaluate(t, x)
    print(np.asarray(digitalControl.control_signal()))
    print(np.asarray(res))
    np.testing.assert_allclose(np.ones(M), digitalControl.control_signal())
    np.testing.assert_allclose(np.ones(M), res)
    res = digitalControl.evaluate(t+Ts, -x)
    print(np.asarray(digitalControl.control_signal()))
    print(np.asarray(res))
    np.testing.assert_allclose(np.zeros(M), digitalControl.control_signal())
    np.testing.assert_allclose(-np.ones(M), res)
    res = digitalControl.evaluate(t, x)
    print(np.asarray(digitalControl.control_signal()))
    print(np.asarray(res))
    np.testing.assert_allclose(np.zeros(M), digitalControl.control_signal())
    np.testing.assert_allclose(-np.ones(M), res)
