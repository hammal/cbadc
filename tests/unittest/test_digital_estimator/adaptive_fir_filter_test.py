import cbadc
from scipy.signal import firwin
import pytest
import numpy as np
import tensorflow as tf
import os


@pytest.fixture()
def setup():
    size = 1 << 12
    R = 2
    M = 5
    K = 1 << 10
    h = np.array([firwin(K, 0.1), firwin(K, 0.2)]).reshape((R, K))
    s_control = np.random.randint(2, size=(size, M + R))
    return R, M, K, h, s_control


def test_init_filter(setup):
    R, _, K, h, _ = setup
    assert h.shape[0] == R
    assert h.shape[1] == K


def test_initializer(setup):
    R, M, K, h, _ = setup
    filter = cbadc.digital_estimator.AdaptiveFIRFilter()


def test_filter_generate_data(setup):
    R, M, K, h, s_control = setup
    filter = cbadc.digital_estimator.AdaptiveFIRFilter()
    x, y = filter.generate_dataset(s_control, h)
    print(x.shape, y.shape)
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == M * K


def test_filter_predict_full(setup):
    R, M, K, h, s_control = setup
    filter = cbadc.digital_estimator.AdaptiveFIRFilter()
    x, y = filter.generate_dataset(s_control, h)
    filter.compile(optimizer="adam", loss="mse")
    filter.fit(x, y, epochs=1, batch_size=1)
    y_hat = filter.predict_full(x, y)
    assert y_hat.shape[0] == y.shape[0]


def test_plot_impulse_response(setup):
    R, M, K, h, s_control = setup
    filter = cbadc.digital_estimator.AdaptiveFIRFilter()
    x, y = filter.generate_dataset(s_control, h)
    filter.compile(optimizer="adam", loss="mse")
    filter.fit(x, y, epochs=1, batch_size=1)
    filter.plot_impulse_response(M)


def test_bode_plot(setup):
    R, M, K, h, s_control = setup
    filter = cbadc.digital_estimator.AdaptiveFIRFilter()
    x, y = filter.generate_dataset(s_control, h)
    filter.compile(optimizer="adam", loss="mse")
    filter.fit(x, y, epochs=1, batch_size=1)
    filter.plot_bode(M)


def test_save_load(setup):
    R, M, K, h, s_control = setup
    filter = cbadc.digital_estimator.AdaptiveFIRFilter()
    x, y = filter.generate_dataset(s_control, h)
    filter.compile(optimizer="adam", loss="mse")
    filter.fit(x, y, epochs=1, batch_size=1)
    filter.save("test.keras")
    tf.keras.models.load_model("test.keras")
    os.remove("test.keras")
