from cbadc import DigitalControl
import numpy as np

M = 4
dt = 1.0


def test_mid_rise_and_mid_thread():
    dc = DigitalControl(M, dt)
    assert ((dc._mid_rise * dc._mid_thread) == 0).all()


def test_alpha_beta():
    alpha = np.ones(4) * 0.1
    beta = np.ones(4) * 0.9
    dc = DigitalControl(M, dt, alpha, beta)
    assert (dc.alpha == alpha).all()
    assert (dc.beta == beta).all()
    assert dc.dac_waveform == "rz"
    delays = np.array([0.0, 0.1, 0.8, 0.9, 1.0])
    expected = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    values = dc.impulse_response(delays)
    assert (values == expected).all()


def test_quantization_levels():
    # levels = 3 * np.ones(M, dtype=float)
    levels = np.array([2, 3, 4, 5])
    dc = DigitalControl(M, dt, quantization_level=levels)
    assert (dc.quantization_level.flatten() == levels).all()
    assert dc.dac_waveform == "nrz"
    assert (dc._mid_thread.flatten() == np.array([0, 1, 0, 1], dtype=int)).all()
    assert (dc._mid_rise.flatten() == np.array([1, 0, 1, 0], dtype=int)).all()


def test_evaluate():
    alpha = 0.1 * np.ones(M)
    beta = 0.9 * np.ones(M)
    levels = np.array([2, 3, 4, 5])
    dc = DigitalControl(M, dt, alpha, beta, quantization_level=levels)

    t = 0.1
    values = np.ones((M, 1)) * 0.3
    res = dc.evaluate(t, values)
    assert res.shape == (M,)


def test_quantize():
    levels = 4 * np.ones(M)
    gain = 4 * np.ones(M)
    dc = DigitalControl(M, dt, quantization_level=levels, quantization_gain=gain)
    values = np.array([0.1, 0.3, 0.6, 0.9]).reshape(-1, 1)
    res = dc.quantize(values)
    expected = np.array([1, 1, 3, 3], dtype=float)
    np.testing.assert_array_almost_equal(res.flatten(), expected)


def test_quantize_with_scaling():
    levels = 4 * np.ones(M)
    out_max = np.ones(M, dtype=float)
    dc = DigitalControl(M, dt, quantization_level=levels, out_max=out_max)
    values = np.array([0.5, 1, 1.5, 2]).reshape(-1, 1)
    res = dc.quantize(values)
    expected = np.array([1 / 3, 1 / 3, 1.0 / 3, 1.0], dtype=float)
    np.testing.assert_array_almost_equal(res.flatten(), expected)


def test_delay_steps():
    alpha = np.zeros(M)
    beta = 3.1 * np.ones(M)
    dc = DigitalControl(M, dt, alpha, beta)
    assert (dc.delay_steps() == 3).all()

    beta = np.ones(M)
    dc = DigitalControl(M, dt, alpha, beta)
    assert (dc.delay_steps() == 0).all()


def test_setting_dac_waveforms():
    dc = DigitalControl(M, dt)
    dc.dac_waveform = "rz"
    assert dc.dac_waveform == "rz"
    dc.dac_waveform = "nrz"
    assert dc.dac_waveform == "nrz"
    dc.dac_waveform = "ld"
    assert dc.dac_waveform == "ld"
    dc.dac_waveform = "qd"
    assert dc.dac_waveform == "qd"
    dc.dac_waveform = "scr"
    assert dc.dac_waveform == "scr"
    dc.dac_waveform = "cos"
    assert dc.dac_waveform == "cos"
    dc.dac_waveform = "ls"
    assert dc.dac_waveform == "ls"
    dc.dac_waveform = "nls"
    assert dc.dac_waveform == "nls"
