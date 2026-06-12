"""AnalogFrontend.evaluate: one-call calibrate -> held-out -> SNR report."""
import numpy as np

from cbadc.analog_frontend import AnalogFrontend, EvaluationResult


def test_evaluate_fft_returns_result():
    af, OSR = AnalogFrontend.chain_of_integrators(N=4, ENOB=10, BW=1e5)
    res = af.evaluate(DSR=int(OSR), K=128, J=8, sim_size=1 << 16, fit="fft",
                      verbose=False)
    assert isinstance(res, EvaluationResult)
    assert res.snr_band > 20.0
    assert res.enob > 3.0
    assert res.h.shape == (128, af.M, af.L)
    assert res.u_hat.ndim == 3 and res.u_hat.shape[1] == af.L
    assert res.u_hat.shape == res.u_ref.shape
    assert "ENOB" in repr(res)


def test_evaluate_lstsq_and_snr_of_f():
    af, OSR = AnalogFrontend.chain_of_integrators(N=4, ENOB=10, BW=1e5)
    res = af.evaluate(DSR=int(OSR), K=96, J=4, sim_size=1 << 15, fit="lstsq",
                      snr_of_f=True, verbose=False)
    assert res.snr_band > 20.0
    f, s = res.snr_of_f
    assert f.shape == s.shape and f.ndim == 1


def test_evaluate_band_is_inner_subband():
    # signal band excludes the edge -> band SNR >= full (or close) on a stable design
    af, OSR = AnalogFrontend.leapfrog(N=6, ENOB=12, BW=1e6)
    res = af.evaluate(DSR=int(OSR), K=256, J=8, sim_size=1 << 17, fit="fft",
                      band=(0.0, 0.7), verbose=False)
    assert res.snr_band >= res.snr - 1.0
