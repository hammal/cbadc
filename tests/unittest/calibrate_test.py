"""Tests for the data-aided calibrate/reconstruct workflow."""

import numpy as np
import pytest

from cbadc import snr as snr_mod
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import Sinusoidal
from cbadc.digital_backend import BlackBoxEstimator, DataAidedEstimator


def test_calibrate_returns_estimator_and_reconstructs():
    af, OSR = AnalogFrontend.chain_of_integrators(N=3, ENOB=10, BW=1e5)
    DSR = int(OSR)
    est = af.calibrate(DSR=DSR, K=1 << 6, J=2, sim_size=1 << 14)
    assert isinstance(est, DataAidedEstimator)
    assert BlackBoxEstimator is DataAidedEstimator  # deprecated alias preserved
    assert est.DSR == DSR

    # reconstruct an in-band tone and check the data-aided readout works
    size = 1 << 15
    fs_bb = 1.0 / (af.dt * DSR)
    f_sig = snr_mod.coherent_frequency(fs_bb / 8, fs_bb, size // DSR)
    af.analog_signal = Sinusoidal(np.array([[0.5]]), np.array([[f_sig]]))
    v = af.simulate(size)["v"]
    u_hat = est.reconstruct(v)
    # decimated baseband length, single channel
    assert u_hat.shape[0] == pytest.approx(v.shape[0] // DSR, abs=2)
    assert u_hat.shape[1:] == (af.L, 1)
    snr = snr_mod.snr_tone(u_hat[:, 0, 0], f_sig, fs_bb, trim=2 * est.K)
    assert snr > 20  # the calibrated FIR reconstructs the tone


def test_calibrate_fit_fft_matches_lstsq():
    af, OSR = AnalogFrontend.chain_of_integrators(N=3, ENOB=10, BW=1e5)
    DSR = int(OSR)
    size = 1 << 15
    fs_bb = 1.0 / (af.dt * DSR)
    f_sig = snr_mod.coherent_frequency(fs_bb / 8, fs_bb, size // DSR)
    af.analog_signal = Sinusoidal(np.array([[0.5]]), np.array([[f_sig]]))
    v = af.simulate(size)["v"]

    snrs = {}
    for fit in ("lstsq", "fft"):
        est = af.calibrate(DSR=DSR, K=1 << 7, J=4, sim_size=1 << 16, fit=fit)
        snrs[fit] = snr_mod.snr_tone(
            est.reconstruct(v)[:, 0, 0], f_sig, fs_bb, trim=est.K, band=1e5
        )
    assert snrs["fft"] > 20
    # the frequency-domain fit tracks the time-domain lstsq within a few dB
    assert snrs["fft"] == pytest.approx(snrs["lstsq"], abs=5.0)


def test_calibrate_accepts_custom_reference():
    af, OSR = AnalogFrontend.chain_of_integrators(N=2, ENOB=8, BW=1e5)
    DSR = int(OSR)
    from cbadc.analog_signal import ZeroOrderHold

    ref = ZeroOrderHold.uniform_reference_signal(
        af.dt * DSR, -np.ones((1, 1)), np.ones((1, 1)), size=(1 << 14) + 64, seed=7
    )
    est = af.calibrate(DSR=DSR, K=1 << 6, J=1, sim_size=1 << 14, reference=ref)
    assert est.DSR == DSR
    assert est.h is not None
