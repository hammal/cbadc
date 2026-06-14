"""Tests for the data-aided calibrate/reconstruct workflow."""

import numpy as np
import pytest

from cbadc import snr as snr_mod
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import Sinusoidal, ZeroOrderHold
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


def test_fit_fft_reconstruction_is_sample_aligned():
    # snr_residual is delay-sensitive: a 1-sample misalignment tanks it. This
    # guards the fft tap-centering (the (K-1)//2 offset). Also covers even K.
    af, OSR = AnalogFrontend.chain_of_integrators(N=3, ENOB=10, BW=1e5)
    DSR = int(OSR)
    est = af.calibrate(DSR=DSR, K=1 << 7, J=4, sim_size=1 << 16, fit="fft")
    ref = ZeroOrderHold.uniform_reference_signal(
        af.dt, -np.ones((1, 4)), np.ones((1, 4)), size=(1 << 15) + (1 << 10), seed=7
    )
    af.analog_signal = ref
    sim = af.simulate((1 << 15) + (1 << 10))
    u_ref = snr_mod.decimate(sim["u"][1 << 10 :], DSR)
    s = snr_mod.snr_residual(est.reconstruct(sim["v"][1 << 10 :]), u_ref, trim=est.K)
    assert s > 30  # aligned -> tens of dB; a 1-sample offset would give < 0


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


def test_calibrate_fit_fft_polish_between_fft_and_lstsq():
    # fft+polish warm-starts from fft then runs matrix-free CG on the SAME
    # objective lstsq minimizes -> it must beat plain fft and land within a few
    # dB of lstsq. Measured on a HELD-OUT reference realisation (fresh seed).
    af, OSR = AnalogFrontend.chain_of_integrators(N=3, ENOB=10, BW=1e5)
    DSR = int(OSR)
    K, J = 1 << 7, 4

    # held-out evaluation against a fresh random reference
    n = (1 << 15) + (1 << 10)
    ref = ZeroOrderHold.uniform_reference_signal(
        af.dt * DSR, -np.ones((1, 4)), np.ones((1, 4)), size=n // DSR + K, seed=4242
    )
    af.analog_signal = ref
    sim = af.simulate(n)
    u_ref = snr_mod.decimate(sim["u"][1 << 10 :], DSR)
    v = sim["v"][1 << 10 :]

    snrs = {}
    for fit in ("fft", "fft+polish", "lstsq"):
        est = af.calibrate(DSR=DSR, K=K, J=J, sim_size=1 << 16, fit=fit)
        snrs[fit] = snr_mod.snr_residual(est.reconstruct(v), u_ref, trim=2 * est.K)

    # polish never hurts the fft warm start and closes most of the gap to lstsq
    assert snrs["fft+polish"] >= snrs["fft"] - 0.1
    assert snrs["fft+polish"] <= snrs["lstsq"] + 0.5
    assert snrs["fft+polish"] == pytest.approx(snrs["lstsq"], abs=3.0)


def test_fit_fft_polish_improves_monotonically_with_iters():
    # more CG iterations -> the matrix-free polish converges toward the lstsq
    # optimum; SNR must improve (within noise) and stay bounded by lstsq.
    af, OSR = AnalogFrontend.chain_of_integrators(N=3, ENOB=10, BW=1e5)
    DSR = int(OSR)
    K, J = 1 << 7, 4
    n = (1 << 15) + (1 << 10)
    ref = ZeroOrderHold.uniform_reference_signal(
        af.dt * DSR, -np.ones((1, 4)), np.ones((1, 4)), size=n // DSR + K, seed=909
    )
    af.analog_signal = ref
    sim = af.simulate(n)
    u_ref = snr_mod.decimate(sim["u"][1 << 10 :], DSR)
    v = sim["v"][1 << 10 :]

    s0 = snr_mod.snr_residual(
        af.calibrate(DSR=DSR, K=K, J=J, sim_size=1 << 16, fit="fft").reconstruct(v),
        u_ref,
        trim=2 * K,
    )
    s8 = snr_mod.snr_residual(
        af.calibrate(
            DSR=DSR, K=K, J=J, sim_size=1 << 16, fit="fft+polish", polish_iters=8
        ).reconstruct(v),
        u_ref,
        trim=2 * K,
    )
    assert s8 > s0  # eight CG steps strictly improve on the fft warm start


def test_calibrate_rejects_unknown_fit():
    af, OSR = AnalogFrontend.chain_of_integrators(N=2, ENOB=8, BW=1e5)
    with pytest.raises(ValueError, match="fft\\+polish"):
        af.calibrate(DSR=int(OSR), K=1 << 6, J=1, sim_size=1 << 13, fit="bogus")
