"""Compare N=3, 4, and 7 leapfrog loops with ngspice circuits.

The netlist implements the same A, B, C, D dynamics, clocked comparators,
and DACs. Check decisions and states until a threshold-sensitive decision
diverges; then compare the signal quality reconstructed from each bit stream.
"""

import numpy as np
import pytest

from cbadc import snr as snr_mod
from cbadc.circuit import GmC
from cbadc.frontend import DAC, Constant, Frontend, Sinusoid

from . import spice


def _leapfrog(N, ENOB, size, amplitude=0.5):
    """Build an immediate-NRZ loop with a coherent test tone and SNR reader."""
    af, OSR = Frontend.leapfrog(N=N, ENOB=ENOB, BW=1e3)
    # Override the default symmetric RZ pulse for this reference circuit.
    af.dac = DAC(Constant(1.0))
    fs = 1.0 / af.dt
    f = snr_mod.coherent_frequency(fs / (2 * OSR) / 4, fs, size)
    af.input = Sinusoid(amplitude, f, -np.pi / 2)  # the netlist drives a sine
    wf = af.wiener_filter(OSR=OSR)

    def snr_of(v):
        return snr_mod.snr_tone(wf(v)[:, 0, 0], f, fs, trim=512, band=1e3)

    return GmC(af, thermal_noise=False), snr_of


def _assert_decisions_agree(v_spice, r, *, min_agree, margin):
    """Return the first mismatch, allowed only near ``y=0`` after ``min_agree``.

    One different ``v`` changes later feedback, so later decisions need not
    agree. Before that point both simulators follow the same trajectory.
    """
    size = v_spice.shape[0]
    agree = np.all(v_spice == r["v"][:, :, 0], axis=1)
    if agree.all():
        return size
    fm = int(np.argmin(agree))
    flipped = v_spice[fm] != r["v"][fm, :, 0]
    y = np.abs(r["y"][fm, :, 0][flipped])
    assert fm >= min_agree, f"decisions part at period {fm}, |y| = {y}"
    assert y.max() < margin, (
        f"decision {fm} differs with comparator input {y} -- not a threshold case"
    )
    return fm


def _golden(af, size, x0):
    """Read ngspice ``v`` at clock periods and ``x`` at clock edges."""
    tr = spice.run(af.netlist(size, x0))
    return spice.decisions(tr, af.N, af.M, af.dt, size), spice.states(
        tr, af.N, af.dt, size
    )


def test_third_order_decisions_states_and_snr():
    """N=3: check early decisions, states, and reconstructed tone quality.

    Run the same 4096-period loop in cbadc and ngspice from ``x0``. Require
    at least half the decisions to match, permit only a near-zero ``y`` at the
    first mismatch, and compare states up to it and SNR over the full record.
    """
    size = 1 << 12
    af, snr_of = _leapfrog(3, 12, size)
    x0 = np.full(af.N, 0.01)
    r = af.simulate(size, x0=x0, record="all")
    v_spice, x_spice = _golden(af, size, x0)

    fm = _assert_decisions_agree(v_spice, r, min_agree=size // 2, margin=5e-4)
    assert np.max(np.abs(x_spice[:fm] - r["x"][:fm, :, 0])) < 5e-4
    assert snr_of(v_spice[:, :, None]) == pytest.approx(snr_of(r["v"]), abs=0.1)
    assert snr_of(r["v"]) > 70


def test_fourth_order_reconstruction():
    """N=4: compare states before divergence and SNR after it.

    The tighter loop can part at a near-threshold decision. Require 1000
    matching periods, bounded state error until the first mismatch, and
    reconstructed SNR within 1 dB over the complete bit streams.
    """
    size = 1 << 12
    af, snr_of = _leapfrog(4, 14, size)
    x0 = np.full(af.N, 0.01)
    r = af.simulate(size, x0=x0, record="all")
    v_spice, x_spice = _golden(af, size, x0)

    fm = _assert_decisions_agree(v_spice, r, min_agree=1000, margin=2e-3)
    assert np.max(np.abs(x_spice[:fm] - r["x"][:fm, :, 0])) < 2e-3
    assert snr_of(v_spice[:, :, None]) == pytest.approx(snr_of(r["v"]), abs=1.0)


def test_seventh_order_osr12_long_record():
    """N=7, OSR=12: compare early trajectories and long-run SNR.

    After at least 128 matching periods, a near-threshold decision may make
    the bit streams diverge. Check the state before that point, then require
    both 16384-period streams to reconstruct the tone with similar SNR.
    """
    size = 1 << 14
    af, osr = Frontend.leapfrog(N=7, OSR=12, BW=1e3)
    af.dac = DAC(Constant(1.0))
    f = snr_mod.coherent_frequency(af.fs / 256, af.fs, size)
    af.input = Sinusoid(0.5, f, -np.pi / 2)
    af = GmC(af, thermal_noise=False)
    x0 = np.full(af.N, 0.01)
    r = af.simulate(size, x0=x0, record="all")
    v_spice, x_spice = _golden(af, size, x0)
    fm = _assert_decisions_agree(v_spice, r, min_agree=128, margin=5e-3)
    assert np.max(np.abs(x_spice[:fm] - r["x"][:fm, :, 0])) < 5e-3
    wf = af.wiener_filter(OSR=osr)

    def snr_of(bits):
        return snr_mod.snr_tone(wf(bits)[:, 0, 0], f, af.fs, trim=512, band=1e3)

    snr_cbadc, snr_spice = snr_of(r["v"]), snr_of(v_spice[:, :, None])
    assert snr_cbadc > 20, f"cbadc reconstructed SNR {snr_cbadc:.1f} dB"
    assert snr_spice == pytest.approx(snr_cbadc, abs=3.0)
