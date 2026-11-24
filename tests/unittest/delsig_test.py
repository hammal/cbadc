"""Smoke-test the optional DELSIG synthesis and simulation bridge.

Synthesize an NTF, realize it in several topologies, assemble/partition its
ABCD matrix, and run a binary modulator from that NTF.
"""

import time

import numpy as np
import pytest
from scipy.signal import ZerosPolesGain

from cbadc import delsig as ds

order = 4
OSR = 64
opt = 2
H_inf = 1.5
f0 = 0.0


def test_synthesizeNTF():
    """NTF synthesis returns a zero-pole-gain transfer function."""
    H = ds.synthesizeNTF(order, OSR, opt, H_inf, f0)
    assert isinstance(H, ZerosPolesGain)


@pytest.mark.parametrize("form", ["CIFB", "CIFF", "CRFB", "CRFF", "CRFBD", "CRFFD"])
def test_realizeNTF(form):
    """Each loop topology yields coefficient vectors of the expected order."""
    H = ds.synthesizeNTF(order, OSR, opt, H_inf, f0)
    a, g, b, c = ds.realizeNTF(H, form=form)
    assert a.shape == (order,)
    assert g.shape == (order // 2,)
    assert b.shape == (order + 1,)
    assert c.shape == (order,)


def test_stuff_and_map_ABCD():
    """Pack realization coefficients into ABCD and recover the same vectors.

    Partition the joint matrix as ``[[A, B], [C, D]]`` and check every block's
    shape before mapping its coefficients back.
    """
    H = ds.synthesizeNTF(order, OSR, opt, H_inf, f0)
    a, g, b, c = ds.realizeNTF(H, form="CRFB")
    ABCD = ds.stuffABCD(a, g, b, c)
    assert ABCD.shape == (order + 1, order + 2)

    A, B, C, D = ds.partitionABCD(ABCD)
    assert A.shape == (order, order)
    assert B.shape == (order, 2)
    assert C.shape == (1, order)
    assert D.shape == (1, 2)

    aa, gg, bb, cc = ds.mapABCD(ABCD)
    np.testing.assert_allclose(a, aa)
    np.testing.assert_allclose(g, gg)
    np.testing.assert_allclose(b, bb)
    np.testing.assert_allclose(c, cc)


def test_calculateTF():
    """The assembled ABCD must yield signal and noise transfer functions."""
    H = ds.synthesizeNTF(order, OSR, opt, H_inf, f0)
    a, g, b, c = ds.realizeNTF(H, form="CRFB")
    ABCD = ds.stuffABCD(a, g, b, c)
    ntf, stf = ds.calculateTF(ABCD)

    assert isinstance(ntf, ZerosPolesGain)
    assert isinstance(stf, ZerosPolesGain)


def test_simulateDSM():
    """A synthesized binary NTF runs for a long sinusoidal input record.

    Check finite decisions and state maxima; print elapsed time as a smoke
    benchmark for the optional DELSIG simulator.
    """
    H = ds.synthesizeNTF(order, OSR, opt, H_inf, f0)
    size = 1 << 18
    Bw = np.ceil(size / (2 * OSR))
    u = 0.5 * np.sin(2 * np.pi * Bw / size * np.arange(size))
    start_time = time.time()
    v, _, xmax, _ = ds.simulateDSM(u, H, nlev=2, x0=0.0)
    end_time = time.time()
    print(f"Simulation time: {end_time - start_time} seconds")
    assert v.size == size and np.isfinite(v).all()
    assert np.isfinite(xmax).all()
