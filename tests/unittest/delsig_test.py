import numpy as np
from cbadc import delsig as ds
from scipy.signal import ZerosPolesGain
import time

order = 4
OSR = 64
opt = 2
H_inf = 1.5
f0 = 0.0


def test_synthesizeNTF():
    H = ds.synthesizeNTF(order, OSR, opt, H_inf, f0)
    assert isinstance(H, ZerosPolesGain)


def test_realizeNTF():
    H = ds.synthesizeNTF(order, OSR, opt, H_inf, f0)

    # CIFB
    a, g, b, c = ds.realizeNTF(H, form="CIFB")
    assert a.shape == (order,)
    assert g.shape == (order // 2,)
    assert b.shape == (order + 1,)
    assert c.shape == (order,)

    # CIFF
    a, g, b, c = ds.realizeNTF(H, form="CIFF")
    assert a.shape == (order,)
    assert g.shape == (order // 2,)
    assert b.shape == (order + 1,)
    assert c.shape == (order,)

    # CRFB
    a, g, b, c = ds.realizeNTF(H, form="CRFB")
    assert a.shape == (order,)
    assert g.shape == (order // 2,)
    assert b.shape == (order + 1,)
    assert c.shape == (order,)

    # CRFF
    a, g, b, c = ds.realizeNTF(H, form="CRFF")
    assert a.shape == (order,)
    assert g.shape == (order // 2,)
    assert b.shape == (order + 1,)
    assert c.shape == (order,)

    # CRFBD
    a, g, b, c = ds.realizeNTF(H, form="CRFBD")
    assert a.shape == (order,)
    assert g.shape == (order // 2,)
    assert b.shape == (order + 1,)
    assert c.shape == (order,)

    # CRFFD
    a, g, b, c = ds.realizeNTF(H, form="CRFFD")
    assert a.shape == (order,)
    assert g.shape == (order // 2,)
    assert b.shape == (order + 1,)
    assert c.shape == (order,)


def test_stuff_and_map_ABCD():
    H = ds.synthesizeNTF(order, OSR, opt, H_inf, f0)
    a, g, b, c = ds.realizeNTF(H, form="CRFB")
    ABCD = ds.stuffABCD(a, g, b, c)
    assert ABCD.shape == (order + 1, order + 2)

    A, B, C, D = ds.partitionABCD(ABCD)
    print(A, ABCD)
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
    H = ds.synthesizeNTF(order, OSR, opt, H_inf, f0)
    a, g, b, c = ds.realizeNTF(H, form="CRFB")
    ABCD = ds.stuffABCD(a, g, b, c)
    ntf, stf = ds.calculateTF(ABCD)

    assert isinstance(ntf, ZerosPolesGain)
    assert isinstance(stf, ZerosPolesGain)


def test_simulateDSM():
    H = ds.synthesizeNTF(order, OSR, opt, H_inf, f0)
    size = 1 << 18
    Bw = np.ceil(size / (2 * OSR))
    u = 0.5 * np.sin(2 * np.pi * Bw / size * np.arange(size))
    start_time = time.time()
    ds.simulateDSM(u, H, nlev=2, x0=0.0)
    end_time = time.time()
    print(f"Simulation time: {end_time - start_time} seconds")
    # assert False
