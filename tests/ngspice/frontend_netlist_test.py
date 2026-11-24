"""Check the physical gmC netlist against ngspice."""

import numpy as np
import pytest

from cbadc.circuit import GmC
from cbadc.frontend import Frontend, Sinusoid

from . import bench, spice


def test_nonideal_gmc_components():
    """Finite ``Ro`` and ``Cp`` alter both the state map and the circuit.

    The DAC still closes the digital loop; ngspice integrates the physical
    ``Cx``, ``Rx`` and ``G`` elements while cbadc uses their realized ``A, B``.
    """
    dt = 1e-5
    ideal = Frontend([[-0.2 / dt]], [[0.5 / dt, -0.5 / dt]], [[1.0]], dt=dt)
    af = GmC(ideal, Cint=2e-12, Cp=0.5e-12, Ro=1e7, thermal_noise=False)
    np.testing.assert_allclose(af.A, ideal.A * 0.8 - 4e4)
    np.testing.assert_allclose(af.B, ideal.B * 0.8)
    netlist = af.netlist(64)
    assert "Cx0 x0 0 2.5e-12" in netlist
    assert "Rx0 x0 0 10000000.0" in netlist
    assert "Gax0_0 0 x0 x0 0" in netlist
    assert "Gu0_0 0 x0 u0 0" in netlist
    assert "Gd0_0 0 x0 d0 0" in netlist
    cmp = bench.run_both(af, 64, np.array([0.01]))
    bench.assert_agree(cmp, min_agree=64, margin=1e-3, state_tol=2e-4)


def test_clock_netlist_is_size_independent():
    dt = 1e-5
    af = Frontend([[0.0]], [[0.5 / dt, -0.5 / dt]], [[1.0]], dt=dt)
    af.input = Sinusoid(0.3, 1 / (32 * dt))
    af = GmC(af, thermal_noise=False)
    size, x0 = 66, np.array([0.01])
    netlist = af.netlist(size, x0)

    def fixed_lines(text):
        return [line for line in text.splitlines() if not line.startswith("tran ")]

    assert "Aclk fclk dclk osc" in fixed_lines(netlist)
    assert fixed_lines(netlist) == fixed_lines(af.netlist(2 * size, x0))
    tran = spice.run(netlist)
    v_spice = spice.decisions(tran, af.N, af.M, dt, size)
    x = spice.states(tran, af.N, dt, size)
    r = af.simulate(size, x0=x0, record=("v", "x"), verbose=False)
    np.testing.assert_array_equal(v_spice[:, 0], r["v"][:, 0, 0])
    np.testing.assert_allclose(x[:, 0], r["x"][:, 0, 0], atol=2e-4)


def test_comparators_share_clock_edges():
    dt, size = 1e-5, 42
    af = Frontend(
        [[0.0]],
        [[0.0, 0.0, 0.0]],
        [[0.0], [0.0]],
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        dt=dt,
        input=Sinusoid(0.3, 1 / (8 * dt), 0.3),
    )
    af = GmC(af, thermal_noise=False)
    netlist = af.netlist(size)
    flip_flops = [
        line.split() for line in netlist.splitlines() if line.startswith("Aff")
    ]
    assert {line[0] for line in flip_flops} >= {"Aff0", "Aff1"}
    assert {line[2] for line in flip_flops} == {"dclk"}

    tran = spice.run(netlist)

    def edges(m):
        v = tran[:, 1 + af.N + m]
        times = tran[np.flatnonzero(np.diff(v > 0)) + 1, 0]
        return times[times > dt / 2]  # ignore bridge startup at t=0

    t0, t1 = edges(0), edges(1)
    assert len(t0) >= 4
    np.testing.assert_allclose(t0, t1, atol=dt / 128, rtol=0)
    np.testing.assert_allclose(t0 / dt, np.rint(t0 / dt), atol=1 / 64, rtol=0)


@pytest.mark.parametrize(
    "delay,width", [(0.0, 1.0), (0.25, 0.5), (1.25, 1.0), (0.5, 1.5), (2.25, 0.5)]
)
def test_dac_windows(delay, width):
    dt, size, x0 = 1e-5, 80, np.array([0.01])
    af = Frontend([[-0.2 / dt]], [[0.5 / dt, -0.5 / dt]], [[1.0]], dt=dt)
    af.input = Sinusoid(0.3, 1 / (32 * dt))
    af.dac = af.dac[0].with_window(delay, width)
    af = GmC(af, thermal_noise=False)
    netlist = af.netlist(size, x0)
    assert "wrdata tran.dat v(x0) v(v0)" in netlist
    netlist = netlist.replace(
        "wrdata tran.dat v(x0) v(v0)", "wrdata tran.dat v(x0) v(v0) v(d0)"
    )
    tran = spice.run(netlist)
    r = af.simulate(size, x0=x0, record=("v", "x", "t"), verbose=False)
    np.testing.assert_array_equal(
        spice.decisions(tran, 1, 1, dt, size)[:, 0], r["v"][:, 0, 0]
    )
    np.testing.assert_allclose(
        spice.states(tran, 1, dt, size)[1:, 0], r["x"][1:, 0, 0], atol=2e-4
    )

    v = r["v"][:, 0, 0]
    # Each decision adds a pulse at its start and removes it at its end.
    starts = r["t"] + delay * dt
    event_times = np.r_[starts, starts + width * dt]
    event_changes = np.r_[v, -v]
    order = np.argsort(event_times)
    times, first = np.unique(np.round(event_times[order] / dt, 10), return_index=True)
    changes = np.add.reduceat(event_changes[order], first)
    expected = times[(changes != 0) & (times >= 0.25) & (times < size - 0.5)] * dt

    d = np.rint(tran[:, -1])
    edges = np.flatnonzero(np.diff(d) != 0) + 1
    edge_times = tran[edges, 0]
    # Merge bridge transitions within one solver step into one DAC edge.
    split = np.r_[0, np.flatnonzero(np.diff(edge_times) > dt / 128) + 1, len(edges)]
    observed = np.array(
        [
            edge_times[a]
            for a, b in zip(split[:-1], split[1:])
            if d[edges[a] - 1] != d[edges[b - 1]]
        ]
    )
    observed = observed[(observed > dt / 4) & (observed < (size - 0.5) * dt)]
    np.testing.assert_allclose(observed, expected, atol=dt / 64, rtol=0)


def test_correlated_state_noise():
    dt, size = 1e-5, 1026
    af = Frontend(np.zeros((2, 2)), np.zeros((2, 3)), np.eye(2), dt=dt)
    af.state_covariance = np.array([[1000.0, 500.0], [500.0, 1000.0]])
    af = GmC(af, thermal_noise=False)
    tran = spice.run(af.netlist(size, seed=11))
    x_spice = spice.states(tran, af.N, dt, size)
    x_cbadc = af.simulate(size, seed=29, record="x", verbose=False)["x"][:, :, 0]
    target = af.state_covariance[0, 0] * dt
    for x in (x_spice, x_cbadc):
        np.testing.assert_allclose(np.cov(np.diff(x, axis=0).T), target, rtol=0.2)


def test_correlated_comparator_noise():
    dt, size = 1e-5, 1026
    af = Frontend(np.zeros((2, 2)), np.zeros((2, 3)), np.eye(2), dt=dt)
    af.output_covariance = np.array([[1.0, 0.8], [0.8, 1.0]])
    af = GmC(af, thermal_noise=False)
    tran = spice.run(af.netlist(size, seed=11))
    v_spice = spice.decisions(tran, af.N, af.M, dt, size)[1:]
    v_cbadc = af.simulate(size, seed=29, record="v", verbose=False)["v"][1:, :, 0]
    expected_agreement = 0.5 + np.arcsin(0.8) / np.pi
    for v in (v_spice, v_cbadc):
        assert np.abs(np.mean(v[:, 0] == v[:, 1]) - expected_agreement) < 0.04
        np.testing.assert_allclose(np.mean(v > 0, axis=0), 0.5, atol=0.05)
