"""Check ``x = x_u + x_hat`` and the twin readout across loop paths."""

import numpy as np
import pytest

from cbadc import DAC, Constant, Frontend, Pole, Sinusoid, StateSpace, cascade

SIZE = 1 << 11


def signal_alone(af: Frontend, size: int) -> np.ndarray:
    """Compute ``x_u`` from A and the input columns of B, without DAC drive."""
    N, L = af.N, af.L
    model = StateSpace(af.A[0, 0], af.B[0, 0, :, :L], np.eye(N), np.zeros((N, L)))
    source = cascade(af.input[0], model)
    X = source.trajectory(af.dt, size)
    return X[:, 0, :N]


def leapfrog() -> Frontend:
    """Use one three-state leapfrog as the shared twin reference."""
    af, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
    af.input = Sinusoid(0.5, 1e5)
    return af


@pytest.fixture(
    params=["nrz", "filtered dac", "trigger", "two columns"],
)
def af(request) -> Frontend:
    """Exercise immediate/filtered DACs, cyclic triggers, and J=2 batches."""
    af = leapfrog()
    if request.param == "filtered dac":
        af.dac = DAC(Constant(1.0)) >> Pole(0.1 * af.dt)
    elif request.param == "trigger":
        af = af.with_trigger(2, 0)
    elif request.param == "two columns":
        af.A = af.A * np.array([1.0, 1.01])[:, None, None, None]
    return af


@pytest.mark.parametrize("engine", ["numba", "numpy"])
def test_the_twin_identity_and_the_readout(af, engine):
    """Both kernels satisfy ``x = x_u + x_hat`` and ``y_hat = C_hat x_hat``.

    Compute the input-only state separately for each batch member, then
    subtract it and the twin state from the full frontend state.
    """
    sim = af.simulate(SIZE, record="all", engine=engine, verbose=False)
    for j in range(af.J):
        x_u = signal_alone(af[j], SIZE)
        residual = sim["x"][..., j] - sim["x_hat"][..., j] - x_u
        assert np.abs(residual).max() < 1e-10 * np.abs(x_u).max()
    np.testing.assert_allclose(
        sim["y_hat"][:, 0], -sim["x_hat"][:, -1], rtol=1e-14, atol=1e-14
    )  # C_hat = -e_N
    assert sim["y_hat"].shape == (SIZE, 1, af.J)


def test_record_selection_does_not_change_decisions(af):
    """Choosing fewer records must leave ``v``, ``u``, and ``y_hat`` unchanged."""
    minimal = af.simulate(SIZE, verbose=False)
    diagnostic = af.simulate(SIZE, record="all", verbose=False)
    states_only = af.simulate(SIZE, record="x", verbose=False)
    assert set(minimal) == {"v", "u", "y_hat"}
    assert set(diagnostic) == {"v", "y_hat", "t", "u", "x", "y", "x_hat"}
    assert set(states_only) == {"x"}
    np.testing.assert_array_equal(states_only["x"], diagnostic["x"])
    np.testing.assert_array_equal(minimal["v"], diagnostic["v"])
    np.testing.assert_array_equal(minimal["u"], diagnostic["u"])
    np.testing.assert_array_equal(minimal["y_hat"], diagnostic["y_hat"])


def test_an_explicit_readout_and_its_validation():
    """Explicit C_hat selects a twin state and rejects a wrong state count."""
    af = leapfrog()
    af.C_hat = [[0.0, 0.0, 1.0]]
    sim = af.simulate(SIZE, record=("x_hat", "y_hat"), verbose=False)
    assert sim["y_hat"].shape == (SIZE, 1, 1)
    assert np.array_equal(sim["y_hat"][:, 0], sim["x_hat"][:, 2])
    assert "C_hat (1, 3)" in str(af)
    with pytest.raises(ValueError):
        Frontend(af.A, af.B, af.C, dt=af.dt, C_hat=np.ones((1, 4)))


def test_twin_readout_can_vary_by_batch_and_phase():
    """For J=K=2, read the new state and previous decisions with phase i%K.

    Compare NumPy and Numba, then evaluate ``C_hat x_hat + D_hat v_prev``
    directly for each batch member and phase.
    """
    af = leapfrog()
    C_hat = np.zeros((2, 2, 1, af.N))
    C_hat[:, :, 0, -1] = [[-1.0, -2.0], [1.0, 2.0]]
    D_hat = np.zeros((2, 2, 1, af.M))
    D_hat[:, :, 0, 0] = [[0.25, -0.5], [0.75, -1.0]]
    af.C_hat = C_hat
    af.D_hat = D_hat
    numpy = af.simulate(
        64, record=("v", "x_hat", "y_hat"), engine="numpy", verbose=False
    )
    numba = af.simulate(64, record=("v", "x_hat", "y_hat"), verbose=False)
    np.testing.assert_array_equal(numba["v"], numpy["v"])
    np.testing.assert_allclose(numba["x_hat"], numpy["x_hat"], atol=1e-12)
    np.testing.assert_allclose(numba["y_hat"], numpy["y_hat"], atol=1e-12)
    for j in range(2):
        for i in range(64):
            np.testing.assert_allclose(
                numpy["y_hat"][i, 0, j],
                C_hat[j, i % 2, 0] @ numpy["x_hat"][i, :, j]
                + (D_hat[j, i % 2, 0] @ numpy["v"][i - 1, :, j] if i else 0),
            )
    assert af[1].D_hat.shape == (1, 2, 1, af.M)
    with pytest.raises(ValueError, match="D_hat must be"):
        Frontend(af.A, af.B, af.C, dt=af.dt, C_hat=C_hat, D_hat=np.ones((2, 2)))


def test_records_are_selected_explicitly():
    """A requested record set is exact, and an unknown name raises."""
    af = leapfrog()
    small = af.simulate(SIZE, record=("x", "t"), verbose=False)
    assert set(small) == {"x", "t"}
    with pytest.raises(ValueError, match="unknown simulation records"):
        af.simulate(SIZE, record="bad", verbose=False)


def test_the_blas_path_replays_the_twin():
    """The large-map BLAS path matches NumPy and keeps the twin identity.

    N=12 makes the augmented map exceed the packed-kernel limit. Compare
    decisions and twin records across engines, then check ``x-x_hat=x_u``.
    """
    N = 12
    af, _ = Frontend.leapfrog(N=N, OSR=32, BW=1e6)
    af.input = Sinusoid(0.5, 1e5)
    sim = af.simulate(SIZE, record=("v", "x", "x_hat", "y_hat"), verbose=False)
    reference = af.simulate(
        SIZE, record=("v", "x_hat", "y_hat"), engine="numpy", verbose=False
    )
    assert af.discretize().N_x + af.M > 24
    np.testing.assert_array_equal(sim["v"], reference["v"])
    for name in ("x_hat", "y_hat"):
        np.testing.assert_allclose(sim[name], reference[name], rtol=1e-9, atol=1e-9)
    x_u = signal_alone(af, SIZE)
    residual = np.abs(sim["x"][..., 0] - sim["x_hat"][..., 0] - x_u).max()
    assert residual < 1e-9 * np.abs(x_u).max()


def test_an_expanding_twin_never_reaches_the_frontend():
    """An overflowing open-loop twin must not alter bounded ``x`` or ``v``.

    Use a CIFB loop whose twin grows without bound, then compare decisions
    with a run that does not record ``x_hat``.
    """
    from cbadc import Samples, delsig

    ntf = delsig.synthesizeNTF(3, 32, opt=1, H_inf=1.5)
    ABCD = delsig.stuffABCD(*delsig.realizeNTF(ntf, "CIFB"), "CIFB")
    af = Frontend.dtsdm(ABCD, levels=2)
    af.input = Samples(0.3 * np.sin(2 * np.pi * np.arange(1 << 12) * 254 / (1 << 16)))
    size = 1 << 18
    with np.errstate(over="ignore", invalid="ignore"):
        sim, plain = (
            af.simulate(size, record=("v", "x", "x_hat"), verbose=False),
            af.simulate(size, verbose=False),
        )
    assert not np.isfinite(sim["x_hat"][-1]).all()  # it did overflow
    assert np.isfinite(sim["x"]).all() and np.array_equal(sim["v"], plain["v"])
