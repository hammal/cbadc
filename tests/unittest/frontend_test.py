"""Check source algebra, one-step ABCD maps, and clocked simulation."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from cbadc import frontend as fe
from cbadc import noise
from cbadc import snr as snr_mod
from cbadc.frontend import (
    DAC,
    Constant,
    Exp,
    Filter,
    Frontend,
    Pole,
    Poly,
    Samples,
    Sinusoid,
)
from cbadc.simulation import _W, _linear_loop, _numpy_loop

SIZE = 1 << 11


@pytest.fixture(scope="module")
def lf3():
    """One three-state leapfrog used by the simulation tests."""
    return Frontend.leapfrog(N=3, BW=1e6, ENOB=12.0)


def test_continuous_frontend_defaults_to_equal_area_rz():
    """CT uses a symmetric equal-area RZ pulse; DT uses immediate NRZ."""
    generic = Frontend([[0.0]], [[1.0, -1.0]], [[1.0]], dt=1e-6)
    leapfrog, _ = Frontend.leapfrog(N=3, OSR=10, BW=1e6)
    chain, _ = Frontend.chain_of_integrators(N=3, OSR=10, BW=1e6)
    assert [model.dac[0].window for model in (generic, leapfrog, chain)] == [
        (0.25, 0.5)
    ] * 3
    assert [model.dac[0].shape.x0[0, 0] for model in (generic, leapfrog, chain)] == [2.0] * 3
    discrete = Frontend([[1.0]], [[1.0, -1.0]], [[1.0]], dt=1e-6, discrete=True)
    assert discrete.dac[0].window is None
    assert discrete.dac[0].shape.x0[0, 0] == 1.0


@pytest.fixture
def af(lf3):
    """Give each test an independent sinusoid-driven leapfrog."""
    from copy import deepcopy

    af = deepcopy(lf3[0])
    f = snr_mod.coherent_frequency(af.fs / 64, af.fs, SIZE)
    af.input = Sinusoid(0.5, f)
    return af


def _snr(af, osr, size=SIZE, **kw):
    """Measure reconstructed tone SNR for an immediate-NRZ reference loop."""
    af = deepcopy(af)
    af.dac = DAC(Constant(1.0))  # legacy Wiener comparison assumes immediate NRZ
    f = snr_mod.coherent_frequency(af.fs / 64, af.fs, size)
    af.input = Sinusoid(0.5, f)
    v = af.simulate(size, **kw)["v"]
    u_hat = af.wiener_filter(OSR=osr)(v)
    return snr_mod.snr_tone(u_hat[:, 0, 0], f, af.fs, trim=512, band=af.fs / (2 * osr))


# ---------------------------------------------------------------- waveforms --


def test_generators_evaluate_to_their_closed_forms():
    """Constant, sinusoidal, exponential, and polynomial sources match formulas."""
    t = np.linspace(0.0, 3.0, 7)
    np.testing.assert_allclose(Constant(0.3).value(t)[:, 0], 0.3)
    np.testing.assert_allclose(
        Sinusoid(0.5, 0.25, 0.3).value(t)[:, 0],
        0.5 * np.cos(2 * np.pi * 0.25 * t + 0.3),
    )
    np.testing.assert_allclose(Exp(0.7).value(t)[:, 0], np.exp(-t / 0.7))
    np.testing.assert_allclose(
        Poly([1.0, -2.0, 0.5]).value(t)[:, 0], 1.0 - 2.0 * t + 0.5 * t**2
    )


def test_trajectory_matches_value_on_the_clock_grid():
    """Read the source's state trajectory through C and compare with ``value``."""
    w = Sinusoid([0.5, 0.2], [0.25, 0.1]) + Constant([0.1, -0.1])
    dt, size = 0.37, 100
    X = w.trajectory(dt, size)
    assert X.shape == (size, 2, 3)
    np.testing.assert_allclose(
        np.einsum("k,ijk->ij", w.C[0, 0, 0], X),
        w.value(np.arange(size) * dt),
        atol=1e-12,
    )


def test_series_composition_is_the_filtered_generator():
    """A sinusoid cascaded through a pole matches steady plus transient output."""
    tau = 0.3
    w = Sinusoid(1.0, 0.2) >> Pole(tau)
    t = np.linspace(0.0, 5.0, 11)
    omega = 2 * np.pi * 0.2
    H = 1.0 / (1 + 1j * omega * tau)  # steady state plus the transient from x_f(0) = 0
    steady = np.real(H * np.exp(1j * omega * t))
    transient = -np.real(H) * np.exp(-t / tau)
    np.testing.assert_allclose(w.value(t)[:, 0], steady + transient, atol=1e-12)


def test_parallel_composition_sums_outputs():
    """Parallel autonomous sources add their readouts at each time."""
    w = Exp(0.5) + Poly([0.0, 1.0])
    t = np.linspace(0.0, 2.0, 5)
    np.testing.assert_allclose(w.value(t)[:, 0], np.exp(-t / 0.5) + t)


def test_waveform_validation():
    """Reject invalid DAC shapes, windows, triggers, and frontend assignments."""
    with pytest.raises(ValueError):
        DAC(Pole(1.0))  # a filter is not a pulse shape
    with pytest.raises(ValueError):
        DAC(Constant(1.0), window=(-0.1, 1.0))
    with pytest.raises(ValueError):
        Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)[0].with_trigger(2, 2)
    np.testing.assert_allclose(Pole(1.0).value(np.zeros(2)), 0.0)  # x0 = 0: at rest
    with pytest.raises(ValueError):
        Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)[0].dac = Constant(1.0)


@pytest.mark.parametrize("tau", [0.0, -1.0, np.nan, np.inf])
@pytest.mark.parametrize("factory", [Exp, Pole])
def test_time_constant_must_be_positive_and_finite(factory, tau):
    """Exp and Pole require a finite, positive time constant."""
    with pytest.raises(ValueError, match="tau must be positive and finite"):
        factory(tau)


def test_state_space_dimensions_are_checked_at_construction():
    """StateSpace checks A/B/C/D/x0 dimensions and compatible J batches.

    First confirm scalar B and D broadcast; then perturb one matrix shape at
    a time and require construction to fail.
    """
    source = fe.StateSpace(np.eye(2), 0.0, [[1.0, 0.0]], 0.0)
    assert source.B.shape[-2:] == (2, 1)
    assert source.D.shape[-2:] == (1, 1)
    good = dict(
        A=np.eye(2),
        B=np.ones((2, 2)),
        C=np.ones((3, 2)),
        D=np.zeros((3, 2)),
        x0=np.zeros(2),
    )
    for change in (
        dict(A=np.ones((2, 3))),
        dict(B=np.ones((3, 2))),
        dict(C=np.ones((3, 3))),
        dict(D=np.zeros((2, 2))),
        dict(D=np.zeros((3, 1))),
        dict(x0=np.zeros(3)),
    ):
        with pytest.raises(ValueError, match="incompatible state-space dimensions"):
            fe.StateSpace(**(good | change))
    with pytest.raises(ValueError, match="incompatible J batch dimensions"):
        fe.StateSpace(
            np.broadcast_to(np.eye(2), (2, 1, 2, 2)),
            np.ones((3, 1, 2, 2)),
            good["C"],
            good["D"],
        )


def test_frontend_rejects_incompatible_batch_matrices_at_construction():
    """A and B cannot have incompatible numbers of J batch members."""
    with pytest.raises(ValueError, match="incompatible J batch dimensions"):
        Frontend(np.ones((2, 1, 1, 1)), np.ones((3, 1, 1, 2)), [[1.0]], dt=1.0)


# ------------------------------------------------------------ discretisation --


def _quadrature_column(A, Bm, dt, d, wd, profile):
    """Reference ``∫exp(A(dt-t)) B_m h(t) dt`` via high-accuracy ODE solve."""
    N, M = Bm.shape
    td, t1 = d * dt, (d + wd) * dt

    def rhs(t, x):
        return (A @ x.reshape(N, M) + Bm * profile(t - td)).ravel()

    x = solve_ivp(
        rhs, (td, t1), np.zeros(N * M), rtol=1e-13, atol=1e-20, method="DOP853"
    ).y[:, -1]
    return expm(A * (dt - t1)) @ x.reshape(N, M)


@pytest.mark.parametrize(
    "d, wd, wave, profile",
    [
        (0.0, 1.0, DAC(Constant(1.0)), lambda t: np.ones_like(t)),
        (0.25, 0.5, DAC(Constant(1.0), window=(0.25, 0.5)), lambda t: np.ones_like(t)),
        (0.1, 0.8, DAC(Exp(0.3), window=(0.1, 0.8)), lambda t: np.exp(-t / 0.3)),
        (0.0, 1.0, DAC(Poly([1.0, -1.0]), window=(0.0, 1.0)), lambda t: 1.0 - t),
        (
            0.2,
            0.6,
            DAC(Constant(0.5) + Sinusoid(0.5, 1.0 / 0.6, np.pi), window=(0.2, 0.6)),
            lambda t: 0.5 - 0.5 * np.cos(2 * np.pi * t / 0.6),
        ),
    ],
)
def test_dac_columns_match_quadrature(d, wd, wave, profile):
    """DAC columns of the exact step matrix match numerical quadrature.

    Compare constant, windowed, exponential, polynomial, and combined pulse
    shapes against an independent high-accuracy ODE solve.
    """
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    dt = af.dt
    af = af.with_dt(1.0)  # unit clock so the shapes above are in seconds
    af.dac = wave
    A, Bm = af.A[0, 0], af.B[0, 0, :, af.L :]
    ref = _quadrature_column(A, Bm, 1.0, d, wd, profile)
    np.testing.assert_allclose(
        af.discretize().ABCD[0, 0, : af.N, af.N :], ref, rtol=1e-11, atol=1e-14
    )
    assert dt != 1.0


def test_dac_filter_is_a_persistent_state():
    """A filtered DAC adds M persistent states to the augmented map.

    Build ``Constant >> Pole`` and compare its A and B step blocks with an
    independently assembled augmented zero-order-hold system.
    """
    af, _ = Frontend.leapfrog(N=2, BW=1e6, OSR=16.0)
    af = af.with_dt(1.0)
    tau = 0.4
    af.dac = DAC(Constant(1.0)) >> Pole(tau)
    dz = af.discretize()
    assert dz.N_x == af.N + af.M
    # reference: the augmented linear system [x; o] with o' = (v - o)/tau, ZOH in v
    N, M = af.N, af.M
    A_aug = np.block(
        [[af.A[0, 0], af.B[0, 0, :, af.L :]], [np.zeros((M, N)), -np.eye(M) / tau]]
    )
    B_aug = np.vstack((np.zeros((N, M)), np.eye(M) / tau))
    Phi, Gamma = fe._zoh(A_aug, B_aug, 1.0)
    np.testing.assert_allclose(dz.A_d[0, 0], Phi, atol=1e-13)
    np.testing.assert_allclose(dz.ABCD[0, 0, : dz.N_x, dz.N_x :], Gamma, atol=1e-13)


def test_one_step_map_matches_scalar_solution_with_source_delay_and_offset():
    """The joint one-step map matches a one-state analytic solution.

    Check the full-period input, half-period DAC drive, D feedthrough, and
    comparator offset in both the state and comparator rows.
    """
    decay, input_gain, feedback_gain = 0.7, 1.2, -0.8
    u, decision, offset = 0.3, 0.4, 0.07
    af = Frontend(
        [[-decay]],
        [[input_gain, feedback_gain]],
        [[1.5]],
        [[0.2, 0.1]],
        dt=1.0,
        input=Constant(u),
        dac=DAC(Constant(1.0), window=(0.5, 1.0)),
    )
    af.comparator_input_offset = [offset]
    dz = af.discretize()
    step = dz.ABCD[0, 0] @ np.r_[dz.x0[0], decision]

    # The input acts for the full period; this decision acts only after T/2.
    x1 = input_gain * u * (-np.expm1(-decay)) / decay
    x1 += feedback_gain * decision * (-np.expm1(-decay / 2)) / decay
    y1 = 1.5 * x1 + 0.2 * u + 0.1 * decision + offset
    np.testing.assert_allclose(step[0], x1, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(step[dz.N_x], y1, rtol=1e-13, atol=1e-14)


def test_pulse_longer_than_a_period_adds_latch_memory():
    """A half-period delayed NRZ pulse needs one step of decision memory.

    Check the memory shift in ABCD and integrate each half-period DAC
    contribution independently.
    """
    af, _ = Frontend.leapfrog(N=2, BW=1e6, OSR=16.0)
    af.dac = DAC(Constant(1.0)).with_window(0.5, 1.0)
    dz = af.discretize()
    assert dz.r == 1 and dz.N_x == af.N + af.M
    # the memory rows shift v[i-1] into the next step
    np.testing.assert_array_equal(
        dz.ABCD[0, 0, af.N : af.N + af.M, dz.N_x :], np.eye(af.M)
    )
    # the two pieces of the pulse add up to the full NRZ column
    nrz = (
        Frontend(af.A, af.B, af.C, dt=af.dt, dac=DAC(Constant(1.0)))
        .discretize()
        .ABCD[0, 0, : af.N, af.N :]
    )
    pieces = dz.ABCD[0, 0, : af.N, dz.N_x :] + dz.ABCD[
        0, 0, : af.N, af.N : af.N + af.M
    ] @ np.eye(af.M)
    # pieces are over different half periods, so compare charge through a brute-force sum instead
    half_a = _quadrature_column(
        af.A[0, 0], af.B[0, 0, :, af.L :], af.dt, 0.5, 0.5, lambda t: np.ones_like(t)
    )
    half_b = expm(af.A[0, 0] * 0.0) @ _quadrature_column(
        af.A[0, 0], af.B[0, 0, :, af.L :], af.dt, 0.0, 0.5, lambda t: np.ones_like(t)
    )
    np.testing.assert_allclose(dz.ABCD[0, 0, : af.N, dz.N_x :], half_a, rtol=1e-10)
    np.testing.assert_allclose(
        dz.ABCD[0, 0, : af.N, af.N : af.N + af.M], half_b, rtol=1e-10
    )
    assert pieces.shape == nrz.shape


def test_samples_through_a_filter_augments_the_frontend():
    """Samples through a Pole add one input-filter state to the step map.

    Compare the sampled-input column with a separately assembled augmented
    zero-order-hold model.
    """
    af, _ = Frontend.leapfrog(N=2, BW=1e6, OSR=16.0)
    af.dac = DAC(Constant(1.0))
    u = np.random.default_rng(1).uniform(-0.5, 0.5, 64)
    af.input = Samples(u) >> Pole(3 * af.dt)
    dz = af.discretize()
    assert dz.N_aux == 1
    # the filter state is driven by the held sample and feeds B_L
    A_aug = np.block(
        [
            [af.A[0, 0], af.B[0, 0, :, :1]],
            [np.zeros((1, af.N)), np.array([[-1 / (3 * af.dt)]])],
        ]
    )
    B_aug = np.vstack((np.zeros((af.N, 1)), np.array([[1 / (3 * af.dt)]])))
    _, Gamma = fe._zoh(A_aug, B_aug, af.dt)
    np.testing.assert_allclose(dz.B_u[0, 0], Gamma, atol=1e-13)


def test_samples_inherit_dac_shape_window_and_filter():
    """Samples retain their drive sequence through DAC shape, window, and filter."""
    values = np.array([1.0, -0.5, 0.25])
    samples = Samples(values, Exp(0.4)).with_window(0.25, 0.5) >> Pole(0.7)
    assert isinstance(samples, DAC)
    assert isinstance(samples, Samples)
    assert samples.window == (0.25, 0.5)
    assert samples.N == 2
    np.testing.assert_array_equal(samples.held(5)[:, 0], values[[0, 1, 2, 0, 1]])


def test_samples_algebra_preserves_the_drive_sequence():
    """Sample arithmetic acts elementwise and rejects incompatible windows."""
    a, b = Samples([1.0, 2.0]), Samples([0.5, -0.5])
    assert isinstance(a + b, Samples)
    np.testing.assert_array_equal((a + b).held(2)[:, 0], [1.5, 1.5])
    np.testing.assert_array_equal((2 * a).held(2)[:, 0], [2.0, 4.0])
    np.testing.assert_array_equal((a * b).held(2)[:, 0], [0.5, -1.0])
    with pytest.raises(ValueError, match="same pulse"):
        _ = a + b.with_window(0.25, 0.5)


@pytest.mark.parametrize("filtered", [False, True])
def test_windowed_samples_use_the_dac_pulse_integral(filtered):
    """The same windowed pulse integrates equally for sampled input and DAC.

    Compare the corresponding B_u and B_d columns, with and without a Pole
    filter after the pulse.
    """
    wave = Exp(0.4)
    samples = Samples([1.0, 0.0], wave).with_window(0.25, 0.5)
    dac = DAC(wave).with_window(0.25, 0.5)
    if filtered:
        samples, dac = samples >> Pole(0.7), dac >> Pole(0.7)
    af = Frontend([[-0.7]], [[1.0, 1.0]], [[1.0]], dt=1.0, input=samples, dac=dac)
    dz = af.discretize()
    np.testing.assert_allclose(dz.B_u[..., 0, 0], dz.B_d[..., 0, 1], atol=1e-14)


def test_sample_window_spanning_periods_affects_state_and_direct_output():
    """A delayed sample changes both the integrated state and direct D output.

    Compare five explicit x and y values as the one-period pulse overlaps
    adjacent clock periods.
    """
    samples = Samples([1.0, 2.0, 0.0]).with_window(0.5, 1.0)
    af = Frontend([[0.0]], [[1.0, 0.0]], [[1.0]], [[1.0, 0.0]], dt=1.0, input=samples)
    result = af.simulate(5, engine="numpy", record="all", verbose=False)
    np.testing.assert_allclose(result["x"][:, 0, 0], [0.0, 0.5, 2.0, 3.0, 3.5])
    np.testing.assert_allclose(result["y"][:, 0, 0], [0.0, 1.5, 4.0, 3.0, 4.5])


def test_silent_channels_and_triggers_hold_through_the_feedthrough():
    """Inactive comparator rows replay the previous decision.

    Confirm zero C/D rows become identity feedback in the discrete y block,
    then check held v values in the skipped phases.
    """
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    C = np.stack([af.C[0, 0], af.C[0, 0]])
    C[1, 2] = 0.0
    af.C = C
    af = af.with_trigger([1, 2, 1])
    dz = af.discretize()
    assert dz.K == 2
    assert not np.any(af.C[0, 1, 1:])
    assert not np.any(af.D[0, 1, 1:])
    assert (
        dz.ABCD[0, 1, dz.N_x + 1, dz.N_x + 1] == 1.0
        and dz.ABCD[0, 1, dz.N_x + 2, dz.N_x + 2] == 1.0
    )
    assert np.all(dz.ABCD[0, 1, dz.N_x + 1 :, : dz.N_x] == 0.0)
    af.input = Sinusoid(0.3, af.fs / 128)
    v = af.simulate(256)["v"]
    assert np.array_equal(v[1::2, 1], v[0:-1:2, 1]) and np.array_equal(
        v[1::2, 2], v[0:-1:2, 2]
    )


def test_zero_comparator_row_holds_per_frontend():
    """A zero C row holds v only for the batch member that has that row."""
    af = Frontend(
        [[0.5]],
        [[0.1, -0.1]],
        np.array([[[[1.0]]], [[[0.0]]]]),
        dt=1.0,
        discrete=True,
    )
    dz = af.discretize()
    assert dz.ABCD[0, 0, dz.N_x, 0] != 0.0
    assert dz.ABCD[1, 0, dz.N_x, dz.N_x] == 1.0


def test_time_interleaved_hold_matches_zero_rows():
    """Generated triggers equal a hand-written cyclic C with silent rows.

    Compare v and x for K=3, with one comparator active in each phase.
    """
    base, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    f = snr_mod.coherent_frequency(base.fs / 64, base.fs, 512)
    trig = Frontend(base.A, base.B, base.C, dt=base.dt, input=Sinusoid(0.5, f))
    trig = trig.with_trigger(3, [0, 1, 2])
    Cs = np.zeros((3, 3, 3))
    for k in range(3):
        Cs[k, k] = base.C[0, 0, k]
    rows = Frontend(base.A, base.B, Cs, dt=base.dt, input=Sinusoid(0.5, f))
    a, b = trig.simulate(512, record=("v", "x")), rows.simulate(512, record=("v", "x"))
    np.testing.assert_array_equal(a["v"], b["v"])
    np.testing.assert_allclose(a["x"], b["x"], atol=1e-12)


def _quantized(y: np.ndarray, levels: int) -> np.ndarray:
    """``Q(y)`` through a stateless model whose comparator sees the sample: ``y = u``."""
    af = Frontend([[0.0]], [[0.0, 0.0]], [[0.0]], [[1.0, 0.0]], dt=1.0, levels=levels)
    af.input = Samples(y)
    return af.simulate(y.size, verbose=False)["v"][:, 0, 0]


def test_multilevel_quantizer_is_idempotent_on_its_codes():
    """Q maps each valid code to itself and uses every level over full scale."""
    for n in (2, 3, 4, 5):
        codes = np.arange(-(n - 1), n, 2) / (n - 1)
        np.testing.assert_array_equal(_quantized(codes, n), codes)
    # full scale resolves every code
    assert np.unique(_quantized(np.linspace(-1, 1, 2001), 5)).size == 5


def test_discretization_reports_stiffness_and_warns():
    """Diagnostics distinguish normal, stiff, and numerically invalid expm.

    Check one-step self-consistency for stable A, then require a detailed
    warning when an unstable A overflows its matrix exponential.
    """
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    d = af.discretize().diagnostics
    assert d["selfcheck"] < 1e-12 and d["radius"] == pytest.approx(1.0, abs=1e-9)
    A = np.array([[-1.0, 0.0], [1.0, -1e9]])
    stiff = Frontend(A, np.array([[1.0, 1.0], [0.0, 1.0]]), np.eye(2)[:1], dt=1.0)
    d = stiff.discretize().diagnostics  # expm copes with a stiff stable system
    assert d["stiffness"] > 1e8 and d["selfcheck"] < 1e-10
    blowup = Frontend(
        np.array([[0.0, 0.0], [0.0, 2e3]]),
        np.array([[1.0, 1.0], [0.0, 1.0]]),
        np.eye(2)[:1],
        dt=1.0,
    )
    with pytest.warns(
        UserWarning, match="matrix-exponential self-check failed"
    ) as record:
        blowup.discretize()
    message = next(
        str(w.message)
        for w in record
        if "matrix-exponential self-check failed" in str(w.message)
    )
    assert "relative Frobenius difference" in message
    assert "limit 1e-10" in message
    assert "stiffness ratio=" in message
    assert "one-step matrices finite=False" in message


# ---------------------------------------------------------------- simulation --


def _loops_agree(af, exact):
    """Compare NumPy and Numba recursions through their first decision mismatch.

    A threshold-roundoff flip can make later paths diverge; large BLAS maps
    must instead be bit-identical when ``exact`` is requested.
    """
    a = af.simulate(SIZE, record="all", engine="numpy")
    b = af.simulate(SIZE, record=("v", "x"))
    if exact:
        np.testing.assert_array_equal(a["v"], b["v"])
        np.testing.assert_array_equal(a["x"], b["x"])
        return
    same = np.all(a["v"] == b["v"], axis=(1, 2))
    first = SIZE if same.all() else int(np.argmin(same))
    assert first >= 512, f"decisions part at period {first}"
    if first < SIZE:  # only a comparator input within roundoff of zero may flip
        assert np.abs(a["y"][first]).min() < 1e-9
    np.testing.assert_allclose(a["x"][:first], b["x"][:first], rtol=0, atol=1e-11)


def test_numpy_and_numba_loops_agree_on_a_small_model(af):
    """Small explicit-sum kernels match NumPy under clipping and offsets."""
    af.state_max, af.state_min = 0.95 * np.ones(af.N), -0.95 * np.ones(af.N)
    af.comparator_input_offset = np.array([0.05, 0.0, -0.02])
    _loops_agree(
        af, exact=False
    )  # explicit loops: the summation order differs from BLAS


def test_numpy_and_numba_loops_agree_bit_for_bit_on_a_large_model():
    """A 28-row BLAS map gives identical Numba and NumPy records."""
    af, _ = Frontend.leapfrog(N=14, BW=1e6, ENOB=12.0)  # n + M = 28: the BLAS path
    af.input = Sinusoid(0.5, snr_mod.coherent_frequency(af.fs / 64, af.fs, SIZE))
    af.state_max, af.state_min = 0.95 * np.ones(af.N), -0.95 * np.ones(af.N)
    _loops_agree(af, exact=True)


def test_recorded_y_is_the_quantizer_input(af):
    """Recorded y equals Cx and its sign determines the binary decision v."""
    res = af.simulate(300, record="all")
    np.testing.assert_array_equal(np.where(res["y"] < 0, -1.0, 1.0), res["v"])
    np.testing.assert_allclose(
        res["y"][1:], np.einsum("mn,inj->imj", af.C[0, 0], res["x"][1:]), atol=1e-12
    )


def test_simulation_reaches_the_design_snr(lf3):
    """The synthesized N=3 loop reaches its target in-band reconstructed SNR."""
    af, osr = lf3
    assert _snr(af, int(np.ceil(osr))) > 68.0


def test_exact_sinusoid_and_held_samples_agree_to_the_zoh_error(af):
    """A held sample approximation tracks the continuous sinusoid briefly.

    Feed the recorded u back as Samples, then compare initial states and a
    basic decision agreement rate before chaotic feedback paths diverge.
    """
    res = af.simulate(SIZE, record=("v", "u", "x"))
    held = Frontend(af.A, af.B, af.C, dt=af.dt, input=Samples(res["u"][:, 0, :]))
    res2 = held.simulate(SIZE, record=("v", "x"))
    assert (
        np.mean(res["v"] == res2["v"]) > 0.5
    )  # chaotic after the first flip, so only a sanity bound
    np.testing.assert_allclose(res["x"][:8], res2["x"][:8], atol=0.05)


def test_parallel_columns_are_independent_simulations(lf3):
    """Two J input tones run independently and match separate Frontends."""
    af, _ = lf3
    f = snr_mod.coherent_frequency(af.fs / 64, af.fs, 256)
    af = Frontend(af.A, af.B, af.C, dt=af.dt, input=Sinusoid([0.5, 0.2], [f, 2 * f]))
    both = af.simulate(256, record=("v", "x"))
    for j, (a, fj) in enumerate(((0.5, f), (0.2, 2 * f))):
        single = Frontend(af.A, af.B, af.C, dt=af.dt, input=Sinusoid(a, fj)).simulate(
            256, record=("v", "x")
        )
        np.testing.assert_array_equal(both["v"][:, :, j], single["v"][:, :, 0])
        np.testing.assert_allclose(both["x"][:, :, j], single["x"][:, :, 0], atol=1e-12)


def test_swing_clips_the_state_and_corrects_the_quantizer_input(af):
    """Clipping x also corrects y so the quantizer sees C times clipped x."""
    af.state_max, af.state_min = 0.3 * np.ones(af.N), -0.3 * np.ones(af.N)
    res = af.simulate(512, record="all")
    assert np.abs(res["x"]).max() <= 0.3 + 1e-12
    np.testing.assert_allclose(
        res["y"][1:], np.einsum("mn,inj->imj", af.C[0, 0], res["x"][1:]), atol=1e-12
    )


def test_slew_limits_the_state_change(af):
    """Per-state slew rate caps ``|x[i]-x[i-1]|`` each period."""
    af.slew_rate = 0.05 * af.fs * np.ones(af.N)  # at most 0.05 per period
    res = af.simulate(512, record="x")
    assert np.abs(np.diff(res["x"], axis=0)).max() <= 0.05 + 1e-12


def test_comparator_input_offset_shifts_the_state_not_the_decisions(af):
    """Feedback absorbs a y offset in mean x while mean v stays near zero."""
    af.comparator_input_offset = np.array([0.5, 0.0, 0.0])
    res = af.simulate(2048, record=("v", "x"))
    assert abs(res["v"][:, 0].mean()) < 0.1
    assert res["x"][256:, 0].mean() == pytest.approx(-0.5, abs=0.1)


def test_initial_state_is_honoured(af):
    """The first recorded state equals the supplied x0 vector."""
    x0 = np.array([0.1, -0.2, 0.3])
    res = af.simulate(4, x0=x0, record="x")
    np.testing.assert_array_equal(res["x"][0, :, 0], x0)


def test_with_dt_keeps_the_decisions(lf3):
    """Rescaling CT time and input frequency preserves v, but DT rejects it."""
    af, _ = lf3
    unit = af.with_dt(1.0)
    f = 1.0 / 64
    unit.input = Sinusoid(0.5, f)
    fast = unit.with_dt(1e-6)
    fast.input = Sinusoid(0.5, f * 1e6)
    np.testing.assert_array_equal(unit.simulate(256)["v"], fast.simulate(256)["v"])
    assert fast.fs == pytest.approx(1e6)
    with pytest.raises(ValueError):
        Frontend(np.eye(1), np.ones((1, 2)), np.eye(1), dt=1.0, discrete=True).with_dt(
            2.0
        )


# --------------------------------------------------------------------- noise --


@pytest.fixture
def rc():
    """One RC state with continuous-time decay and input gain 1/(RC)."""
    R, C = 10e3, 1e-12
    af = Frontend(
        [[-1 / (R * C)]], [[1 / (R * C), 0.0]], np.zeros((1, 1)), dt=R * C / 4
    )
    return af, R, C


def test_rc_noise_settles_to_kT_over_C(rc):
    """An RC state reaches thermal variance kT/C by either noise API.

    Compare a state-noise intensity and the same resistor noise referred
    through the input column of B after settling.
    """
    af, R, C = rc
    af.state_covariance = noise.per_state_intensity(
        [noise.resistor_density(R) / (R * C)]
    )
    var = af.simulate(1 << 14, record="x")["x"][1024:, 0, 0].var()
    assert var == pytest.approx(noise.k_B * 300.0 / C, rel=0.1)
    # input-referred: the same resistor noise at the input port
    af.state_covariance = af.input_referred_covariance(
        [[noise.resistor_density(R) ** 2]]
    )
    var = af.simulate(1 << 14, record="x")["x"][1024:, 0, 0].var()
    assert var == pytest.approx(fe.k_B * 300.0 / C, rel=0.1)


def test_discrete_noise_cov_is_the_lyapunov_increment():
    """Q_d matches the covariance ODE integrated over one clock period.

    Compare the matrix-exponential result with ``P'=AP+PA.T+Q`` solved
    independently at high accuracy.
    """
    A = np.array([[-1.0, 0.2], [0.0, -3.0]])
    Q = np.diag([1.0, 0.5])
    dt = 0.3
    Qd = noise.discrete_process_noise_cov(A, Q, dt)
    ref = (
        solve_ivp(
            lambda t, p: (A @ p.reshape(2, 2) + p.reshape(2, 2) @ A.T + Q).ravel(),
            (0, dt),
            np.zeros(4),
            rtol=1e-12,
            atol=1e-14,
        )
        .y[:, -1]
        .reshape(2, 2)
    )
    np.testing.assert_allclose(Qd, ref, rtol=1e-9)


@pytest.mark.parametrize("ratio", [10, 1e3, 1e4])
def test_discrete_noise_cov_survives_stiffness(ratio):
    """Q_d remains accurate for stiff A where a direct Van Loan map strains.

    Compare against an implicit high-accuracy covariance ODE solve.
    """
    A = np.array([[-1.0, 0.0], [5.0, -ratio]])
    Q = np.eye(2)
    dt = 30.0 / ratio * ratio  # |A dt| = 30 ratio: far beyond plain van Loan
    Qd = noise.discrete_process_noise_cov(A, Q, dt)
    ref = (
        solve_ivp(
            lambda t, p: (A @ p.reshape(2, 2) + p.reshape(2, 2) @ A.T + Q).ravel(),
            (0, dt),
            np.zeros(4),
            rtol=1e-12,
            atol=1e-16,
            method="Radau",
        )
        .y[:, -1]
        .reshape(2, 2)
    )
    np.testing.assert_allclose(Qd, ref, rtol=1e-6)


def test_noise_degrades_snr_predictably(lf3):
    """Input-referred state noise lowers reconstructed SNR near 60 dB."""
    af, osr = lf3
    osr = int(np.ceil(osr))
    noisy = Frontend(af.A, af.B, af.C, dt=af.dt)
    noisy.state_covariance = noisy.input_referred_covariance([[1.25e-13]])
    assert _snr(noisy, osr) == pytest.approx(60.0, abs=2.5)


def test_removed_polynomial_coefficients_raise(af):
    """Removed nonlinear coefficients and the old substep option fail visibly."""
    for name in ("a2", "a3"):
        with pytest.raises(AttributeError, match="not supported"):
            setattr(af, name, 0.1)
    with pytest.raises(TypeError):
        af.simulate(8, n_sub=4)


def test_removed_clock_jitter_cannot_be_silently_assigned(af):
    """Removed jitter attributes raise instead of silently doing nothing."""
    for name in ("clock_jitter", "comparator_jitter"):
        with pytest.raises(AttributeError, match="not supported"):
            setattr(af, name, np.zeros(2))


# ---------------------------------------------------------- discrete / delsig --


def test_discrete_time_model_steps_its_matrices():
    """The DT recurrence uses A x[i-1] + B_u u[i] + B_v v[i-1].

    Simulate a two-state model and evaluate that matrix equation directly
    at every later step.
    """
    A = np.array([[1.0, 0.0], [0.5, 1.0]])
    B = np.array([[0.5, -0.5, 0.0], [0.0, 0.0, -0.5]])
    af = Frontend(
        A, B, np.eye(2), dt=1.0, discrete=True, input=Samples(0.3 * np.ones(8))
    )
    res = af.simulate(8, record=("v", "x"))
    x, v = res["x"][:, :, 0], res["v"][:, :, 0]
    for i in range(1, 8):
        np.testing.assert_allclose(
            x[i], A @ x[i - 1] + B[:, :1] @ [0.3] + B[:, 1:] @ v[i - 1], atol=1e-15
        )


def test_dtsdm_matches_the_toolbox_simulator():
    """A DELSIG ABCD gives identical binary and ternary decision streams.

    Normalize DELSIG codes to cbadc full scale, then compare every v sample.
    """
    from cbadc import delsig

    for nlev in (2, 3):
        ntf = delsig.synthesizeNTF(3, 32, 1, 1.5)
        a, g, b, c = delsig.realizeNTF(ntf, "CRFB")
        ABCD = delsig.stuffABCD(a, g, b, c, "CRFB")
        u = 0.5 * (nlev - 1) * np.sin(2 * np.pi * 0.01 * np.arange(512))
        v_ref, *_ = delsig.simulateDSM(u, ABCD, nlev=nlev)
        af = Frontend.dtsdm(ABCD, levels=nlev, L=1, M=1)
        af.input = Samples(u / (nlev - 1))
        v = af.simulate(512)["v"][:, 0, 0] * (nlev - 1)
        np.testing.assert_array_equal(v, np.asarray(v_ref).ravel())


def test_transfer_function_matches_a_direct_evaluation(lf3):
    """Frequency response equals ``C(sI-A)^-1 B + D`` at two frequencies."""
    af, _ = lf3
    jw = 1j * 2 * np.pi * np.array([1e5, 1e6])
    _, H = af.transfer_function(jw)
    for i, s in enumerate(jw):
        ref = (
            af.C[0, 0] @ np.linalg.solve(s * np.eye(af.N) - af.A[0, 0], af.B[0, 0])
            + af.D[0, 0]
        )
        np.testing.assert_allclose(H[i], ref)


def test_str_is_informative(lf3):
    """Frontend and discretization summaries expose dimensions and stiffness."""
    af, _ = lf3
    assert str(af).startswith("Frontend(J=1, K=1, L=1, M=3, N=3")
    assert "stiffness" in str(af.discretize())


def test_with_flicker_filters_a_record_without_adding_states():
    """Flicker noise adds a 1/f-like PSD without adding frontend states.

    Subtract a same-seed white-noise run, fit the residual PSD slope, and
    check time-scale conversion and missing-intensity validation.
    """
    from scipy.signal import welch

    af = Frontend([[0.0]], [[0.0, 0.0]], [[1.0]], dt=1.0)
    af.state_covariance = np.array([[1e-6]])
    fl = af.with_flicker(0, f_corner=1e-3)
    assert fl.N == af.N and fl.discretize().N_x == af.discretize().N_x
    assert fl.simulate(1, record="x", verbose=False)["x"].shape == (1, 1, 1)
    size = 1 << 16
    base = af.simulate(size, record="x", seed=17, verbose=False)["x"][:, 0, 0]
    noisy = fl.simulate(size, record="x", seed=17, verbose=False)["x"][:, 0, 0]
    flicker = np.diff(noisy - base)
    f, P = welch(flicker, nperseg=1 << 14)
    band = (f > 1e-3) & (f < 0.1)
    slope = np.polyfit(np.log10(f[band]), np.log10(P[band]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.12)
    np.testing.assert_allclose(fl.with_dt(2.0)._flicker[0][0], 5e-4)
    with pytest.raises(ValueError, match="intensity"):
        Frontend([[0.0]], [[0.0, 0.0]], [[1.0]], dt=1.0).with_flicker(0, 1e-3)


def test_simulation_seed_selects_noise_draw():
    """The same seed repeats state noise; a different seed changes it."""
    af = Frontend([[0.5]], [[0.0, 0.0]], [[1.0]], dt=1.0, discrete=True)
    af.state_covariance = np.array([[0.01]])
    first = af.simulate(32, record="x", seed=7, verbose=False)["x"]
    repeated = af.simulate(32, record="x", seed=7, verbose=False)["x"]
    different = af.simulate(32, record="x", seed=8, verbose=False)["x"]
    np.testing.assert_array_equal(first, repeated)
    assert np.any(first != different)


def test_state_covariance_batch_and_phase_axes():
    """Noise alone may vary over J and K; selection retains its matching row."""
    af = Frontend([[0.0]], [[0.0, 0.0]], [[1.0]], dt=1.0, discrete=True)
    af.state_covariance = np.array([[[[1.0]], [[4.0]]], [[[9.0]], [[16.0]]]])
    assert af.state_covariance.shape == (2, 2, 1, 1)
    assert (af.J, af.K) == (2, 2)
    F = af.discretize().noise_factor
    np.testing.assert_allclose(F[:, :, 0, 0] ** 2, [[1.0, 4.0], [9.0, 16.0]])
    np.testing.assert_array_equal(af[1].state_covariance[0], af.state_covariance[1])
    with pytest.raises(ValueError, match="state_covariance must be symmetric"):
        af.state_covariance = np.zeros((2, 1, 2))


def test_state_covariance_cyclic_input_accepts_three_axes():
    """A (K, N, N) intensity is shared across batch members."""
    af = Frontend([[0.0]], [[0.0, 0.0]], [[1.0]], dt=1.0, discrete=True)
    af.state_covariance = np.array([[[1.0]], [[4.0]]])
    assert af.state_covariance.shape == (1, 2, 1, 1)
    np.testing.assert_allclose(af.discretize().noise_factor[0, :, 0, 0], [1.0, 2.0])


# ------------------------------------------------------------ batch axis on ABCD --


def test_frequency_sweep_runs_one_matrix_per_column():
    """Distinct J tone frequencies require distinct one-step matrices.

    Compare each batch member's u, v, and x with an isolated frontend at
    the same frequency; the source's A and x0 must also select correctly.
    """
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    freqs = np.array([1e5, 2e5, 3e5])
    af.input = Sinusoid(0.5, freqs)
    dz = af.discretize()
    assert dz.J_m == 3 and dz.N_x == af.N + 2
    batch = af.simulate(SIZE, record=("v", "u", "x"))
    for j, f in enumerate(freqs):
        single, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
        single.input = Sinusoid(0.5, f)
        one = single.simulate(SIZE, record=("v", "u", "x"))
        selected = af[j]
        assert selected.J == 1
        np.testing.assert_array_equal(selected.input[0].A, single.input[0].A)
        np.testing.assert_array_equal(selected.input[0].x0, single.input[0].x0)
        np.testing.assert_array_equal(selected.simulate(SIZE)["v"], one["v"])
        np.testing.assert_array_equal(one["v"][:, :, 0], batch["v"][:, :, j])
        np.testing.assert_allclose(one["x"][:, :, 0], batch["x"][:, :, j], atol=1e-12)
        np.testing.assert_allclose(one["u"][:, :, 0], batch["u"][:, :, j], atol=1e-12)


def test_amplitude_sweep_shares_the_matrix():
    """J tone amplitudes at one frequency share a single ABCD map."""
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    af.input = Sinusoid([0.1, 0.3, 0.5], 1e5)
    assert af.discretize().J_m == 1 and af.J == 3


def test_batch_index_selects_complete_cycle_and_sample_column():
    """``af[j]`` copies a complete K-cycle, source column, x0, and C_hat.

    Compare each isolated run with the corresponding J record; then check
    the copy is independent and invalid batch/phase indices raise.
    """
    A = np.array([[[[-0.2]], [[-0.3]]], [[[-0.4]], [[-0.5]]]])
    values = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]])
    af = Frontend(
        A,
        [[1.0, -0.25]],
        [[1.0]],
        dt=1.0,
        discrete=True,
        input=Samples(values),
        C_hat=np.array([[[[1.0]]], [[[2.0]]]]),
    )
    af.x0 = np.array([[0.1], [0.2]])
    batch = af.simulate(9, record="all", verbose=False)
    assert af.J == 2 and af.K == 2
    for j in range(af.J):
        one = af[j]
        assert one.J == 1 and one.K == 2
        np.testing.assert_array_equal(one.A, af.A[j : j + 1])
        np.testing.assert_array_equal(one.C_hat, af.C_hat[j : j + 1])
        np.testing.assert_array_equal(one.x0, af.x0[j : j + 1])
        np.testing.assert_array_equal(one.input[0].values[:, 0], values[:, j])
        result = one.simulate(9, record="all", verbose=False)
        for key in ("u", "v", "x", "y", "y_hat"):
            np.testing.assert_allclose(result[key][:, :, 0], batch[key][:, :, j])
    last = af[-1]
    np.testing.assert_array_equal(last.input[0].values[:, 0], values[:, 1])
    last.A[0, 0, 0, 0] = -9.0
    last.input[0].values[0, 0] = -9.0
    assert af.A[1, 0, 0, 0] == -0.4 and af.input[0].values[0, 1] == 0.5
    with pytest.raises(IndexError):
        _ = af[2]
    with pytest.raises(TypeError, match="clock phase"):
        _ = af[0, 1]


def test_batch_index_selects_source_readout_column():
    """Batch selection copies the source C readout as well as its dynamics.

    A J=2 source has different gains and x0; compare its u, v, and x with
    each selected frontend.
    """
    source = fe.StateSpace(
        [[0.0]], 0.0, np.array([[[1.0]], [[2.0]]]), 0.0, [[0.2], [0.3]]
    )
    af = Frontend([[-0.2]], [[1.0, -0.1]], [[1.0]], dt=1.0, input=source)
    batch = af.simulate(6, record=("v", "u", "x"), verbose=False)
    np.testing.assert_allclose(batch["u"][:, 0], [[0.2, 0.6]] * 6)
    for j in range(af.J):
        one = af[j]
        np.testing.assert_array_equal(one.input[0].C, source.C[j : j + 1])
        result = one.simulate(6, record=("v", "u", "x"), verbose=False)
        for key in ("u", "v", "x"):
            np.testing.assert_allclose(result[key][:, :, 0], batch[key][:, :, j])


def test_frontend_monte_carlo_runs_one_frontend_per_batch():
    """J perturbed A matrices run as independent Monte Carlo frontends.

    Compare batched v and x with each selected frontend and reject a
    single-front-end transfer function for the whole batch.
    """
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    af.input = Sinusoid(0.5, 1e5)
    rng = np.random.default_rng(3)
    spread = 1.0 + 0.05 * rng.standard_normal((4, af.N, af.N))
    mc = Frontend(af.A * spread[:, None], af.B, af.C, dt=af.dt, input=af.input)
    assert mc.J_frontend == 4 and mc.J == 4 and mc.discretize().J_m == 4
    batch = mc.simulate(SIZE, record=("v", "x"))
    for j in range(4):
        one = mc[j]
        assert one.A.shape[:2] == (1, 1) and one.J == 1
        res = one.simulate(SIZE, record=("v", "x"))
        np.testing.assert_array_equal(res["v"][:, :, 0], batch["v"][:, :, j])
        np.testing.assert_allclose(res["x"][:, :, 0], batch["x"][:, :, j], atol=1e-12)
    with pytest.raises(ValueError, match="one frontend"):
        mc.transfer_function(2j * np.pi * np.array([1e5]))


def test_batch_and_cyclic_axes_together_match_individual_frontends():
    """The J-specific and K-cyclic maps select correctly in both engines.

    Combine three A matrices with two trigger phases, then compare NumPy,
    Numba, and individual frontend records.
    """
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    spread = np.array([0.98, 1.0, 1.02])[:, None, None, None]
    mc = Frontend(af.A * spread, af.B, af.C, dt=af.dt, input=Sinusoid(0.3, 1e5))
    mc = mc.with_trigger([2, 2, 1], [0, 1, 0])
    assert mc.discretize().ABCD.shape[:2] == (3, 2)
    batch = mc.simulate(128, record="all", verbose=False)
    reference = mc.simulate(128, record="all", engine="numpy", verbose=False)
    np.testing.assert_array_equal(batch["v"], reference["v"])
    np.testing.assert_allclose(batch["x"], reference["x"], atol=1e-12)
    for j in range(3):
        one = mc[j].simulate(128, record="all", verbose=False)
        np.testing.assert_array_equal(batch["v"][:, :, j], one["v"][:, :, 0])
        np.testing.assert_allclose(batch["x"][:, :, j], one["x"][:, :, 0], atol=1e-12)


def test_large_per_frontend_contraction_matches_numpy_with_twin():
    """Packed 25–64-row maps reproduce the NumPy frontend and twin.

    Give more than one Numba block distinct A matrices and compare v, x, y,
    x_hat, and y_hat across the two engines.
    """
    af, _ = Frontend.leapfrog(N=12, BW=1e6, OSR=16.0)
    rng = np.random.default_rng(23)
    spread = 1.0 + 0.01 * rng.standard_normal((_W + 3, af.N, af.N))
    mc = Frontend(
        af.A * spread[:, None], af.B, af.C, dt=af.dt, input=Sinusoid(0.2, 1e5)
    )
    dz = mc.discretize()
    assert 24 < dz.N_x + mc.M <= 64 and dz.J_m == mc.J == _W + 3
    packed = mc.simulate(64, record="all", threads=2, verbose=False)
    reference = mc.simulate(64, record="all", engine="numpy", verbose=False)
    np.testing.assert_array_equal(packed["v"], reference["v"])
    for name in ("x", "y", "x_hat", "y_hat"):
        # The high-order recurrence amplifies last-bit differences in sum order.
        np.testing.assert_allclose(packed[name], reference[name], rtol=1e-7, atol=3e-8)


@pytest.mark.parametrize("J", [2, _W + 3])
def test_per_frontend_blas_reuses_columns_and_matches_numpy(J):
    """Per-frontend BLAS above 64 rows agrees with the NumPy loop.

    Build J distinct synthetic step matrices and matching twin maps;
    compare decisions and twin readout for short runs.
    """
    N = M = 33
    rows, K, size = N + M, 1, 8
    step = np.zeros((J, K, rows, rows))
    twin = np.zeros((J, K, N + 1, rows))
    for j in range(J):
        step[j, 0, :N, :N] = 0.9 * np.eye(N)
        step[j, 0, :N, N:] = 0.01 * (j + 1)
        step[j, 0, N:, :N] = np.eye(N)
        twin[j, 0, :N] = step[j, 0, :N]
        twin[j, 0, N, :N] = np.ones(N) / N
    xv0 = np.zeros((rows, J))
    xv0[0] = np.linspace(-0.2, 0.2, J)

    def run(kernel):
        v = np.empty((size, M, J))
        y_hat = np.empty((size, 1, J))
        args = (
            step,
            twin,
            np.zeros((0, rows, J)),
            np.zeros((J, K, M, N)),
            xv0,
            v,
            np.zeros((0, N, J)),
            np.zeros((0, N, J)),
            np.zeros((0, M, J)),
            y_hat,
            -np.ones(N),
            np.ones(N),
            np.ones((K, N)),
            np.full(M, 2.0),
            False,
        )
        if kernel is _linear_loop:
            dummy = np.zeros((1, K, 1, 1, _W))
            kernel(*args[:2], dummy, dummy, *args[2:])
        else:
            kernel(*args)
        return v, y_hat

    actual_v, actual_y_hat = run(_linear_loop)
    expected_v, expected_y_hat = run(_numpy_loop)
    np.testing.assert_array_equal(actual_v, expected_v)
    np.testing.assert_allclose(actual_y_hat, expected_y_hat, atol=1e-12)


@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("per_member", [False, True])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("size", [1, 19])
def test_precomputed_excitation_matches_stepwise_sum(K, per_member, dense, size):
    """Precomputation equals B_u u + F w and C e_z + D_L u + G n.

    Compare every step/member with ordinary matrix-vector products, covering
    diagonal and dense C, shared and member-specific noise, and cyclic phases.
    The autonomous sine adds augmented states that must receive no white noise.
    """
    J, N, M = 3, 2, 2
    C = np.diag([2.0, 3.0])
    if dense:
        C += np.array([[0.0, 0.2], [-0.1, 0.0]])
    scale = np.arange(1, J + 1) if per_member else np.ones(1)
    af = Frontend(
        np.repeat((-10 * np.eye(N))[None], K, axis=0),
        np.array([[0.7, 0.1, -1.0, 0.0], [-0.2, 0.2, 0.0, -1.0]]),
        scale[:, None, None, None] * C,
        [[0.1, 0.0, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0]],
        dt=0.01,
    )
    samples = np.random.default_rng(1).normal(size=(size, J))
    af.input = [Samples(samples), Sinusoid(0.2, 3.0)]
    af.state_covariance = (
        scale[:, None, None, None]
        * np.arange(1, K + 1)[None, :, None, None]
        * np.array([[1.0, 0.2], [0.2, 2.0]])
    )
    af.output_covariance = [[0.03, 0.01], [0.01, 0.04]]
    dz = af.discretize()
    excitation = np.full((size, dz.N_x + M, J), np.nan)
    dz.precompute_excitation(excitation, np.random.default_rng(7))

    rng = np.random.default_rng(7)
    w = rng.normal(size=(size - 1, N, J))
    n = rng.normal(size=(size, M, J))
    G = fe.positive_semidefinite(af.output_covariance)
    expected = np.zeros_like(excitation)
    for i in range(size):
        k = i % K
        for j in range(J):
            j_m = j if per_member else 0
            e_z = expected[i, : dz.N_x, j]
            if i:
                e_z[:] = (
                    dz.B_u[j_m, k, :, 0] * samples[i - 1, j]
                    + dz.noise_factor[j_m, k] @ w[i - 1, :, j]
                )
            expected[i, dz.N_x :, j] = (
                dz.C[j_m, k] @ e_z
                + dz.D_L[j_m, k, :, 0] * samples[i, j]
                + G @ n[i, :, j]
            )
    np.testing.assert_allclose(excitation, expected, rtol=1e-13, atol=1e-15)


def test_offsets_and_generators_leave_no_excitation():
    """Deterministic sources and offsets need no per-step excitation array.

    Precomputation must reject an empty stochastic workload; adding state
    covariance makes later excitation rows nonzero.
    """
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    af.input = Sinusoid(0.5, 1e5)
    af.comparator_input_offset = np.array([0.05, 0.0, -0.02])
    dz = af.discretize()
    assert dz.N_z == 3  # two tone states and the offset state
    excitation = np.ones((64, dz.N_x + af.M, 1))
    with pytest.raises(ValueError, match="no sampled input or noise"):
        dz.precompute_excitation(excitation, np.random.default_rng(0))
    af.state_covariance = 1e-9 * np.eye(af.N)
    af.discretize().precompute_excitation(excitation, np.random.default_rng(0))
    assert np.all(excitation[0] == 0.0)
    assert np.any(excitation[1:] != 0.0)


# -- the simulation plan: message, records, Monte Carlo threads ----------------


def _mc(J: int = 2 * _W + 3) -> Frontend:
    """J leap-frogs with their own matrices: an odd J leaves a narrow last block."""
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    rng = np.random.default_rng(7)
    spread = 1.0 + 0.05 * rng.standard_normal((J, af.N, af.N))
    return Frontend(
        af.A * spread[:, None], af.B, af.C, dt=af.dt, input=Sinusoid(0.5, 1e5)
    )


def test_simulate_announces_dimensions_memory_and_time(capsys):
    """The simulation plan reports dimensions, records, and runtime path.

    Check quiet mode and verify a J batch also reports threaded execution.
    """
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    af.input = Sinusoid(0.5, 1e5)
    af.simulate(64)
    err = capsys.readouterr().err
    assert err.startswith("simulate: J=1 K=1 L=1 M=3 N=3 size=64")
    assert (
        "numba" in err
        and "v " in err
        and "u " in err
        and "y_hat " in err
        and err.rstrip().endswith("ms")
    )
    af.simulate(64, verbose=False)
    assert capsys.readouterr().err == ""
    mc = _mc()
    mc.simulate(64)
    assert "threads" in capsys.readouterr().err


def test_decisions_are_int8_for_binary_quantisers_only():
    """Binary v uses compact int8; multilevel v uses floating-point codes."""
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    af.input = Sinusoid(0.5, 1e5)
    assert af.simulate(64, verbose=False)["v"].dtype == np.int8
    af.levels = 4
    assert af.simulate(64, verbose=False)["v"].dtype == np.float64


def test_record_selection_keeps_the_decisions_with_noise(capsys):
    """Record selection cannot alter noisy v, u, or the default twin readout.

    Compare full and lean Numba records with a NumPy run; also check the
    reported excitation estimate.
    """

    def noisy() -> Frontend:
        af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
        af.input = Sinusoid(0.5, 1e5)
        af.state_covariance = 1e-12 * np.eye(3)  # excitation: the standard loop
        return af

    full = noisy().simulate(SIZE, record="all", verbose=False)
    lean = noisy().simulate(SIZE, verbose=False)
    ref = noisy().simulate(SIZE, engine="numpy", verbose=False)
    assert set(lean) == set(ref) == {"v", "u", "y_hat"}
    np.testing.assert_array_equal(full["v"], lean["v"])
    np.testing.assert_array_equal(full["u"], lean["u"])
    np.testing.assert_array_equal(full["v"], ref["v"])
    np.testing.assert_allclose(full["y_hat"], lean["y_hat"], atol=1e-12)
    noisy().simulate(64)
    assert "excitation peak" in capsys.readouterr().err


def test_monte_carlo_matches_the_column_runs_with_any_thread_count():
    """Monte Carlo decisions do not depend on thread count or record choice.

    J leaves a partial final block; compare that and other batch members
    with isolated runs for v, x, y, and u.
    """
    mc, J = _mc(), 2 * _W + 3
    batch = mc.simulate(SIZE, record="all", verbose=False)
    one = mc.simulate(SIZE, record="all", threads=1, verbose=False)
    lean = mc.simulate(SIZE, verbose=False)
    assert batch["v"].dtype == np.int8 and batch["v"].shape == (SIZE, 3, J)
    assert "x" in batch and "x" not in lean
    np.testing.assert_array_equal(batch["v"], one["v"])
    np.testing.assert_array_equal(batch["v"], lean["v"])
    np.testing.assert_array_equal(batch["y"], one["y"])
    assert batch["u"].shape == (SIZE, 1, J)
    for j in (0, _W, J - 1):
        res = mc[j].simulate(SIZE, record="all", verbose=False)
        np.testing.assert_array_equal(res["v"][:, :, 0], batch["v"][:, :, j])
        np.testing.assert_allclose(res["x"][:, :, 0], batch["x"][:, :, j], atol=1e-12)
        np.testing.assert_allclose(res["y"][:, :, 0], batch["y"][:, :, j], atol=1e-12)
        np.testing.assert_allclose(res["u"][:, :, 0], batch["u"][:, :, j], atol=1e-12)


def test_per_channel_and_per_state_attributes_normalise():
    """Per-channel offsets and scalar swing normalize to vectors.

    Reject wrong channel counts and the removed ``output_offset`` alias.
    """
    af, _ = Frontend.leapfrog(N=3, BW=1e6, OSR=16.0)
    af.comparator_input_offset, af.state_max = [0.01, 0.0, 0.0], 0.8
    assert (
        af.comparator_input_offset.shape == (3,)
        and af.comparator_input_offset.dtype == float
    )
    np.testing.assert_array_equal(af.state_max, [0.8, 0.8, 0.8])
    with pytest.raises(ValueError):
        af.comparator_input_offset = [0.0, 0.0]
    with pytest.raises(AttributeError, match="comparator_input_offset"):
        af.output_offset = [0.0, 0.0, 0.0]


def test_str_block_matrix():
    """String views show ABCD blocks, x0, and the selected J/K member."""
    integrator = fe.StateSpace(A=0.0, B=1.0, C=1.0, D=0.0)
    assert str(integrator) == "  A   B\n│ 0 │ 1 │\n├───┼───┤\n│ 1 │ 0 │\n  C   D"
    # a waveform adds x0 right of B; the batch column is noted
    tone = str(Sinusoid([0.1, 0.5], 1e5))
    assert tone.splitlines()[0].split() == ["A", "B", "x0"]
    assert tone.splitlines()[1].endswith("│        0.1 │")
    assert tone.endswith("[0, 0] of J=2, K=1")
    # a frontend: table when it fits, A to D one by one when it does not
    small, _ = Frontend.leapfrog(N=3, OSR=32, BW=1e6)
    large, _ = Frontend.leapfrog(N=8, OSR=32, BW=1e6)
    assert "├" in str(small) and "A (3, 3)" not in str(small)
    assert "A (8, 8):" in str(large) and "├" not in str(large)
    assert "[0, 0] of J=1, K=2" in str(small.with_trigger([2, 2, 1], [0, 1, 0]))
