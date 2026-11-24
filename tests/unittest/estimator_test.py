"""Check Wiener, causal state-space, and digital-twin estimators."""

import numpy as np
import pytest
from numba import get_num_threads, set_num_threads

from cbadc import snr as snr_mod
from cbadc.circuit import ActiveRC
from cbadc.estimator import StateSpaceFilter, WienerFilter, _care, _dare, _message
from cbadc.frontend import DAC, Constant, Exp, Frontend, Poly, Sinusoid


def _ideal_leapfrog(*args, **kwargs):
    """Keep the Wiener estimator tests on their immediate-NRZ reference loop."""
    af, osr = Frontend.leapfrog(*args, **kwargs)
    af.dac = DAC(Constant(1.0))
    return af, osr


def _tone(af, amplitude, size, trim, f_target=None):
    """Choose a coherent input tone after trimming both record ends."""
    fs = 1.0 / af.dt
    f = snr_mod.coherent_frequency(f_target or 1e6 / 4, fs, size - 2 * trim)
    af.input = Sinusoid(amplitude, f)
    return f, fs


def test_care_builds_across_eta2():
    """Continuous Riccati solutions give finite Wiener gains across eta2."""
    af, _ = _ideal_leapfrog(N=4, ENOB=12, BW=1e6)
    for eta2 in (1e-2, 1e-6, 1e-9, 1e-10):
        wf = af.wiener_filter(eta2=float(eta2))
        assert wf.W.shape == (af.L, af.N) and np.all(np.isfinite(wf.W))


@pytest.mark.parametrize("order,design", [(7, {"OSR": 32}), (12, {"SNR": 150})])
@pytest.mark.parametrize("direction", [1, -1])
def test_stiff_care_has_small_residual_and_stable_filter(order, design, direction):
    """Stiff op-amp dynamics and large eta2 retain a stabilizing solution.

    Check both message directions against A.T V + V A - V V/eta2 + Q = 0,
    independently of simulation, and require every pole of A - V/eta2 to decay.
    """
    ideal, osr = _ideal_leapfrog(N=order, BW=1e6, **design)
    af = ActiveRC(ideal, Cint=2e-12, gm=1e-3, Ro=1e7, Co=1e-14, thermal_noise=False)
    _, H = af.transfer_function(
        np.array([1j * np.pi * af.fs / osr]), input_index=0, output_index=-1
    )
    eta2 = float(abs(H[0, 0, 0]) ** 2)
    A, B = direction * af.A[0, 0].T, af.B[0, 0, :, : af.L]
    Q = B @ B.T
    V = _care(A, Q, eta2)
    terms = (A.T @ V, V @ A, -V @ V / eta2, Q)
    assert np.linalg.norm(sum(terms)) / sum(np.linalg.norm(t) for t in terms) < 1e-8
    assert np.max(np.linalg.eigvals(A - V / eta2).real) < 0


def test_wiener_reaches_high_snr():
    """A bounded six-state loop reconstructs its tone above 100 dB SNR."""
    af, OSR = _ideal_leapfrog(N=6, ENOB=20, BW=1e6)
    size, trim = 1 << 18, 1 << 12
    f, fs = _tone(af, 0.2, size, trim)
    sim = af.simulate(size, record=("v", "x"))
    assert np.abs(sim["x"]).max() < 1.0
    u_hat = af.wiener_filter(OSR=int(OSR))(sim["v"])[:, 0, 0]
    assert snr_mod.snr_tone(u_hat, f, fs, trim=trim, band=1e6) > 100.0


def test_wiener_reaches_120db():
    """A long OSR=32 record supports over 120 dB reconstructed tone SNR."""
    af, _ = _ideal_leapfrog(OSR=32, N=6, BW=1e6)
    size, trim = 1 << 19, 1 << 13
    f, fs = _tone(af, 0.2, size, trim)
    sim = af.simulate(size)
    u_hat = af.wiener_filter(OSR=32)(sim["v"])[:, 0, 0]
    assert snr_mod.snr_tone(u_hat, f, fs, trim=trim, band=1e6) > 120.0


def _discrete_frontend(af):
    """Use the continuous frontend's exact A_d and B_d as a DT model."""
    dz = af.discretize()
    return Frontend(dz.A_d[0, 0], dz.B_d[0, 0], af.C[0, 0], dt=af.dt, discrete=True)


def test_discrete_time_wiener_builds_and_matches_continuous():
    """CT and equivalent DT Wiener filters give similar reconstructed SNR.

    Build the DT frontend from A_d/B_d, check finite filter matrices, then
    apply both filters to the same v record.
    """
    af, _ = _ideal_leapfrog(OSR=32, N=6, BW=1e6)
    afd = _discrete_frontend(af)
    assert afd.is_discrete_time
    wf_dt = afd.wiener_filter(eta2=1e-3)
    for attr in ("W", "A_f", "B_f", "A_b", "B_b"):
        assert np.all(np.isfinite(getattr(wf_dt, attr))), attr
    assert afd.wiener_filter(OSR=32).eta2 == pytest.approx(
        af.wiener_filter(OSR=32).eta2, rel=1e-2
    )
    size, trim = 1 << 14, 1 << 10
    f, fs = _tone(af, 0.2, size, trim)
    v = af.simulate(size)["v"]
    eta2 = af.wiener_filter(OSR=32).eta2
    u_ct = af.wiener_filter(eta2=eta2)(v)[:, 0, 0]
    u_dt = afd.wiener_filter(eta2=eta2)(v)[:, 0, 0]
    snr_ct = snr_mod.snr_tone(u_ct, f, fs, trim=trim, band=1e6)
    snr_dt = snr_mod.snr_tone(u_dt, f, fs, trim=trim, band=1e6)
    assert snr_dt == pytest.approx(snr_ct, abs=1.0)


def test_dare_solution_satisfies_the_riccati_equation():
    """The DT covariance solves its Riccati equation at three noise scales.

    Substitute X back into the matrix equation and bound its relative
    residual, including for very large eta2.
    """
    af, _ = _ideal_leapfrog(OSR=32, N=6, BW=1e6)
    afd = _discrete_frontend(af)
    A_d, B_u = afd.A[0, 0], afd.B[0, 0, :, : afd.L]
    Q, eye = B_u @ B_u.T, np.eye(afd.N)
    for eta2 in (1e2, 1e6, 1e10):
        X = _dare(A_d, Q, float(eta2))
        propagated = A_d @ X @ A_d.T
        measured = A_d @ X @ np.linalg.solve(X + eta2 * eye, X) @ A_d.T
        resid = propagated - measured + Q - X
        scale = max(np.linalg.norm(propagated), np.linalg.norm(Q), np.linalg.norm(X))
        assert np.linalg.norm(resid) / scale < 1e-6


@pytest.mark.parametrize(
    "dac",
    [
        DAC(Constant(1.0)),
        DAC(Constant(1.0), window=(0.25, 0.5)),
        DAC(Poly([1.0, -1.0]), window=(0.0, 1.0)),
        DAC(Exp(0.3), window=(0.0, 0.9)),
        DAC(Constant(1.0), window=(0.4, 0.6)),
    ],
)
def test_wiener_reconstructs_for_every_pulse_shape(dac):
    """Wiener reconstruction works for constant and shaped DAC pulses.

    Match each pulse's integrated feedback charge to immediate NRZ, then
    require good SNR from its clocked decision stream.
    """
    af, OSR = _ideal_leapfrog(N=3, BW=1e6, ENOB=12.0)
    af = af.with_dt(1.0)  # the pulse shapes above are in clock periods
    af.dac = dac
    charge = af.discretize().ABCD[
        0, 0, : af.N, af.N :
    ]  # scale B_M so every pulse delivers NRZ charge
    nrz = (
        Frontend(af.A, af.B, af.C, dt=1.0, dac=DAC(Constant(1.0)))
        .discretize()
        .ABCD[0, 0, : af.N, af.N :]
    )
    af.B[..., af.L :] *= np.trace(nrz) / np.trace(charge)
    size, trim = 1 << 13, 512
    f = snr_mod.coherent_frequency(1.0 / 64, 1.0, size)
    af.input = Sinusoid(0.5, f)
    v = af.simulate(size)["v"]
    wf = af.wiener_filter(OSR=OSR)
    snr = snr_mod.snr_tone(wf(v)[:, 0, 0], f, 1.0, trim=trim, band=0.5 / OSR)
    assert snr > 65.0


def test_taps_reject_degenerate_K():
    """FIR tap extraction needs at least two requested taps."""
    af, _ = _ideal_leapfrog(N=4, ENOB=12, BW=1e6)
    wf = af.wiener_filter(eta2=1e-3)
    for bad in (0, 1):
        with pytest.raises(ValueError):
            wf.taps(bad)


def test_taps_decay_away_from_the_centre():
    """Finite Wiener FIR taps carry far less energy at their truncated edges."""
    af, _ = _ideal_leapfrog(OSR=32, N=6, BW=1e6)
    wf = af.wiener_filter(OSR=32)
    K = 1 << 12
    h = wf.taps(K)
    energy = np.linalg.norm(h.reshape(K, -1), axis=1)
    assert energy[K // 2] > 1e3 * max(energy[0], energy[-1])


def test_fir_and_state_space_agree_to_machine_precision():
    """FIR and two-sided state-space Wiener forms agree after edge trimming."""
    af, _ = _ideal_leapfrog(OSR=32, N=6, BW=1e6)
    size, trim = 1 << 14, 1 << 11
    f, fs = _tone(af, 0.2, size, trim)
    v = af.simulate(size)["v"]
    wf = af.wiener_filter(OSR=32)
    exact = wf(v, method="state-space")[:, 0, 0]
    fir = wf(v)[:, 0, 0]
    rel = np.linalg.norm(exact[trim:-trim] - fir[trim:-trim]) / np.linalg.norm(
        exact[trim:-trim]
    )
    assert rel < 1e-12


def test_truncated_taps_converge_to_the_recursion():
    """Longer FIR support approaches the full state-space recursion."""
    af, _ = _ideal_leapfrog(OSR=32, N=6, BW=1e6)
    size, trim = 1 << 13, 1 << 11
    f, fs = _tone(af, 0.2, size, trim)
    v = af.simulate(size)["v"]
    wf = af.wiener_filter(OSR=32)
    exact = wf(v, method="state-space")[trim:-trim, 0, 0]
    errors = []
    for K in (64, 256, 1024):
        wf._taps = None
        h = wf.taps(K)
        wf._taps = h
        errors.append(
            np.linalg.norm(wf(v)[trim:-trim, 0, 0] - exact) / np.linalg.norm(exact)
        )
    assert errors[0] > errors[1] > errors[2]


def test_unknown_method_and_eta2_cache():
    """Unknown methods fail; changing eta2 invalidates taps and rejects negatives."""
    af, _ = _ideal_leapfrog(N=4, ENOB=12, BW=1e6)
    wf = af.wiener_filter(eta2=1e-3)
    with pytest.raises(ValueError):
        wf(np.zeros((8, af.M, 1)), method="nope")
    v = np.zeros((64, af.M, 1))
    wf(v)
    assert wf._taps is not None
    wf.eta2 = 1e-4
    assert wf._taps is None
    with pytest.raises(ValueError):
        wf.eta2 = -1.0


def test_transfer_functions_have_the_right_shapes():
    """Wiener G, signal, and noise transfer maps have the expected axes.

    Also check near-unit low-frequency signal gain for a chain of integrators.
    """
    af, _ = Frontend.chain_of_integrators(ENOB=12.0, N=5, BW=1e7)
    af.dac = DAC(Constant(1.0))
    wf = af.wiener_filter(OSR=16)
    jw = np.logspace(6, 8, 50) * 2j * np.pi
    assert wf.G(jw).shape == (50, af.N, af.L + af.M)
    assert wf.stf(jw)[1].shape == (50, af.L, af.L)
    assert wf.ntf(jw)[1].shape == (50, af.L, af.N)
    assert np.abs(wf.stf(jw)[1][0, 0, 0]) == pytest.approx(
        1.0, abs=0.05
    )  # unity in band


# ------------------------------------------------------------ causal half --


@pytest.mark.parametrize(
    "J,K,shared",
    [
        (1, 1, True),
        (2, 2, True),
        (3, 2, True),
        (16, 1, False),
        (19, 3, False),
        (19, 3, True),
    ],
)
@pytest.mark.parametrize("forward", [True, False])
def test_message_matches_matrix_recursion(J, K, shared, forward):
    """Both kernels reproduce A m + B v, then add W m to existing output.

    Exercise shared and per-member models, phase broadcasting, a partial
    worker batch, and strided decisions in both recursion directions, with
    one worker and the configured worker count.
    """
    rng = np.random.default_rng(7)
    size, N, M, L = 32, 3, 2, 2
    J_m = 1 if shared else J
    A = rng.normal(scale=0.1, size=(J_m, 1, N, N))
    B = rng.normal(size=(J_m, K, N, M))
    W = rng.normal(size=(1, K, L, N))
    v = rng.integers(-1, 2, size=(size, M, J * 2), dtype=np.int8)[:, :, ::2]
    out = rng.normal(size=(size, L, J))
    expected = out.copy()
    for j in range(J):
        m = np.zeros(N)
        for i in range(1, size) if forward else range(size - 1, -1, -1):
            k = i % K
            m = A[j % J_m, 0] @ m + B[j % J_m, k] @ v[i - 1 if forward else i, :, j]
            expected[i, :, j] += W[0, k] @ m
    original_threads = get_num_threads()
    try:
        for threads in (1, original_threads):
            set_num_threads(threads)
            actual = out.copy()
            _message(A, B, W, v, actual, forward)
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    finally:
        set_num_threads(original_threads)


def test_state_space_filter_replays_the_wiener_forward_message():
    """The causal filter reproduces the Wiener forward state recurrence.

    Drive both with identical random decisions and check every state plus
    the causal filter's stable pole radii.
    """
    af, _ = _ideal_leapfrog(OSR=32, N=4, BW=1e6)
    wf = WienerFilter(af, 1e6)
    ssf = StateSpaceFilter.from_model(af, eta2=1e6)
    v = np.where(np.random.default_rng(0).uniform(size=(256, af.M, 1)) < 0.5, -1.0, 1.0)
    m = np.zeros((257, af.N, 1))
    for i in range(256):
        m[i + 1] = wf.A_f @ m[i] + wf.B_f @ v[i]
    np.testing.assert_allclose(
        ssf.states(v), m[:256], atol=1e-12
    )  # kernel sums in another order
    assert np.all(ssf.pole_radii() < 1.0)


def test_state_space_filter_estimates_the_input_causally():
    """Causal output tracks the input and supports decimated output.

    Measure correlation with u after startup, then check DSR=4 length.
    """
    af, OSR = _ideal_leapfrog(N=3, BW=1e6, ENOB=12.0)
    size = 1 << 13
    f, fs = _tone(af, 0.5, size, 512, f_target=fs_target(af, OSR))
    res = af.simulate(size, record=("v", "u"))
    ssf = StateSpaceFilter.from_model(af, OSR=OSR)
    u_hat = ssf(res["v"])
    assert u_hat.shape == (size, af.L, 1)
    rho = np.corrcoef(u_hat[1024:, 0, 0], res["u"][1024:, 0, 0])[0, 1]
    assert rho > 0.9  # causal, so a lag, but clearly the signal
    half = StateSpaceFilter.from_model(af, OSR=OSR, DSR=4)(res["v"])
    assert half.shape[0] == size // 4


def fs_target(af, OSR):
    """Choose an in-band tone away from DC for causal estimation."""
    return af.fs / (2 * OSR) / 8


# -- the digital twin and the Monte Carlo estimator ----------------------------


def _monte_carlo(J: int = 6, damping: float = 0.0, size: int = 1 << 13):
    """Create J perturbed A matrices with shared B, C, DAC, and tone."""
    af, OSR = _ideal_leapfrog(N=3, BW=1e6, SNR=100.0)
    A = af.A[0, 0] - damping * 2 * np.pi * 1e6 * np.eye(3)
    rng = np.random.default_rng(11)
    spread = 1.0 + 0.02 * rng.standard_normal((J, 3, 3))
    f = round(size * 2.5e5 / af.fs) * af.fs / size
    mc = Frontend(
        A * spread[:, None],
        af.B[0, 0],
        af.C[0, 0],
        dt=af.dt,
        input=Sinusoid(0.5, f),
        dac=DAC(Constant(1.0)),
    )
    return mc, OSR, f, size


def test_digital_twin_is_the_state_minus_the_open_loop_response():
    """Twin state equals full x minus the input-only response for each J.

    Disconnect DAC columns in a second frontend to obtain x_u, then check
    ``x_hat=x-x_u`` and the default readout ``y_hat=-x_hat[-1]``.
    """
    mc, _, _, size = _monte_carlo(J=3)
    sim = mc.simulate(size, record=("x", "x_hat", "y_hat"), verbose=False)
    open_loop = Frontend(
        mc.A, mc.B * np.r_[1.0, np.zeros(mc.M)], mc.C, dt=mc.dt, input=mc.input
    )
    x_u = open_loop.simulate(size, record="x", verbose=False)[
        "x"
    ]  # the DACs disconnected
    np.testing.assert_allclose(
        sim["x_hat"], sim["x"] - x_u, atol=1e-9 * np.abs(x_u).max()
    )
    np.testing.assert_allclose(sim["y_hat"], -sim["x_hat"][:, -1:])  # C_hat = -e_N


def test_damped_twin_matches_the_wiener_snr_in_band():
    """A damped twin and Wiener filter give similar in-band tone SNR.

    Compare ``y_hat``, an estimate of the input-only last state, with the
    separately reconstructed input u after their transients settle.
    """
    mc, OSR, f, size = _monte_carlo(
        J=1, damping=0.1, size=1 << 14
    )  # the twin settles with the frontend damping
    af = mc[0]
    sim = af.simulate(size, verbose=False)
    g_u = sim["y_hat"][:, 0, 0]  # the last state of the signal alone, error x_3
    u_hat = af.wiener_filter(OSR=OSR)(sim["v"])[:, 0, 0]
    snr_twin = snr_mod.snr_tone(g_u, f, af.fs, trim=4096, band=1e6)
    snr_wiener = snr_mod.snr_tone(u_hat, f, af.fs, trim=4096, band=1e6)
    assert snr_wiener > 90 and abs(snr_twin - snr_wiener) < 3.0


def test_wiener_filter_over_monte_carlo_frontends_matches_the_column_filters():
    """A J-batch Wiener filter equals one filter per frontend.

    Compare state-space reconstructions for selected columns, check causal
    decimation shape, and reject unsupported batch FIR taps.
    """
    mc, OSR, f, size = _monte_carlo(J=6)
    sim = mc.simulate(size, verbose=False)
    wf = mc.wiener_filter(OSR=OSR)
    assert wf.J_m == 6 and wf.W.shape == (6, 1, 1, 3)
    u_all = wf(sim["v"], method="state-space")
    for j in (0, 3, 5):
        one = WienerFilter(mc[j], wf.eta2)  # the batch shares the nominal eta2
        u_j = one(sim["v"][:, :, j : j + 1], method="state-space")
        np.testing.assert_allclose(u_all[:, :, j : j + 1], u_j, atol=1e-12)
        np.testing.assert_allclose(one(sim["v"][:, :, j : j + 1]), u_j, atol=1e-10)
    causal = StateSpaceFilter.from_model(mc, OSR=OSR, DSR=2)(sim["v"])
    assert causal.shape == (size // 2, 1, 6)
    with pytest.raises(NotImplementedError, match="state-space"):
        wf(sim["v"])
    with pytest.raises(NotImplementedError, match="state-space"):
        wf.taps()


def test_cyclic_twin_replays_a_chopped_converter():
    """The twin identity holds when B and C alternate sign over K=2.

    Compare x_hat with x minus an input-only chopped frontend, then check
    that the non-cyclic Wiener estimator rejects this model.
    """
    af, _ = _ideal_leapfrog(N=3, BW=1e6, OSR=16.0)
    size = 1 << 12
    f = round(size * 1e5 / af.fs) * af.fs / size
    B = np.stack(
        [af.B[0, 0], af.B[0, 0] * np.r_[1.0, -np.ones(af.M)]]
    )  # DACs flip at odd phases
    C = np.stack([af.C[0, 0], -af.C[0, 0]])  # and so do the comparators
    chopped = Frontend(af.A[0, 0], B, C, dt=af.dt, input=Sinusoid(0.5, f))
    chopped.dac = DAC(Constant(1.0))
    assert chopped.K == 2
    sim = chopped.simulate(size, record=("x", "x_hat"), verbose=False)
    open_loop = Frontend(
        af.A[0, 0], B * np.r_[1.0, np.zeros(af.M)], C, dt=af.dt, input=chopped.input
    )
    x_u = open_loop.simulate(size, record="x", verbose=False)["x"]
    np.testing.assert_allclose(
        sim["x_hat"], sim["x"] - x_u, atol=1e-9 * np.abs(x_u).max()
    )
    with pytest.raises(NotImplementedError, match="non-cyclic"):
        chopped.wiener_filter(OSR=16.0)


def test_a_cyclic_C_hat_reads_out_the_baseband_envelope():
    """Cyclic C_hat demodulates a twin tone into baseband I and Q.

    Set K=64 readout rows at carrier phase, decimate, then check in-band
    tone SNR and the positive-frequency direction of ``I+jQ``.
    """
    af0, _ = _ideal_leapfrog(N=3, BW=1e6, OSR=16.0)
    size, DSR = 1 << 14, 16
    f_c = af0.fs / 64  # in band, divides the clock
    df = round(size * 5e4 / af0.fs) * af0.fs / size
    A = af0.A[0, 0] - 0.1 * 2 * np.pi * 1e6 * np.eye(
        3
    )  # damped: the twin's ringing dies
    af = Frontend(A, af0.B[0, 0], af0.C[0, 0], dt=af0.dt, input=Sinusoid(0.5, f_c + df))
    phi = 2 * np.pi * f_c * np.arange(64) * af.dt  # the carrier phase at step k
    last = np.eye(3)[2]  # the last state, largest gain
    af.C_hat = np.stack(
        [-2 * np.cos(phi)[:, None] * last, 2 * np.sin(phi)[:, None] * last], axis=1
    )  # (64, 2, 3): I = 2 cos (-x_hat), Q = -2 sin (-x_hat)
    assert af.K == 64
    sim = af.simulate(size, verbose=False)
    iq = snr_mod.decimate(sim["y_hat"], DSR)
    assert iq.shape == (size // DSR, 2, 1)
    I, Q = iq[:, 0, 0], iq[:, 1, 0]
    snr_I = snr_mod.snr_tone(I, df, af.fs / DSR, trim=64, band=af.fs / DSR / 2)
    assert snr_I > 50
    # the envelope rotates one way: I + jQ has its tone at +df only
    E = np.fft.fft((I + 1j * Q)[64:-64])
    k = round(df / (af.fs / DSR) * E.size)
    assert abs(E[k]) > 100 * abs(E[-k])
