"""Check DELSIG realizations as Frontend matrices, DAC windows, and codes."""

import numpy as np
import pytest
from scipy.signal import TransferFunction, lfilter, zpk2tf

from cbadc import Frontend, delsig, snr
from cbadc.frontend import DAC, Constant, Sinusoid

N, OSR, BW = 3, 48, 1e6


def _sdm(nlev, h_inf):
    """Synthesize an NTF and turn it into a continuous-time frontend."""
    ntf = delsig.synthesizeNTF(N, OSR, 1, h_inf)
    return Frontend.from_delsig_ntf(ntf, BW=BW, OSR=OSR, nlev=nlev)[0]


def test_ctsdm_levels_take_effect():
    """The same CT ABCD with 2 or 5 levels must produce that many codes."""
    ntf = delsig.synthesizeNTF(N, OSR, 1, 1.5)
    ABCDc, tdac2 = delsig.realizeNTF_ct(ntf, form="FB", tdac=(0, 1))
    dt = 1.0 / (2 * BW * OSR)
    af2 = Frontend.ctsdm(ABCDc, tdac2, dt=dt, levels=2)
    af5 = Frontend.ctsdm(ABCDc, tdac2, dt=dt, levels=5)
    assert af2.levels.tolist() == [2] and af5.levels.tolist() == [5]
    assert af2.M == 1 and af2.dt == pytest.approx(dt)
    assert af5.dac[0].window is None  # NRZ from tdac (0, 1)
    np.testing.assert_array_equal(af2.C_hat[0, 0], np.zeros((1, af2.N)))
    np.testing.assert_array_equal(af2.D_hat[0, 0], [[1.0]])
    f = af2.fs / 300
    af2.input = af5.input = Sinusoid(0.5, f)
    assert np.unique(af2.simulate(1024)["v"]).size == 2
    assert np.unique(af5.simulate(1024)["v"]).size == 5


def test_ctsdm_rz_timing_becomes_a_window():
    """Map DELSIG's RZ pulse edges ``tdac`` to DAC delay and width."""
    ntf = delsig.synthesizeNTF(N, OSR, 1, 1.5)
    ABCDc, tdac2 = delsig.realizeNTF_ct(ntf, form="FB", tdac=(0.25, 0.75))
    af = Frontend.ctsdm(ABCDc, tdac2, dt=1e-7)
    assert af.dac[0].window == pytest.approx((0.25, 0.5))


def test_from_delsig_ntf_matches_leapfrog_signature_and_stf():
    """NTF synthesis must preserve order and clock, with unit DC signal gain.

    Compare N and dt with a leapfrog, then evaluate the closed-loop input
    transfer function near DC.
    """
    ntf = delsig.synthesizeNTF(N, OSR, 1, 1.5)
    afd, osr_d = Frontend.from_delsig_ntf(ntf, BW=BW, OSR=OSR)
    afl, _ = Frontend.leapfrog(N=N, OSR=OSR, BW=BW)
    assert osr_d == pytest.approx(OSR) and afd.N == afl.N == N
    assert afd.dt == pytest.approx(afl.dt)
    jw = np.array([1j * 2 * np.pi * 1e-3 * BW])
    _, h = afd.transfer_function(jw, input_index=0, output_index=-1, open_loop=False)
    assert abs(h[0, 0, 0]) == pytest.approx(1.0, rel=1e-3)


def test_multibit_quantizer_exercises_all_levels():
    """An 8-level converter must use bipolar odd codes across a driven run."""
    af = _sdm(8, 2.0)
    af.input = Sinusoid(0.8, 0.11 * BW, -np.pi / 2)
    codes = np.unique(np.round(af.simulate(1 << 12)["v"][:, 0, 0] * 7))
    assert codes.min() <= -3 and codes.max() >= 3
    assert set(codes.astype(int)) <= {-7, -5, -3, -1, 1, 3, 5, 7}


def test_is_stable_dsm_flags_overloading_single_bit():
    """The NTF probe distinguishes a stable design from an overloaded one."""
    assert (
        delsig.is_stable_dsm(delsig.synthesizeNTF(3, 48, opt=1, H_inf=1.5), nlev=2)
        is True
    )
    assert (
        delsig.is_stable_dsm(delsig.synthesizeNTF(5, 17, opt=1, H_inf=2.0), nlev=2)
        is False
    )


def test_max_stable_amplitude_orders_the_architectures():
    """Compare the overload threshold of leapfrog, chain, and DELSIG loops.

    Use immediate NRZ for the two synthesized frontends, then require the
    expected order of their maximum stable input amplitudes.
    """
    af_lf, _ = Frontend.leapfrog(N=N, OSR=OSR, BW=BW)
    af_ci, _ = Frontend.chain_of_integrators(N=N, OSR=OSR, BW=BW)
    af_lf.dac = af_ci.dac = DAC(Constant(1.0))  # compare ideal-NRZ designs
    msa_lf = delsig.max_stable_amplitude(af_lf, DSR=OSR)
    msa_ci = delsig.max_stable_amplitude(af_ci, DSR=OSR)
    assert msa_lf > 1.0 and msa_ci >= 1.0 and msa_lf > msa_ci
    msa_1b = delsig.max_stable_amplitude(_sdm(2, 1.5), DSR=OSR)
    msa_nb = delsig.max_stable_amplitude(_sdm(8, 2.0), DSR=OSR)
    assert 0.0 < msa_1b < 1.0 and 0.0 < msa_nb < 1.0 and msa_nb > msa_1b


def test_max_stable_amplitude_detects_clear_overload():
    """A high-order single-bit NTF must overload below 0.9 full scale."""
    af, _ = Frontend.from_delsig_ntf(
        delsig.synthesizeNTF(5, 17, opt=1, H_inf=2.0), BW=BW, OSR=17, nlev=2
    )
    assert delsig.max_stable_amplitude(af, DSR=17) < 0.9


def test_mash_structure():
    """A two-stage MASH keeps block states separate and feeds stage 1 residue forward.

    Check the A, B, C sparsity pattern, final-stage twin readout, immediate
    DAC windows, and errors for unsupported or empty stage lists.
    """
    osr = 25
    ntfs = [delsig.synthesizeNTF(2, osr, opt=1) for _ in range(2)]
    af, got = Frontend.mash(ntfs, BW=1e6, OSR=osr, nlev=2)
    assert all(dac.window is None for dac in af.dac)
    assert got == pytest.approx(osr) and af.is_discrete_time
    assert af.N == 4 and af.M == 2 and af.L == 1
    A, B, C = af.A[0, 0], af.B[0, 0], af.C[0, 0]
    s0, s1 = slice(0, 2), slice(2, 4)
    assert np.count_nonzero(C[0, s1]) == 0 and np.count_nonzero(C[1, s0]) == 0
    assert np.count_nonzero(C[0, s0]) > 0 and np.count_nonzero(C[1, s1]) > 0
    assert np.count_nonzero(A[s1, s0]) > 0  # inter-stage residue coupling
    assert np.count_nonzero(B[s1, 0]) == 0 and np.count_nonzero(B[s1, 1]) > 0
    np.testing.assert_array_equal(af.C_hat[0, 0], -C[-1:])
    np.testing.assert_array_equal(af.D_hat[0, 0], np.zeros((1, af.M)))
    assert np.count_nonzero(C[-1] @ A @ B[:, 1]) > 0
    assert np.count_nonzero(C[-1] @ B[:, 2]) > 0
    with pytest.raises(NotImplementedError):
        Frontend.mash([TransferFunction([1.0, 0.0], [1.0, 1.0])], BW=1e6, OSR=osr)
    with pytest.raises(ValueError):
        Frontend.mash([], BW=1e6, OSR=osr)


def test_mash_wiener_matches_digital_cancellation_sndr():
    """Filtering v_2 by stage 1's NTF cancels its error in v_1.

    Wiener reconstruction should retain that in-band SNDR; raw twin mode
    must match a direct FFT of the simulated y_hat record.
    """
    OSR, BW, gain = 25, 1e6, 1.25
    ntf = delsig.synthesizeNTF(2, OSR, opt=1)
    af, _ = Frontend.mash([ntf, ntf], BW=BW, OSR=OSR, inter_stage_gain=gain)
    size, warmup = 1 << 10, 128
    f = snr.coherent_frequency(BW / 4, af.fs, size)
    af.input = Sinusoid(10 ** (-12 / 20), f, -np.pi / 2)
    sim = af.simulate(
        warmup + size, record=("v", "y_hat"), engine="numpy", verbose=False
    )
    v = sim["v"][:, :, 0]
    b, a = zpk2tf(ntf.zeros, ntf.poles, ntf.gain)
    cancelled = v[:, 0] + lfilter(b, a, v[:, 1]) / gain

    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
    band = af.fs / (2 * OSR)

    def measured(signal):
        return snr.fft_sndr(
            signal[warmup:] * window, f, af.fs, band=band, extra_bins=1, n_dc=1
        ).sndr_dB

    cancelled_sndr = measured(cancelled)
    assert cancelled_sndr > measured(v[:, 0]) + 10
    raw_twin_sndr = measured(sim["y_hat"][:, 0, 0])
    swept = snr.simulate_snr(
        af, OSR, amp=[-12], full_scale=1, f=f, k=10, warmup=warmup, engine="numpy"
    )
    raw_swept = snr.simulate_snr(
        af,
        OSR,
        amp=[-12],
        full_scale=1,
        f=f,
        k=10,
        warmup=warmup,
        readouts=("y_hat",),
        engine="numpy",
    )
    assert raw_twin_sndr == pytest.approx(raw_swept["snr_y_hat"][0, 0], abs=1e-6)
    assert swept["snr_u_hat"][0, 0] >= cancelled_sndr - 3


@pytest.mark.parametrize(
    "time, form, levels",
    [("dt", "CRFB", 2), ("dt", "CRFF", 2), ("ct", "FB", 2), ("ct", "FF", 5)],
)
@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_single_comparator_sdm_readout_is_delayed_decision(time, form, levels, engine):
    """DT and CT feedback/feedforward SDMs return the preceding normalized code."""
    OSR, BW = 32, 1e6
    ntf = delsig.synthesizeNTF(3, OSR, opt=1, H_inf=1.5)
    if time == "dt":
        a, g, b, c = delsig.realizeNTF(ntf, form)
        af = Frontend.dtsdm(delsig.stuffABCD(a, g, b, c, form), levels=levels)
    else:
        af, _ = Frontend.from_delsig_ntf(ntf, BW=BW, OSR=OSR, form=form, nlev=levels)
    np.testing.assert_array_equal(af.C_hat[0, 0], np.zeros((1, af.N)))
    np.testing.assert_array_equal(af.D_hat[0, 0], [[1.0]])
    af.input = Sinusoid(0.3, af.fs / 200)
    sim = af.simulate(1024, record=("v", "y_hat"), engine=engine, verbose=False)
    np.testing.assert_array_equal(sim["y_hat"][0], 0.0)
    np.testing.assert_array_equal(sim["y_hat"][1:], sim["v"][:-1])


def test_dtsdm_three_levels_simulates_on_the_full_scale_convention():
    """The discrete-time ABCD uses codes -1, 0, 1 with mean near a DC input."""
    ntf = delsig.synthesizeNTF(3, 32, 1, 1.5)
    a, g, b, c = delsig.realizeNTF(ntf, "CRFB")
    af = Frontend.dtsdm(delsig.stuffABCD(a, g, b, c, "CRFB"), levels=3)
    assert af.dac[0].window is None
    np.testing.assert_array_equal(af.C_hat[0, 0], np.zeros((1, af.N)))
    np.testing.assert_array_equal(af.D_hat[0, 0], [[1.0]])
    af.input = Constant(0.3)
    v = af.simulate(512)["v"]
    assert set(np.unique(v)) <= {-1.0, 0.0, 1.0}
    assert v[64:].mean() == pytest.approx(0.3, abs=0.05)


def test_feedforward_sdm_default_readout_ignores_the_state_sum():
    """A CRFF decision readout is independent of its weighted comparator mapping."""
    ntf = delsig.synthesizeNTF(3, 32, opt=1)
    a, g, b, c = delsig.realizeNTF(ntf, "CRFF")
    af = Frontend.dtsdm(delsig.stuffABCD(a, g, b, c, "CRFF"))
    assert np.count_nonzero(af.C[0, 0, 0, :-1]) > 0
    af.input = Constant(0.1)
    sim = af.simulate(64, record=("v", "y_hat"), engine="numpy", verbose=False)
    np.testing.assert_array_equal(sim["y_hat"][1:], sim["v"][:-1])


def test_from_delsig_ntf_normalises_rz_like_nrz():
    """RZ and NRZ NTF frontends must both pass a DC input at unit gain.

    Run each DAC window with ``u=0.3``. Check mean decisions and Wiener
    estimate against u, and keep all states within the swing budget.
    """
    ntf = delsig.synthesizeNTF(N, OSR, 1, 1.5)
    for tdac in ((0, 1), (0.25, 0.75)):
        af, _ = Frontend.from_delsig_ntf(ntf, BW=BW, OSR=OSR, tdac=tdac)
        af.input = Constant(0.3)
        sim = af.simulate(1 << 13, record=("v", "x"))
        assert sim["v"][2048:].mean() / 0.3 == pytest.approx(1.0, abs=0.01)
        u_hat = af.wiener_filter(OSR=OSR)(sim["v"])[2048:-512, 0, 0]
        assert u_hat.mean() / 0.3 == pytest.approx(1.0, abs=0.01)
        assert np.max(np.abs(sim["x"])) < 5.0
