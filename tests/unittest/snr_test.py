"""Check tone, residual, and FFT SNR against signals with known error power."""

from inspect import signature

import numpy as np
import pytest

from cbadc import snr
from cbadc.circuit import ActiveRC, GmC
from cbadc.frontend import Constant, Frontend, Samples, Sinusoid


def _tone(fs, f, n, noise_rms=0.0, seed=0):
    """Unit cosine plus reproducible white error at the requested RMS level."""
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    return np.cos(2 * np.pi * f * t) + noise_rms * rng.standard_normal(n)


def test_snr_tone_clean_tone_is_high():
    """An integer-cycle, noiseless tone has only numerical residual error."""
    # a coherent (integer-cycle) clean tone projects to ~machine precision
    fs, n = 1e3, 1 << 14
    f = 800 * fs / n  # integer number of cycles in the record
    y = _tone(fs, f, n, noise_rms=0.0)
    assert snr.snr_tone(y, f, fs) > 200  # essentially noiseless


def test_snr_tone_matches_known_noise_floor():
    """A unit cosine plus white noise yields ``10log10(0.5/noise_var)``."""
    # signal power 0.5 (unit amplitude cos), noise variance noise_rms**2
    # -> SNR ≈ 10log10(0.5 / noise_rms**2)
    noise_rms = 1e-3
    y = _tone(1e3, 50.0, 1 << 15, noise_rms=noise_rms)
    expected = 10 * np.log10(0.5 / noise_rms**2)
    assert snr.snr_tone(y, 50.0, 1e3) == pytest.approx(expected, abs=1.0)


def test_snr_tone_leakage_free_off_grid():
    """Fitting the exact tone avoids FFT leakage for a noncoherent frequency."""
    # a non-integer number of cycles still reads correctly (no window leakage)
    fs, n = 1e3, 1 << 14
    f = 50.0 * (1 + 1e-3)  # deliberately off the FFT grid
    y = _tone(fs, f, n, noise_rms=1e-4)
    assert snr.snr_tone(y, f, fs) > 70


def test_snr_tone_exact_on_short_noncoherent_window():
    """A short fractional-cycle record still recovers its injected noise floor.

    This exercises the exact sine/cosine projection where simple 2/N
    correlations leave appreciable tone residue.
    """
    fs, n = 32e6, 256
    f = 1.973e6  # ~15.8 cycles in the window -- deliberately non-coherent
    noise_rms = 1e-6
    y = _tone(fs, f, n, noise_rms=noise_rms, seed=3)
    expected = 10 * np.log10(0.5 / noise_rms**2)  # ~117 dB
    assert snr.snr_tone(y, f, fs) == pytest.approx(expected, abs=3.0)


def test_snr_tone_excludes_dc_like_fft_sndr():
    """DC offset is fitted separately, so it does not count as tone noise."""
    fs, n = 1e3, 1 << 14
    f = snr.coherent_frequency(50.0, fs, n)
    noise_rms = 1e-4
    y = _tone(fs, f, n, noise_rms=noise_rms, seed=7) + 0.1  # big offset
    expected = 10 * np.log10(0.5 / noise_rms**2)  # ~77 dB
    assert snr.snr_tone(y, f, fs) == pytest.approx(expected, abs=1.5)


def test_snr_residual_recovers_known_ratio():
    """Estimate/reference residual power gives the injected white-noise SNR."""
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(1 << 15)
    err_rms = 1e-2
    u_hat = ref + err_rms * rng.standard_normal(ref.size)
    expected = 10 * np.log10(np.var(ref) / err_rms**2)
    assert snr.snr_residual(u_hat, ref) == pytest.approx(expected, abs=0.5)


def test_snr_residual_default_returns_bare_float():
    """Default residual SNR returns a scalar unless diagnostics are requested."""
    # backward-compat: without the kwarg the return is a plain float, not a tuple
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(1 << 14)
    u_hat = ref + 1e-2 * rng.standard_normal(ref.size)
    out = snr.snr_residual(u_hat, ref)
    assert isinstance(out, float)
    assert not isinstance(out, tuple)


def test_snr_residual_diagnostics_decompose_gain_and_noise():
    """Residual diagnostics recover injected gain, correlation, and SNR identity.

    Construct ``u_hat = g*ref + noise`` with known g and noise power, then
    compare the fitted quantities with their analytic values.
    """
    # construct û = g * ref + noise with KNOWN gain and noise level, then check
    # the diagnostics recover g and rho and satisfy the exact identity
    #   1 / SNR_res = (g - 1) ** 2 + g ** 2 / SNR_rho   (linear power ratios).
    rng = np.random.default_rng(4)
    n = 1 << 17
    ref = rng.standard_normal(n)
    g_true = 1.2
    noise_rms = 5e-2
    noise = noise_rms * rng.standard_normal(n)
    u_hat = g_true * ref + noise

    snr_db, diag = snr.snr_residual(u_hat, ref, return_diagnostics=True)
    assert isinstance(diag, snr.ResidualDiagnostics)

    # best-fit gain g = <û, ref> / ||ref||**2 recovers the injected gain
    assert diag.g == pytest.approx(g_true, abs=2e-2)

    # rho is the correlation of g*ref + noise with ref; expected from the model:
    #   rho**2 = g**2 var(ref) / (g**2 var(ref) + var(noise))
    var_ref = np.var(ref)
    rho_expected = np.sqrt(g_true**2 * var_ref / (g_true**2 * var_ref + noise_rms**2))
    assert diag.rho == pytest.approx(rho_expected, abs=2e-3)

    # the headline identity, in LINEAR ratios
    snr_res_lin = 10 ** (snr_db / 10)
    snr_rho_lin = 10 ** (diag.snr_rho / 10)
    lhs = 1.0 / snr_res_lin
    rhs = (diag.g - 1.0) ** 2 + diag.g**2 / snr_rho_lin
    assert lhs == pytest.approx(rhs, rel=1e-6)


def test_snr_vs_frequency_flat_for_white_error():
    """White reference and error give a roughly flat SNR versus frequency."""
    rng = np.random.default_rng(2)
    fs = 1e3
    ref = rng.standard_normal(1 << 16)
    u_hat = ref + 1e-2 * rng.standard_normal(ref.size)
    f, snr_f = snr.snr_vs_frequency(u_hat, ref, fs)
    assert f[0] == 0.0 and f[-1] == pytest.approx(fs / 2)
    # white reference + white error -> roughly flat SNR(f)
    assert np.std(snr_f) < 3.0


def test_coherent_frequency_lands_on_fft_bin():
    """Coherent-frequency selection gives whole cycles near the target tone.

    Check its FFT-bin index and the known white-noise SNR on that bin.
    """
    fs, n = 1e3, 1 << 12
    f = snr.coherent_frequency(50.0, fs, n)
    cycles = f * n / fs
    assert cycles == pytest.approx(round(cycles))  # integer cycles
    assert abs(f - 50.0) < fs / n  # within one bin of the target
    # a coherent tone is a single FFT bin -> windowed and bin SNR now agree
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    y = np.cos(2 * np.pi * f * t) + 1e-3 * rng.standard_normal(n)
    assert snr.snr_tone(y, f, fs) == pytest.approx(
        10 * np.log10(0.5 / 1e-3**2), abs=1.5
    )


def test_snr_tone_is_trim_robust():
    """Discarding different edge lengths leaves long-record SNR within 1 dB."""
    # with a long enough record, the in-band SNR must not depend on trim
    fs, n = 1e3, 1 << 15
    f = snr.coherent_frequency(50.0, fs, n)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    y = np.cos(2 * np.pi * f * t) + 1e-3 * rng.standard_normal(n)
    vals = [snr.snr_tone(y, f, fs, trim=tr, band=fs / 4) for tr in (0, 500, 2000, 5000)]
    assert max(vals) - min(vals) < 1.0  # flat to within 1 dB


def test_snr_tone_handles_batch_members():
    """Tone SNR accepts ``(size, J, L)`` while preserving separate time series."""
    fs, n = 1e3, 1 << 15
    f = snr.coherent_frequency(50.0, fs, n)
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    u_hat = np.stack(
        [np.cos(2 * np.pi * f * t) + 1e-3 * rng.standard_normal(n) for _ in range(4)],
        axis=-1,
    )[:, :, None]  # (size, J=4, L=1)
    s = snr.snr_tone(u_hat, f, fs, band=fs / 4)
    # Pooling independent records preserves their expected tone/noise ratio.
    assert 50 < s < 65


def test_snr_residual_handles_batch_members():
    """Residual SNR uses all J members while preserving the known noise ratio."""
    rng = np.random.default_rng(2)
    ref = rng.standard_normal((1 << 14, 3, 1))
    u_hat = ref + 1e-2 * rng.standard_normal(ref.shape)
    expected = 10 * np.log10(np.var(ref) / 1e-2**2)
    assert snr.snr_residual(u_hat, ref) == pytest.approx(expected, abs=0.5)


def test_decimate_reduces_length():
    """Decimating by four reduces the sample axis by about four."""
    x = np.ones((1 << 12, 1, 1))
    y = snr.decimate(x, 4)
    assert y.shape[0] == pytest.approx(x.shape[0] // 4, abs=2)


def test_fft_sndr_recovers_harmonic_and_noise():
    """FFT metrics separate a known HD3 spur from the white noise floor.

    A -60 dBc third harmonic limits SNDR/SFDR while SNR excludes it.
    """
    # fundamental amp 1.0, HD3 at -60 dBc, white noise at a known floor.
    fs, N = 1e3, 1 << 14
    f = snr.coherent_frequency(50.0, fs, N)
    t = np.arange(N) / fs
    rng = np.random.default_rng(0)
    noise_rms = 3e-5
    y = (
        np.cos(2 * np.pi * f * t)
        + 1e-3 * np.cos(2 * np.pi * 3 * f * t + 0.4)  # HD3 = -60 dBc
        + noise_rms * rng.standard_normal(N)
    )
    r = snr.fft_sndr(y, f, fs)
    # SNDR is dominated by the -60 dBc HD3 spur
    assert r.sndr_dB == pytest.approx(60.0, abs=0.5)
    # SFDR is that same spur
    assert r.sfdr_dBc == pytest.approx(60.0, abs=0.5)
    # HD3 (index 1 = h=3 in HD2..) recovered to ~0.5 dB
    assert r.hd_dBc[1] == pytest.approx(-60.0, abs=0.5)
    # SNR excludes the harmonic -> only the white floor, well above SNDR
    assert r.snr_dB > 80.0


def test_fft_sndr_folds_aliased_harmonic():
    """HD3 above Nyquist folds into band at the correct -60 dBc level."""
    fs, n = 1e3, 1 << 14
    f = snr.coherent_frequency(200.0, fs, n)
    t = np.arange(n) / fs
    y = np.cos(2 * np.pi * f * t) + 1e-3 * np.cos(2 * np.pi * 3 * f * t)
    r = snr.fft_sndr(y, f, fs)
    assert r.hd_dBc[1] == pytest.approx(-60.0, abs=0.1)
    assert r.sndr_dB == pytest.approx(60.0, abs=0.1)
    assert r.snr_dB > r.sndr_dB


def test_fft_sndr_matches_snr_tone_on_clean_tone():
    """Without harmonics, FFT SNDR agrees with the fitted-tone SNR."""
    # with no harmonics, fft_sndr SNDR must agree with the snr_tone projection
    fs, N = 1e3, 1 << 14
    f = snr.coherent_frequency(50.0, fs, N)
    t = np.arange(N) / fs
    rng = np.random.default_rng(1)
    y = np.cos(2 * np.pi * f * t) + 1e-3 * rng.standard_normal(N)
    r = snr.fft_sndr(y, f, fs)
    assert r.sndr_dB == pytest.approx(snr.snr_tone(y, f, fs), abs=1.0)


def test_fft_sndr_channels_with_distinct_frequencies():
    """Separate L channels may carry distinct tones in one ``(size, J, L)`` array."""
    fs, N = 1e3, 1 << 14
    f0 = snr.coherent_frequency(40.0, fs, N)
    f1 = snr.coherent_frequency(123.0, fs, N)
    t = np.arange(N) / fs
    rng = np.random.default_rng(2)
    noise_rms = 3e-5
    Y = np.stack(
        [
            np.cos(2 * np.pi * f0 * t) + noise_rms * rng.standard_normal(N),
            0.5 * np.cos(2 * np.pi * f1 * t) + noise_rms * rng.standard_normal(N),
        ],
        axis=1,
    )[:, None, :]  # (size=N, J=1, L=2)
    r = snr.fft_sndr(Y, np.array([f0, f1]), fs)
    sndr = np.ravel(r.sndr_dB)
    assert sndr.size == 2
    # both clean tones -> high SNDR (noise-floor limited)
    assert np.all(sndr > 70.0)


def test_fft_sndr_in_band_excludes_out_of_band_spur():
    """A strong spur outside the requested band cannot lower in-band SNDR."""
    # a strong out-of-band spur must not affect an in-band SNDR
    fs, N = 1e3, 1 << 14
    f = snr.coherent_frequency(20.0, fs, N)
    f_spur = snr.coherent_frequency(300.0, fs, N)  # out of a 100 Hz band
    t = np.arange(N) / fs
    rng = np.random.default_rng(3)
    y = (
        np.cos(2 * np.pi * f * t)
        + 0.1 * np.cos(2 * np.pi * f_spur * t)  # large, but out of band
        + 1e-4 * rng.standard_normal(N)
    )
    r = snr.fft_sndr(y, f, fs, band=(0.0, 100.0))
    assert r.sndr_dB > 70.0  # the out-of-band spur is excluded


@pytest.mark.parametrize("input_index", [0, 1])
@pytest.mark.parametrize("readout", ["v", "u_hat", "y_hat"])
def test_simulate_snr_sweeps_every_batch_member_without_changing_frontend(
    input_index,
    readout,
):
    """Each Monte Carlo frontend gets the same dBFS sweep and its own curve.

    Compare the J=2 run with each selected frontend's J=1 run, including
    the shape and member-then-amplitude order of each SNDR curve.
    Both input gains resolve the tones instead of an idle-pattern FFT floor.
    """
    A = np.array([[[[-1.0]]], [[[-1.1]]]])
    C_hat = np.array([[[[-1.0]]], [[[-0.8]]]])
    af = Frontend(A, [[1.0, 1.5, -1.0]], [[1.0]], dt=0.1, C_hat=C_hat)
    af.input = (
        [af.input[0], Samples(np.tile([0.02, -0.03], (8, 1)))]
        if input_index == 0
        else [Constant([0.02, -0.03]), af.input[1]]
    )
    original_input = af.input[input_index]
    amp = np.array([-20.0, -12.0, -6.0])
    kwargs = dict(
        OSR=8,
        amp=amp,
        full_scale=1.0,
        k=8,
        warmup=32,
        engine="numpy",
        input_index=input_index,
        readouts=(readout,),
    )
    if readout == "u_hat":
        # Rebuilding af[j] must use the same nominal regularization as the batch.
        kwargs["eta2"] = af.wiener_filter(OSR=8).eta2

    result = snr.simulate_snr(af, **kwargs)
    name = f"snr_{readout}"
    curves, levels = result[name], result["amp"]
    assert curves.shape == (2, 3)
    assert np.isfinite(curves).all()
    np.testing.assert_array_equal(levels, amp)
    assert af.input[input_index] is original_input
    chunked = snr.simulate_snr(af, batch_size=4, **kwargs)
    np.testing.assert_allclose(chunked[name], curves)
    for j in range(af.J):
        one = snr.simulate_snr(af[j], **kwargs)[name]
        assert one.shape == (1, 3)
        np.testing.assert_allclose(curves[j], one[0])


@pytest.mark.parametrize("realization", ["GmC", "ActiveRC"])
@pytest.mark.parametrize("phases,readout", [(1, "u_hat"), (1, "y_hat"), (2, "y_hat")])
def test_simulate_snr_noisy_circuit_batch_sweep(
    realization, phases, readout, monkeypatch
):
    """Thermal noise follows each member through full and chunked sweeps.

    The two members have different physical capacitances and hence different
    state-noise intensities. A single-member sweep provides an exact reference
    for the chunked result and fixes the member-then-amplitude ordering.
    """
    A = np.array([[[[-1.0]], [[-1.2]]]])[:, :phases]
    af = Frontend(A, [[1.0, -2.0]], [[1.0]], dt=0.1)
    capacitance = np.array([[1e-16], [4e-16]])
    if realization == "GmC":
        circuit = GmC(af, Cint=capacitance, thermal_noise=True)
    else:
        circuit = ActiveRC(
            af,
            Cint=capacitance,
            gm=1e-9,
            Ro=np.inf,
            Co=1e-16,
            omega_u=1.0,
            thermal_noise=True,
        )
    assert circuit.J == 2
    covariance = circuit.state_covariance
    assert covariance.shape[:2] == (2, phases)
    assert covariance[0, 0, 0, 0] > covariance[1, 0, 0, 0]

    amp = np.array([-24.0, -16.0, -8.0])
    kwargs = dict(
        OSR=8,
        amp=amp,
        full_scale=0.5,
        k=8,
        warmup=32,
        engine="numpy",
        seed=719,
        readouts=(readout,),
    )

    def rebuild(af, size, x0, record, engine, threads, verbose, seed, **_):
        return af.simulate(size, x0, record, engine, threads, verbose, seed)

    for batch_size in (None, 2, 1):
        sweep = dict(kwargs, batch_size=batch_size)
        result = snr.simulate_snr(circuit, **sweep)
        repeated = snr.simulate_snr(circuit, **sweep)
        with monkeypatch.context() as patch:
            patch.setattr(snr, "simulate", rebuild)
            rebuilt = snr.simulate_snr(circuit, **sweep)
        name = f"snr_{readout}"
        assert result[name].shape == (2, len(amp))
        np.testing.assert_array_equal(result["amp"], amp)
        np.testing.assert_array_equal(result[name], repeated[name])
        np.testing.assert_array_equal(result[name], rebuilt[name])
        assert np.isfinite(result[name]).all()
        if batch_size == 1:
            for j in range(circuit.J):
                for a, level in enumerate(amp):
                    one = snr.simulate_snr(
                        circuit[j],
                        OSR=8,
                        amp=[level],
                        full_scale=0.5,
                        k=8,
                        warmup=32,
                        engine="numpy",
                        seed=719 + j * len(amp) + a,
                        readouts=(readout,),
                    )
                    np.testing.assert_array_equal(result[name][j, a], one[name][0, 0])


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("readout", ["v", "u_hat", "y_hat"])
def test_simulate_snr_matches_explicit_readout(discrete, readout):
    """The sweep equals simulation, optional Wiener filtering, then a Hann FFT.

    Wiener uses extra samples at both record ends. Twin readout is used raw;
    neither path applies a known-tone startup correction.
    """
    A = 0.995 if discrete else -0.005
    af = Frontend([[A]], [[0.2, -0.2]], [[1.0]], dt=1.0, discrete=discrete)
    size, warmup, OSR = 1 << 8, 32, 8
    tone = snr.coherent_frequency(af.fs / (8 * OSR), af.fs, size)
    amplitude = 10 ** (-12 / 20)
    af.input = Sinusoid(amplitude, tone, -np.pi / 2)
    length = size + 2 * warmup
    sim = af.simulate(length, record=("v", "y_hat"), engine="numpy", verbose=False)
    signal = (
        af.wiener_filter(OSR=OSR)(sim["v"], method="state-space")
        if readout == "u_hat"
        else sim[readout]
    )
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
    band = af.fs / (2 * OSR)
    kwargs = dict(
        OSR=OSR,
        amp=[-12],
        full_scale=1.0,
        f=tone,
        k=8,
        warmup=warmup,
        engine="numpy",
    )
    expected = snr.fft_sndr(
        signal[warmup : warmup + size, 0, 0] * window,
        tone,
        af.fs,
        band=band,
        extra_bins=1,
        n_dc=1,
    ).sndr_dB
    name = f"snr_{readout}"
    measured = snr.simulate_snr(af, readouts=(readout,), **kwargs)
    assert measured[name][0, 0] == pytest.approx(expected, abs=1e-7)
    default = snr.simulate_snr(af, **kwargs)
    assert set(default) == {"amp", "snr_v", "snr_y_hat", "snr_u_hat", "arguments"}
    np.testing.assert_array_equal(default[name], measured[name])


def test_simulate_snr_readouts_share_simulation_and_filter_setup(monkeypatch):
    """All three SNDR curves share one simulation per chunk and one filter setup.

    J=2 circuits and three amplitudes exercise the full and chunked paths;
    neither discretization nor Wiener synthesis is repeated per amplitude.
    """
    af = Frontend(np.array([[[[-1.0]]], [[[-1.1]]]]), [[1.0, -2.0]], [[1.0]], dt=0.1)
    simulate, wiener_filter = snr.simulate, Frontend.wiener_filter
    runs, setups = [], []

    def capture_simulation(*args, **kwargs):
        runs.append(args[1])
        return simulate(*args, **kwargs)

    def capture_filter(self, *args, **kwargs):
        setups.append(self.J_frontend)
        return wiener_filter(self, *args, **kwargs)

    monkeypatch.setattr(snr, "simulate", capture_simulation)
    monkeypatch.setattr(Frontend, "wiener_filter", capture_filter)
    for batch_size, count in ((None, 1), (2, 3)):
        runs.clear()
        setups.clear()
        result = snr.simulate_snr(
            af,
            8,
            amp=[-20, -12, -6],
            full_scale=1,
            k=8,
            warmup=32,
            batch_size=batch_size,
            engine="numpy",
        )
        assert set(result) == {"amp", "snr_v", "snr_y_hat", "snr_u_hat", "arguments"}
        for name in ("snr_v", "snr_y_hat", "snr_u_hat"):
            assert result[name].shape == (2, 3)
        assert runs == [256 + 2 * 32] * count
        assert setups == [2]


def test_simulate_snr_cyclic_models_need_raw_twin_readout():
    """Wiener rejects cyclic models; an explicit y_hat request remains valid."""
    A = np.array([[[0.99]], [[0.98]]])
    B = np.array([[[0.2, -0.2]], [[0.15, -0.2]]])
    af = Frontend(A, B, [[1.0]], dt=1.0, discrete=True)
    options = dict(OSR=8, amp=[-12], full_scale=1, k=8, warmup=32, engine="numpy")
    with pytest.raises(NotImplementedError, match="non-cyclic"):
        snr.simulate_snr(af, **options)
    result = snr.simulate_snr(af, readouts=("y_hat",), **options)
    assert set(result) == {"amp", "snr_y_hat", "arguments"}
    assert (
        result["snr_y_hat"].shape == (1, 1) and np.isfinite(result["snr_y_hat"]).all()
    )
    with pytest.raises(ValueError, match="readout"):
        snr.simulate_snr(af, readouts=("invalid",), **options)


@pytest.mark.parametrize("realization", ["GmC", "ActiveRC"])
@pytest.mark.parametrize("batch_size", [None, 2])
def test_simulate_snr_arguments_replay_noisy_sweep(realization, batch_size):
    """Saved arguments replay all curves, even after the caller changes inputs.

    Fresh noise seeds are recorded. The frontend and amplitude snapshots retain
    the original J=2 circuit parameters for both full and chunked sweeps.
    """
    af = Frontend([[-1.0]], [[1.0, -2.0]], [[1.0]], dt=0.1)
    capacitance = np.array([[1e-16], [4e-16]])
    circuit = (
        GmC(af, Cint=capacitance, thermal_noise=True)
        if realization == "GmC"
        else ActiveRC(
            af,
            Cint=capacitance,
            gm=1e-9,
            Ro=np.inf,
            Co=1e-16,
            omega_u=1.0,
            thermal_noise=True,
        )
    )
    amp = np.array([-20.0, -12.0, -6.0])
    result = snr.simulate_snr(
        circuit,
        8,
        amp=amp,
        full_scale=0.5,
        k=8,
        warmup=32,
        batch_size=batch_size,
        engine="numpy",
        seed=None,
    )
    arguments = result["arguments"]
    assert set(arguments) == set(signature(snr.simulate_snr).parameters)
    assert isinstance(arguments["seed"], int)
    assert arguments["eta2"] > 0
    assert arguments["af"] is not circuit
    assert arguments["f"] == snr.coherent_frequency(af.fs / 64, af.fs, 256)
    circuit.A = circuit.A * 0.9
    circuit.state_covariance *= 2
    amp[:] = -40
    replayed = snr.simulate_snr(**arguments)
    for name in ("amp", "snr_v", "snr_y_hat", "snr_u_hat"):
        np.testing.assert_array_equal(replayed[name], result[name])


def test_simulate_snr_selects_comparator_channel():
    """Raw v SNDR selects the requested comparator without combining stages."""
    af = Frontend(
        np.diag([0.99, 0.98]),
        [[0.2, -0.2, 0], [0.3, 0, -0.3]],
        np.eye(2),
        dt=1,
        discrete=True,
    )
    size, warmup = 256, 32
    tone = snr.coherent_frequency(af.fs / 64, af.fs, size)
    af.input = Sinusoid(10 ** (-12 / 20), tone, -np.pi / 2)
    v = af.simulate(size + 2 * warmup, record=("v",), engine="numpy")["v"]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
    expected = snr.fft_sndr(
        v[warmup : warmup + size, 1, 0] * window,
        tone,
        af.fs,
        band=af.fs / 16,
        extra_bins=1,
        n_dc=1,
    ).sndr_dB
    options = dict(
        OSR=8,
        amp=[-12],
        full_scale=1,
        k=8,
        warmup=warmup,
        readouts=("v",),
        engine="numpy",
    )
    result = snr.simulate_snr(af, readout_index=1, **options)
    assert result["snr_v"][0, 0] == pytest.approx(expected, abs=1e-7)
    with pytest.raises(IndexError, match="v decisions"):
        snr.simulate_snr(af, readout_index=2, **options)
