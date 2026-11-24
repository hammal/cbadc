"""Exercise tone sweeps across converter architectures, orders, and bandwidths.

Orders 2--7 and OSR 4, 8, 16, 32 cover binary/multibit DT and CT SDMs,
MASH, leapfrog, integrator chains, and RC chains. Each run measures three
amplitudes at two frequencies with the same Hann FFT for v, y_hat, and u_hat.
State-based y_hat is not applicable when A_d has poles on/outside the unit
circle. A direct decision readout remains valid. JUnit properties retain the
curves and this applicability flag for inspection.
Order means requested filter order: Active-RC adds op-amp states and CRFFD
adds a quantizer-delay state.

For single-comparator SDMs, compare v with cbadc-deltasigma.simulateSNR too.
Its short, soft-started transient differs from cbadc's longer warmup, so compare
SNDR statistically rather than requiring identical nonlinear trajectories.
"""

import importlib
import json

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.signal import cont2discrete, lfilter, zpk2tf

from cbadc import delsig, snr
from cbadc.circuit import ActiveRC, GmC
from cbadc.frontend import Frontend, Samples, Sinusoid

BW = 1e6
AMPLITUDES = (-24.0, -12.0, -6.0)
K_FFT, WARMUP = 14, 4096
SDM_FORMS = ("CRFB", "CRFF", "CIFB", "CIFF", "CRFBD", "CRFFD", "Stratos")
ARCHITECTURES = (
    [(f"dt-{form}", levels) for form in SDM_FORMS for levels in (2, 5)]
    + [(f"ct-{form}", levels) for form in ("FB", "FF") for levels in (2, 5)]
    + [("mash", levels) for levels in (2, 5)]
    + [
        (name, 2)
        for name in (
            "leapfrog",
            "integrators",
            "leaky-chain",
            "rc-chain",
            "gmC",
            "activeRC",
        )
    ]
)


def make_frontend(architecture, order, osr, levels=2):
    """Construct a test circuit; RC chains include explicit, fixed leakage.

    The passive RC ladder uses equal R and C: A has adjacent +1/RC
    couplings and negative row sums; B injects u at its first node and one
    feedback DAC per node. No circuit adds thermal noise in this matrix.
    """
    if architecture.startswith(("dt-", "ct-")):
        time, form = architecture.split("-")
        # CIFB/CIFF implement zeros at DC, not the optimized resonator zeros.
        ntf = delsig.synthesizeNTF(
            order, osr, opt=0 if form in ("CIFB", "CIFF") else 1, H_inf=1.5
        )
        if time == "ct":
            return Frontend.from_delsig_ntf(
                ntf, BW=BW, OSR=osr, form=form, nlev=levels
            )[0]
        coefficients = delsig.realizeNTF(ntf, form)
        return Frontend.dtsdm(delsig.stuffABCD(*coefficients, form), levels=levels)
    if architecture == "mash":
        # Limit each stage to first/second order; N is the total filter order.
        orders = [2] * (order // 2) + [1] * (order % 2)
        if order == 2:
            orders = [1, 1]
        ntfs = [delsig.synthesizeNTF(n, osr, opt=1) for n in orders]
        return Frontend.mash(ntfs, BW=BW, OSR=osr, nlev=levels)[0]
    if architecture in ("leapfrog", "gmC", "activeRC"):
        ideal, _ = Frontend.leapfrog(order, BW=BW, OSR=osr)
        if architecture == "gmC":
            return GmC(ideal, Cint=2e-12, Ro=np.inf, thermal_noise=False)
        if architecture == "activeRC":
            return ActiveRC(
                ideal, Cint=2e-12, gm=1e-3, Ro=1e7, Co=1e-14, thermal_noise=False
            )
        return ideal
    if architecture in ("integrators", "leaky-chain"):
        rho = -2 * np.pi * BW if architecture == "leaky-chain" else 0
        return Frontend.chain_of_integrators(order, BW=BW, OSR=osr, rho=rho)[0]
    if architecture == "rc-chain":
        rate = 2 * np.pi * BW
        A = rate * (
            np.diag([-2.0] * (order - 1) + [-1.0])
            + np.diag(np.ones(order - 1), 1)
            + np.diag(np.ones(order - 1), -1)
        )
        B = rate * np.column_stack((np.eye(order)[:, 0], -np.eye(order)))
        return Frontend(
            A, B, np.eye(order), dt=1 / (2 * osr * BW), C_hat=-np.eye(order)[-1:]
        )
    raise ValueError(architecture)


def _toolbox_snr(af, osr, levels, frequency, k):
    """Return the native toolbox amplitude curve for an independent ZOH ABCD."""
    A, B, C, D = (matrix[0, 0] for matrix in af.abcd)
    if not af.discrete:
        A, B, C, D, _ = cont2discrete((A, B, C, D), af.dt)
    reference, amps = delsig.simulateSNR(
        np.block([[A, B], [C, D]]),
        osr,
        amp=AMPLITUDES,
        nlev=levels,
        f=frequency / af.fs,
        k=k,
    )
    np.testing.assert_array_equal(amps, AMPLITUDES)
    return reference


def _y_hat_applicable(af):
    """Require strictly stable state dynamics unless y_hat reads decisions only.

    For these K=1 models, A_d=A in DT and A_d=expm(A*dt) in CT. Use a
    1e-9 margin to identify numerical unit-circle poles conservatively.
    """
    if af.C_hat is not None and not np.any(af.C_hat):
        return True
    poles = np.linalg.eigvals(af.A[:, 0])
    if not af.discrete:
        poles = np.exp(poles * af.dt)
    return bool(np.all(np.abs(poles) < 1 - 1e-9))


@pytest.mark.parametrize(
    "architecture,applicable",
    [
        ("dt-CRFB", True),
        ("dt-CRFF", True),
        ("dt-CRFFD", True),
        ("ct-FB", True),
        ("mash", False),
        ("leapfrog", False),
        ("integrators", False),
        ("gmC", False),
        ("leaky-chain", True),
        ("rc-chain", True),
        ("activeRC", True),
    ],
)
def test_y_hat_applicability(architecture, applicable):
    """Marginal state twins are excluded; stable and direct readouts are valid."""
    assert _y_hat_applicable(make_frontend(architecture, 4, 16)) is applicable


@pytest.mark.parametrize("order", range(2, 8))
@pytest.mark.parametrize("osr", (4, 8, 16, 32))
@pytest.mark.parametrize("architecture,levels", ARCHITECTURES)
def test_simulate_snr_architecture_matrix(
    architecture, levels, order, osr, request, monkeypatch
):
    """All readouts produce ordered curves; compatible SDMs match DELSIG.

    The continuous-time reference holds sine samples for one clock period,
    whereas cbadc integrates a continuous sine. Its independent ZOH ABCD
    checks loop synthesis and SNDR, allowing for that input-waveform difference.
    The first tone is BW/4, the second 3BW/4. No NTF target is imposed on
    passive RC filters or on noise-free circuit realizations with finite gain.
    """
    af = make_frontend(architecture, order, osr, levels)
    # CRFFD adds a zero-eigenvalue delay state. The backward Wiener map
    # currently needs A^-1; keep testing its decision/twin path explicitly.
    readouts = ("v", "y_hat") if architecture == "dt-CRFFD" else ("v", "y_hat", "u_hat")
    y_hat_applicable = _y_hat_applicable(af)
    if not y_hat_applicable:
        readouts = tuple(name for name in readouts if name != "y_hat")
    simulate = snr.simulate
    unresolved = []

    def capture(*args, **kwargs):
        result = simulate(*args, **kwargs)
        v = result["v"][WARMUP : WARMUP + (1 << K_FFT)]
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1 << K_FFT) / (1 << K_FFT))
        spectrum = np.abs(np.fft.rfft(v * window[:, None, None], axis=0)) ** 2
        bin_index = round(f / af.fs * (1 << K_FFT))
        tone_power = spectrum[bin_index - 1 : bin_index + 2].sum(axis=0)
        unresolved.append(tone_power <= 1e-24 * spectrum.sum(axis=0))
        return result

    monkeypatch.setattr(snr, "simulate", capture)
    rows = []
    for fraction in (0.25, 0.75):
        f = snr.coherent_frequency(fraction * af.fs / (2 * osr), af.fs, 1 << K_FFT)
        result = snr.simulate_snr(
            af,
            osr,
            amp=AMPLITUDES,
            full_scale=1,
            f=f,
            k=K_FFT,
            warmup=WARMUP,
            seed=7,
            readouts=readouts,
        )
        curves = {name: result[f"snr_{name}"][0] for name in readouts}
        rows.append(
            dict(
                fraction=fraction,
                y_hat_applicable=y_hat_applicable,
                **{name: curve.tolist() for name, curve in curves.items()},
            )
        )
        np.testing.assert_array_equal(result["amp"], AMPLITUDES)
        if af.M == 1 and fraction == 0.25:
            reference = _toolbox_snr(af, osr, levels, f, K_FFT)
            rows[-1]["deltasigma"] = reference.tolist()
    request.node.user_properties.append(("curves", json.dumps(rows)))
    for i, row in enumerate(rows):
        for name in readouts:
            curve = np.array(row[name])
            assert curve.shape == (len(AMPLITUDES),)
            assert not np.isinf(curve).any()
            # Dead zones and idle patterns can have zero tone power. Allow
            # undefined SNDR only when the actual decision FFT confirms it.
            no_tone = unresolved[i][0] if name == "v" else unresolved[i].all(axis=0)
            assert np.all(np.isfinite(curve) | no_tone)
        if af.M == 1:
            # y_hat is v delayed one clock: only Hann endpoint weighting differs.
            np.testing.assert_allclose(
                row["y_hat"], row["v"], atol=0.5, rtol=0, equal_nan=True
            )
        if "deltasigma" in row:
            reference = np.array(row["deltasigma"])
            # Below 20 dB the tone can lock to an idle pattern. Require
            # comparable SNDR where the reference actually resolves it.
            # The reference reports inf even for an all-zero record; cbadc
            # deliberately reports NaN when the fundamental has zero power.
            resolved = np.isfinite(reference) & (reference > 20)
            if resolved.any():
                measured_peak = np.max(np.array(row["v"])[resolved])
                reference_peak = np.max(reference[resolved])
                if abs(measured_peak - reference_peak) > 3:
                    # Resolve finite-record/idle-pattern sensitivity by using
                    # four times as many samples; keep the same 3 dB gate.
                    k = K_FFT + 2
                    frequency = snr.coherent_frequency(af.fs / (8 * osr), af.fs, 1 << k)
                    longer = snr.simulate_snr(
                        af,
                        osr,
                        amp=AMPLITUDES,
                        full_scale=1,
                        f=frequency,
                        k=k,
                        warmup=WARMUP,
                        readouts=("v", "y_hat"),
                        seed=7,
                    )["snr_v"][0]
                    reference = _toolbox_snr(af, osr, levels, frequency, k)
                    measured_peak, reference_peak = (
                        np.nanmax(longer),
                        np.nanmax(reference),
                    )
                    request.node.user_properties.append(
                        (
                            "longer_reference",
                            json.dumps(
                                dict(
                                    k=k,
                                    v=longer.tolist(),
                                    deltasigma=reference.tolist(),
                                )
                            ),
                        )
                    )
                assert abs(measured_peak - reference_peak) <= 3


@pytest.mark.xfail(
    strict=True,
    raises=np.linalg.LinAlgError,
    reason="CRFFD's zero-eigenvalue delay state cannot be inverted by the backward Wiener map",
)
def test_crffd_default_wiener_support():
    """Track the default-readout limitation separately from valid raw sweeps."""
    af = make_frontend("dt-CRFFD", 3, 16)
    snr.simulate_snr(af, 16, amp=[-12], full_scale=1, k=10)


@pytest.mark.parametrize("form", ("CRFB", "CRFF"))
@pytest.mark.parametrize("levels", (2, 5))
def test_toolbox_window_and_soft_start_reference(form, levels, monkeypatch):
    """Check FFT conventions and simulation against the exact toolbox stimulus.

    Observe simulateSNR's own inputs and decisions without changing them.
    Its band omits the upper edge bin; use that same band for an exact FFT
    comparison. Replay its soft-started samples in cbadc to check normalized
    codes and sample alignment independently of the differing sweep warmups.
    """
    af = make_frontend(f"dt-{form}", 3, 16, levels)
    ABCD = np.block([[af.A[0, 0], af.B[0, 0]], [af.C[0, 0], af.D[0, 0]]])
    module = importlib.import_module("cbadc_deltasigma._simulateSNR")
    simulate_dsm = module.simulateDSM
    records = []

    def capture(u, *args):
        result = simulate_dsm(u, *args)
        records.append((u.copy(), result[0].reshape(-1).copy()))
        return result

    monkeypatch.setattr(module, "simulateDSM", capture)
    size = 1 << 12
    frequency = snr.coherent_frequency(af.fs / 128, af.fs, size)
    reference, _ = delsig.simulateSNR(
        ABCD, 16, amp=[-12, -6], nlev=levels, f=frequency / af.fs, k=12
    )
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
    for expected, (u, v) in zip(reference, records):
        measured = snr.fft_sndr(
            v[100:] * window,
            frequency,
            af.fs,
            band=np.nextafter(af.fs / 32, 0),
            extra_bins=1,
            n_dc=1,
        ).sndr_dB
        assert measured == pytest.approx(expected, abs=1e-9)
        af.input = Samples(u / (levels - 1))
        replayed = af.simulate(len(u), record=("v",), engine="numpy", verbose=False)[
            "v"
        ][:, 0, 0]
        np.testing.assert_allclose(replayed[:128], v[:128] / (levels - 1), atol=1e-12)
        replayed_sndr = snr.fft_sndr(
            replayed[100:] * window,
            frequency,
            af.fs,
            band=np.nextafter(af.fs / 32, 0),
            extra_bins=1,
            n_dc=1,
        ).sndr_dB
        assert replayed_sndr == pytest.approx(expected, abs=3)


def _sndr(signal, af, osr, frequency):
    """Measure a (size + 2*warmup, ...) record with the sweep's Hann FFT."""
    size = 1 << K_FFT
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / size)
    return snr.fft_sndr(
        signal[WARMUP : WARMUP + size]
        * window.reshape((-1,) + (1,) * (signal.ndim - 1)),
        frequency,
        af.fs,
        band=af.fs / (2 * osr),
        extra_bins=1,
        n_dc=1,
    ).sndr_dB


@pytest.mark.parametrize("architecture", ("leapfrog", "integrators", "leaky-chain"))
def test_raw_twin_contains_zero_state_input_transient(architecture, request):
    """Explain y_hat using the linear input response and bounded physical x.

    For R=-C_hat, y_hat = R*x_input - R*x. A sine from rest has
    x_input(t) = Im[p exp(j*w*t)] - exp(A*t) Im[p],
    p = (j*w*I-A)^-1 B_L * amplitude. The second term persists for the
    lossless models; it decays for a leaky chain. Longer warmup alone cannot
    remove a nondecaying homogeneous solution. No correction is applied.
    """
    af = make_frontend(architecture, 4, 32)
    frequency = snr.coherent_frequency(af.fs / 256, af.fs, 1 << K_FFT)
    amplitude = 0.5
    af.input = Sinusoid(amplitude, frequency, -np.pi / 2)
    sim = af.simulate(2 * WARMUP + 1, record=("x", "y_hat"), verbose=False)
    A, B, R = af.A[0, 0], af.B[0, 0, :, 0], -af.C_hat[0, 0, 0]
    p = np.linalg.solve(2j * np.pi * frequency * np.eye(af.N) - A, B * amplitude)
    indices = np.array([WARMUP // 2, WARMUP, 2 * WARMUP])
    transient = np.array([R @ expm(A * i * af.dt) @ p.imag for i in indices])
    periodic = np.imag(R @ p * np.exp(2j * np.pi * frequency * indices * af.dt))
    observed = sim["y_hat"][indices, 0, 0] + sim["x"][indices, :, 0] @ R
    np.testing.assert_allclose(observed, periodic - transient, rtol=1e-8, atol=1e-6)
    ratio = np.max(np.abs(transient[1:])) / abs(R @ p)
    request.node.user_properties.append(("startup_to_tone_ratio", str(ratio)))
    assert np.max(np.abs(sim["x"][WARMUP:])) < 2
    if architecture == "leaky-chain":
        assert ratio < 1e-8
    else:
        assert ratio > 0.01


@pytest.mark.parametrize("order", (2, 4, 7))
@pytest.mark.parametrize("osr", (16, 32))
@pytest.mark.parametrize("levels", (2, 5))
def test_mash_cancellation_uses_actual_stage_stf(order, osr, levels, request):
    """Combine all MASH decisions with their actual signal/noise transfers.

    Stage m gives v_m = S_m u_m + H_m e_m, with u_m = -e_(m-1).
    Therefore sum_m [prod_(j<m) H_j prod_(j>m) S_j] v_m cancels all but the
    last error. These stages omit direct input feedthrough, so S_m=1-H_m;
    assuming S_m=1 leaves earlier-stage noise, especially for many stages.
    Evaluate first/second-order sections separately to keep them well scaled.
    """
    af = make_frontend("mash", order, osr, levels)
    frequency = snr.coherent_frequency(af.fs / (8 * osr), af.fs, 1 << K_FFT)
    af.input = Sinusoid(10 ** (np.array([-12, -6]) / 20), frequency, -np.pi / 2)
    sim = af.simulate(
        (1 << K_FFT) + 2 * WARMUP, record=("v", "u", "y", "y_hat"), verbose=False
    )
    v = sim["v"]
    orders = [2] * (order // 2) + [1] * (order % 2) if order != 2 else [1, 1]
    sections = []
    for n in orders:
        ntf = delsig.synthesizeNTF(n, osr, opt=1)
        sections.append(zpk2tf(ntf.zeros, ntf.poles, ntf.gain))
    cancelled = np.zeros_like(v[:, 0], dtype=float)
    unit_stf = np.zeros_like(cancelled)
    for m in range(af.M):
        term = v[:, m]
        for b, a in sections[:m]:
            term = lfilter(b, a, term, axis=0)
        unit_stf += term
        for b, a in sections[m + 1 :]:
            term = lfilter(a - b, a, term, axis=0)
        cancelled += term
    # Independent oracle: only the last quantization error may survive.
    # The cancellation algorithm above itself never uses u or comparator y.
    signal = sim["u"][:, 0]
    error = v[:, -1] - sim["y"][:, -1]
    for b, a in sections:
        signal = lfilter(a - b, a, signal, axis=0)
        error = lfilter(b, a, error, axis=0)
    predicted = signal + error
    np.testing.assert_allclose(cancelled, predicted, rtol=1e-9, atol=1e-9)
    u_hat = af.wiener_filter(OSR=osr)(v, method="state-space")[:, 0]
    values = {
        name: np.atleast_1d(_sndr(signal, af, osr, frequency)).tolist()
        for name, signal in (
            ("v_first", v[:, 0]),
            ("cancelled", cancelled),
            ("unit_stf_assumption", unit_stf),
            ("u_hat", u_hat),
            ("y_hat", sim["y_hat"][:, 0]),
        )
    }
    request.node.user_properties.append(("cancellation", json.dumps(values)))
    assert max(values["cancelled"]) > max(values["v_first"]) + 3
    np.testing.assert_allclose(
        values["cancelled"], _sndr(predicted, af, osr, frequency), atol=0.01, rtol=0
    )
    if order == 7 and osr == 32:
        assert max(values["cancelled"]) > max(values["unit_stf_assumption"]) + 20
