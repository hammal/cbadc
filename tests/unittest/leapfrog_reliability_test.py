"""End-to-end leapfrog resolution and physical-noise checks.

The design SNR formula describes the paper's immediate-NRZ loop. Check it
and symmetric, equal-area pulses starting up to T/4 later across tones and
record lengths, then check actual states.
"""

import numpy as np
import pytest

from cbadc.circuit import ActiveRC, GmC
from cbadc.frontend import DAC, Constant, Frontend, Sinusoid
from cbadc.snr import simulate_snr

BW = 1e6
AMP_DB = (-18.0, -12.0, -6.0, -3.0, 0.0)
LONG_RUN = (1 << 18) + 8192


def paper_leapfrog(order, target, edge=0.0):
    """Keep pulse area fixed while moving both edges inward by ``edge*T``."""
    af, osr = Frontend.leapfrog(N=order, SNR=target, BW=BW)
    width = 1.0 - 2.0 * edge
    af.dac = DAC(Constant(1.0 / width), window=None if edge == 0 else (edge, width))
    return af, osr


@pytest.mark.parametrize("edge", (0.0, 0.125, 0.25))
@pytest.mark.parametrize("order", range(2, 13))
@pytest.mark.parametrize("target", (60, 90, 120, 150))
def test_leapfrog_resolution_matrix(order, target, edge):
    """Equal-area pulses approach the target SNR for feasible OSR <= 32.

    Sweep amplitude at two in-band frequencies. The coherent Hann FFT uses
    65,536 samples after warmup; SNDR includes harmonics and is therefore a
    conservative check of the SNR design target.
    """
    af, osr = paper_leapfrog(order, target, edge)
    if osr > 32:
        pytest.skip(f"order {order} needs OSR {osr:.1f} for {target} dB")
    for frequency in (BW / 8, BW / 2):
        result = simulate_snr(
            af,
            osr,
            amp=AMP_DB,
            full_scale=1.0,
            f=frequency,
            k=16,
            warmup=8192,
            seed=7,
        )
        curve, levels = result["snr_u_hat"], result["amp"]
        assert curve.shape == (1, len(AMP_DB))
        np.testing.assert_array_equal(levels, AMP_DB)
        assert np.isfinite(curve).all()
        assert curve.max() >= target - 6, (
            f"N={order}, target={target} dB, edge={edge}, OSR={osr:.1f}, "
            f"f={frequency:g} Hz: peak SNDR={curve.max():.1f} dB"
        )


@pytest.mark.parametrize("edge", (0.0, 0.125, 0.25))
@pytest.mark.parametrize("order", (7, 12))
def test_high_resolution_survives_long_records_and_bounded_states(order, edge):
    """Near-150 dB persists over 262,144 steps with bounded states."""
    af, osr = paper_leapfrog(order, 150, edge)
    measured = []
    for k in (14, 18):
        result = simulate_snr(
            af,
            osr,
            amp=[0.0],
            full_scale=1.0,
            f=BW / 4,
            k=k,
            warmup=8192,
            seed=7,
        )
        measured.append(result["snr_u_hat"][0, 0])
    assert min(measured) >= 144
    assert abs(measured[1] - measured[0]) < 2

    af.input = Sinusoid(1.0, BW / 4, -np.pi / 2)
    states = af.simulate(LONG_RUN, record="x", verbose=False)["x"]
    states = states[8192:]
    assert np.isfinite(states).all()
    assert np.max(np.abs(states)) < 1.1
    early = np.sqrt(np.mean(states[:8192] ** 2))
    late = np.sqrt(np.mean(states[-8192:] ** 2))
    assert abs(late - early) < 0.1 * early


@pytest.mark.parametrize("edge", (0.0, 0.125, 0.25))
@pytest.mark.parametrize("order, target", ((7, 120), (7, 150), (12, 150)))
def test_quiet_circuit_realizations_preserve_high_resolution(order, target, edge):
    """Physical Gm-C and finite-bandwidth Active-RC retain the ideal result.

    Use 262,144 samples at 150 dB to resolve sub-dB circuit differences.
    """
    ideal, osr = paper_leapfrog(order, target, edge)
    circuits = (
        GmC(ideal, Cint=2e-12, Ro=np.inf, thermal_noise=False),
        ActiveRC(
            ideal,
            Cint=2e-12,
            gm=1e-3,
            Ro=1e7,
            Co=1e-14,
            thermal_noise=False,
        ),
    )

    def peak(model):
        result = simulate_snr(
            model,
            osr,
            amp=(-12.0, -6.0, -3.0, 0.0),
            full_scale=1.0,
            f=BW / 4,
            k=18 if target == 150 else 16,
            warmup=8192,
            seed=7,
        )
        return float(result["snr_u_hat"].max())

    reference = peak(ideal)
    assert reference >= target - 6
    for circuit in circuits:
        assert abs(peak(circuit) - reference) < 1
        if order == 12:
            circuit.input = Sinusoid(1.0, BW / 4, -np.pi / 2)
            result = circuit.simulate(LONG_RUN, record="x", verbose=False)
            states = result["x"]
            assert np.isfinite(states).all()
            assert np.max(np.abs(states[8192:])) < 3


@pytest.mark.parametrize("edge", (0.0, 0.125, 0.25))
@pytest.mark.parametrize("order, target", ((7, 120), (7, 150), (12, 150)))
def test_high_resolution_numpy_and_numba_agree(order, target, edge):
    """The reference and compiled recursions agree at 120--150 dB."""
    af, osr = paper_leapfrog(order, target, edge)
    options = dict(
        amp=(-6.0, -3.0, 0.0),
        full_scale=1.0,
        f=BW / 4,
        k=14,
        warmup=8192,
        seed=7,
    )
    compiled = simulate_snr(af, osr, engine="numba", **options)["snr_u_hat"]
    reference = simulate_snr(af, osr, engine="numpy", **options)["snr_u_hat"]
    np.testing.assert_allclose(compiled, reference, atol=0.01)


@pytest.mark.parametrize("edge", (0.0, 0.125, 0.25))
@pytest.mark.parametrize("order, target", ((7, 120), (12, 150)))
@pytest.mark.parametrize("realization", ("gmC", "activeRC"))
def test_high_order_noisy_circuits_are_stationary(order, target, realization, edge):
    """Long noisy runs retain their SNDR and bounded states across seeds."""
    ideal, osr = paper_leapfrog(order, target, edge)
    if realization == "gmC":
        model = GmC(ideal, Cint=2e-11, Ro=1e6, thermal_noise=True)
    else:
        model = ActiveRC(
            ideal, Cint=2e-11, gm=1e-3, Ro=1e6, Co=1e-13, thermal_noise=True
        )

    measured = np.array(
        [
            simulate_snr(
                model,
                osr,
                amp=[0.0],
                full_scale=1.0,
                f=BW / 4,
                k=18,
                warmup=8192,
                seed=seed,
            )["snr_u_hat"][0, 0]
            for seed in (7, 17, 27)
        ]
    )
    assert measured.min() > 80
    assert np.ptp(measured) < 2

    model.input = Sinusoid(1.0, BW / 4, -np.pi / 2)
    result = model.simulate(LONG_RUN, record="x", seed=7, verbose=False)
    states = result["x"][8192:]
    assert np.isfinite(states).all()
    assert np.max(np.abs(states)) < 3
    early = np.sqrt(np.mean(states[:8192] ** 2))
    late = np.sqrt(np.mean(states[-8192:] ** 2))
    assert abs(late - early) < 0.1 * early


@pytest.mark.parametrize("edge", (0.0, 0.125, 0.25))
@pytest.mark.parametrize("realization", ("gmC", "activeRC"))
def test_noisy_circuit_resolution_matrix(realization, edge):
    """Damped circuit noise has stable SNDR and follows T and C scaling.

    Gm-C keeps Ro*C fixed across capacitor values, so its deterministic
    leakage stays fixed. Active-RC keeps its op-amp Ro fixed for the same
    reason. The physical noise floor need not meet the ideal SNR target.
    """
    ideal, osr = paper_leapfrog(4, 90, edge)

    def circuit(C, temperature=300, thermal_noise=True):
        if realization == "gmC":
            return GmC(
                ideal,
                Cint=C,
                Ro=2e-5 / C,
                temperature=temperature,
                thermal_noise=thermal_noise,
            )
        return ActiveRC(
            ideal,
            Cint=C,
            gm=1e-3,
            Ro=1e6,
            Co=1e-13,
            temperature=temperature,
            thermal_noise=thermal_noise,
        )

    def sndr(model, k=18):
        return np.array(
            [
                simulate_snr(
                    model,
                    osr,
                    amp=[-3.0],
                    full_scale=1.0,
                    f=BW / 4,
                    k=k,
                    warmup=8192,
                    seed=seed,
                )["snr_u_hat"][0, 0]
                for seed in (7, 17, 27)
            ]
        )

    # Keep thermal noise above the quiet quantization floor, so temperature
    # and 1/C scaling remain measurable. Capacitance must not change A or B.
    capacitance = 1e-13
    small, large = circuit(capacitance), circuit(10 * capacitance)
    np.testing.assert_allclose(small.A, large.A, rtol=1e-14, atol=0)
    np.testing.assert_allclose(small.B, large.B, rtol=1e-14, atol=0)
    quiet = sndr(circuit(capacitance, thermal_noise=False))
    baseline = sndr(small)
    hot = sndr(circuit(capacitance, temperature=600))
    larger = sndr(large)
    short = sndr(small, k=14)
    assert np.isfinite(np.r_[quiet, baseline, hot, larger, short]).all()
    assert quiet.mean() - baseline.mean() > 10
    assert 1.5 < baseline.mean() - hot.mean() < 4.5
    assert 7 < larger.mean() - baseline.mean() < 12
    assert abs(short.mean() - baseline.mean()) < 5
    assert np.ptp(baseline) < 4

    model = small
    model.input = Sinusoid(10 ** (-3 / 20), BW / 4, -np.pi / 2)
    result = model.simulate(LONG_RUN, record="x", seed=7, verbose=False)
    states = result["x"][8192:]
    assert np.isfinite(states).all()
    assert np.max(np.abs(states)) < 2.5
    early = np.sqrt(np.mean(states[:8192] ** 2))
    late = np.sqrt(np.mean(states[-8192:] ** 2))
    assert abs(late - early) < 0.1 * early
