"""Excess-loop-delay sweep for a damped N=7 leapfrog and its digital twin."""

import numpy as np

from cbadc import snr
from cbadc.frontend import DAC, Constant, Frontend, Sinusoid


def test_leapfrog_eld_amplitude_sweep():
    """Measure how DAC delay changes bounded swing and digital-twin SNR.

    For N=7, OSR=10, sweep tone frequency, amplitude, and DAC delay.
    Read ``max|x|`` for the frontend and in-band SNR from ``y_hat``;
    assert the expected performance loss and overload at longer delays.
    """
    bw, size, trim = 1e6, 1 << 14, 1 << 11
    af, osr = Frontend.leapfrog(N=7, OSR=10, BW=bw)
    assert osr == 10
    # The synthesized open-loop poles are undamped. Fixed, small leakage lets
    # the digital twin's natural response settle before SNR is measured.
    af.A = af.A - 0.1 * 2 * np.pi * bw * np.eye(af.N)
    delays = (0.0, 0.25, 0.5, 0.75, 1.0)
    amplitudes = (0.1, 0.25, 0.5, 0.75, 0.9)
    results = {}
    for target in (bw / 8, bw / 4, bw / 2):
        frequency = snr.coherent_frequency(target, af.fs, size - 2 * trim)
        for delay in delays:
            af.dac = DAC(Constant(1.0), window=None if delay == 0 else (delay, 1.0))
            for amplitude in amplitudes:
                af.input = Sinusoid(amplitude, frequency)
                sim = af.simulate(size, record=("x", "y_hat"), verbose=False)
                peak = float(np.max(np.abs(sim["x"])))
                twin_snr = float(
                    snr.snr_tone(
                        sim["y_hat"][:, 0, 0],
                        frequency,
                        af.fs,
                        trim=trim,
                        band=bw,
                    )
                )
                results[target, delay, amplitude] = (peak, twin_snr)

        # A quarter-period delay is benign, while half a period costs about
        # 5–6 dB. The larger delays cross the 5x-full-scale state-swing limit.
        s0 = results[target, 0.0, 0.5][1]
        assert abs(results[target, 0.25, 0.5][1] - s0) < 3.0
        assert s0 - results[target, 0.5, 0.5][1] > 3.0
        assert results[target, 0.5, 0.9][0] < 5.0
        assert results[target, 0.75, 0.5][0] > 5.0
        assert results[target, 1.0, 0.1][0] > 5.0

    for target in (bw / 8, bw / 4, bw / 2):
        for delay in delays:
            stable = [a for a in amplitudes if results[target, delay, a][0] < 5.0]
            print(
                f"f/BW={target / bw:.3f}, ELD={delay:.2f}: "
                f"largest tested bounded amplitude={max(stable, default=0):.2f}, "
                f"twin SNR at A=0.5={results[target, delay, 0.5][1]:.1f} dB"
            )
