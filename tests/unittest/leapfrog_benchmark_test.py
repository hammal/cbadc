"""Benchmark cbadc's SNR against Feyling et al. (2023), Fig. 3.

F. Feyling, H. Malmberg, C. Wulff, H.-A. Loeliger, T. Ytterdal, "Design and
Analysis of the Leapfrog Control-Bounded A/D Converter," IEEE TVLSI, 2023,
doi: 10.1109/TVLSI.2023.3320279.

Fig. 3 reports the simulated SNR of the leapfrog (LF) ADC versus system order N
for target SNRs of 70/90/110 dB at a 1 MHz bandwidth, and shows the simulated
SNR approaches the target within ~3 dB for higher N. Here we reproduce a few
stable points end-to-end -- leapfrog design (the paper's equations) + analytical
reconstruction + ``cbadc.snr.snr_tone`` -- and check we land near the target.
"""

import numpy as np
import pytest

from cbadc import snr as snr_mod
from cbadc.fom import snr_to_enob
from cbadc.frontend import DAC, Constant, Frontend, Sinusoid

BW = 1e6


def _leapfrog_snr(target_dB, N, sim_size=1 << 16):
    """Build the paper's immediate-NRZ loop and measure reconstructed tone SNR."""
    af, OSR = Frontend.leapfrog(N=N, ENOB=snr_to_enob(target_dB), BW=BW)
    af.dac = DAC(Constant(1.0))  # reproduce the paper's immediate-NRZ loop
    fs = 1 / af.dt
    f_i = round(fs / 1024 * sim_size / fs) * fs / sim_size  # ~fs/1024 (paper)
    af.input = Sinusoid(1.0, f_i)  # full scale
    res = af.simulate(sim_size)
    u_hat = af.wiener_filter(OSR=OSR)(res["v"])[:, 0, 0]
    return snr_mod.snr_tone(u_hat, f_i, fs, trim=1 << 9, band=BW)


# Stable (target, N) points from Fig. 3; tolerance is generous to stay robust
# across platforms while still pinning the result near the published target.
@pytest.mark.parametrize("target, N", [(70, 4), (90, 4), (90, 6)])
def test_leapfrog_snr_matches_feyling_2023_fig3(target, N):
    """Published stable (target SNR, order) points reproduce within 5 dB.

    Synthesize the leapfrog, drive a coherent full-scale tone, reconstruct
    its decisions, and compare in-band SNR with the chosen figure point.
    """
    measured = _leapfrog_snr(target, N)
    assert measured == pytest.approx(target, abs=5.0)
