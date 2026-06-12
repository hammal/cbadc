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
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import Sinusoidal
from cbadc.fom import snr_to_enob

BW = 1e6


def _leapfrog_snr(target_dB, N, sim_size=1 << 16):
    af, OSR = AnalogFrontend.leapfrog(N=N, ENOB=snr_to_enob(target_dB), BW=BW)
    fs = 1 / af.dt
    f_i = round(fs / 1024 * sim_size / fs) * fs / sim_size  # ~fs/1024 (paper)
    af.analog_signal = Sinusoidal(np.array([[1.0]]), np.array([[f_i]]))  # full scale
    res = af.simulate(sim_size)
    u_hat = af.wiener_filter(OSR=OSR)(res["v"])[:, 0, 0]
    return snr_mod.snr_tone(u_hat, f_i, fs, trim=1 << 9, band=BW)


# Stable (target, N) points from Fig. 3; tolerance is generous to stay robust
# across platforms while still pinning the result near the published target.
@pytest.mark.parametrize("target, N", [(70, 4), (90, 4), (90, 6)])
def test_leapfrog_snr_matches_feyling_2023_fig3(target, N):
    measured = _leapfrog_snr(target, N)
    assert measured == pytest.approx(target, abs=5.0)
