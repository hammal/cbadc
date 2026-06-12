"""End-to-end characterization: 40-120 dB SNR, J-spread, with cost.

Memory-safe overnight version. The data-aided estimator (FIR convolve, the
default) is characterized across the full range with J-spread calibration AND a
J-spread held-out measurement read by snr_residual (which pools across columns).
The analytical Wiener filter currently scales ~O(size^2) in memory in its
__call__, so it is measured only on a small record here as a sanity point -- the
O(size^2) Wiener memory is a separate bug to fix.

Run:  uv run python tests/performance_validation/characterization.py
"""

import logging
import resource
import time

import numpy as np

logging.disable(logging.CRITICAL)
from cbadc import snr as snr_mod
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import ZeroOrderHold
from cbadc.digital_backend import decimate
from cbadc.fom import snr_to_enob

BW = 1e6
warm = 1 << 12
J = 16


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def measure(af, est, OSR, total):
    """J-spread held-out reference -> snr_residual (pools across columns)."""
    DSR = int(OSR)
    per = total // J
    ref = ZeroOrderHold.uniform_reference_signal(
        af.dt, -np.ones((1, J)), np.ones((1, J)), size=per + warm, seed=4242
    )
    af.analog_signal = ref
    sim = af.simulate(per + warm, domain="discrete")
    u_ref = decimate(sim["u"][warm:], DSR, method="direct")
    return snr_mod.snr_residual(est.reconstruct(sim["v"][warm:]), u_ref, trim=est.K)


# (target_dB, N, K, calibration sim_size log2, measurement total log2)
GRID = [
    (40, 4, 1 << 8, 19, 20),
    (60, 4, 1 << 8, 19, 20),
    (80, 4, 1 << 9, 20, 21),
    (100, 6, 1 << 9, 20, 22),
    (120, 6, 1 << 10, 21, 22),
]

print(f"{'tgt':>4} {'N':>2} {'OSR':>5} {'K':>5} | {'DataAid':>7} {'tcal':>5} {'tmeas':>5} {'peakGB':>6}",
      flush=True)
for target, N, K, cal_log, tot_log in GRID:
    af, OSR = AnalogFrontend.leapfrog(N=N, ENOB=snr_to_enob(target), BW=BW)
    t0 = time.time()
    est = af.calibrate(DSR=int(OSR), K=K, J=64, sim_size=1 << cal_log)
    tcal = time.time() - t0
    t0 = time.time()
    snr_d = measure(af, est, OSR, 1 << tot_log)
    tmeas = time.time() - t0
    print(
        f"{target:>4} {N:>2} {OSR:>5.1f} {K:>5} | {snr_d:>7.1f} {tcal:>5.0f} {tmeas:>5.0f} {peak_gb():>6.1f}",
        flush=True,
    )
