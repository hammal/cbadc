"""Validate fit='fft' on the configs where fit='lstsq' hit 107-134 GB."""
import logging, resource, time

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
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)  # Linux KB->GB


def measure(af, est, OSR, total):
    DSR = int(OSR)
    per = total // J
    ref = ZeroOrderHold.uniform_reference_signal(
        af.dt, -np.ones((1, J)), np.ones((1, J)), size=per + warm, seed=4242
    )
    af.analog_signal = ref
    sim = af.simulate(per + warm, domain="discrete")
    u_ref = decimate(sim["u"][warm:], DSR, method="direct")
    return snr_mod.snr_residual(est.reconstruct(sim["v"][warm:]), u_ref, trim=est.K)


# (target, N, K, cal_log, tot_log) -- same as overnight grid
for target, N, K, cal_log, tot_log in [(80, 4, 1 << 9, 20, 21), (120, 6, 1 << 10, 21, 22)]:
    af, OSR = AnalogFrontend.leapfrog(N=N, ENOB=snr_to_enob(target), BW=BW)
    t0 = time.time()
    est = af.calibrate(DSR=int(OSR), K=K, J=64, sim_size=1 << cal_log, fit="fft")
    tcal = time.time() - t0
    snr_d = measure(af, est, OSR, 1 << tot_log)
    print(
        f"target {target}: fit=fft  SNR={snr_d:.1f} dB  tcal={tcal:.0f}s  peakRSS={peak_gb():.1f} GB",
        flush=True,
    )
