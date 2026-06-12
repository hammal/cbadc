"""Data-aided (lstsq) reaches 120 dB on N=6/OSR=32 (Wiener gives 130).

Earlier this used fit='fft' and plateaued ~82-108 dB; the exact least-squares
fit ('lstsq') is far more accurate and reaches ~122 dB at *tiny* total data,
tolerating per-column length down to ~1.1*K with large J. The natural metric
for the data-aided path is snr_residual (it is fit to that broadband objective);
sub-band (0,0.8) drops the rolloff edge.

The cost knob for lstsq is the design matrix ~ (per_dec*J*K*M) floats -- small
per + moderate J keeps it cheap (sub-GB here). (A normal-equations / streaming
lstsq would remove even that ceiling.)
"""
import logging
import resource
import sys
import time

import numpy as np

logging.disable(logging.CRITICAL)
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import ZeroOrderHold
from cbadc.digital_backend import decimate
from cbadc import snr as snr_mod

BW = 1e6
N = 6
af, OSR = AnalogFrontend.leapfrog(OSR=32, N=N, BW=BW)
DSR = int(OSR)
dt = af.dt


def peak_gb():  # ru_maxrss is KB on Linux, bytes on macOS
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024**2 if sys.platform == "linux" else rss / 1024**3


# fixed held-out random reference (long columns so large-K trims leave samples)
rv = ZeroOrderHold.uniform_reference_signal(
    dt * DSR, -np.ones((1, 8)), np.ones((1, 8)), size=1 << 18, seed=99)
af.analog_signal = rv
sv = af.simulate(1 << 18)
vv = sv["v"]
uvr = decimate(sv["u"], DSR, method="direct")


def held_out(est):
    uh = est.reconstruct(vv)
    full = snr_mod.snr_residual(uh, uvr, trim=est.K)
    sub = snr_mod.snr_residual(uh, uvr, trim=est.K, band=(0.0, 0.8))
    return full, sub


print(f"leapfrog N={N} OSR={OSR:.1f} DSR={DSR}  Wiener ref ~130 dB  (fit=lstsq, snr_residual)")
print(f"{'K':>5} {'per_dec':>8} {'per/K':>6} {'J':>5} {'dec_data':>9} "
      f"{'full_dB':>8} {'sub_dB':>8} {'memGB':>6} {'t_s':>5}")
for K, per_dec, J in [(256, 512, 128), (256, 288, 256), (512, 1024, 128),
                      (1024, 2048, 64)]:
    t0 = time.time()
    sim_size = per_dec * DSR * J
    est = af.calibrate(DSR=DSR, K=K, J=J, sim_size=sim_size, fit="lstsq")
    full, sub = held_out(est)
    print(f"{K:>5} {per_dec:>8} {per_dec/K:>6.1f} {J:>5} {per_dec*J:>9} "
          f"{full:>8.1f} {sub:>8.1f} {peak_gb():>6.1f} {time.time()-t0:>5.0f}",
          flush=True)
