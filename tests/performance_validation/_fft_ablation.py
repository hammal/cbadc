"""fit_fft ablation: which floor binds the SNR ceiling?

Isolates the four data-aided/fft error floors by sweeping them one at a time
on a fixed frontend, reporting held-out snr_residual:
  K        -> FIR truncation tail
  nperseg  -> Welch bias / frequency resolution
  floor    -> Tikhonov regularization (band-edge bias / dynamic range)
  J        -> estimation variance (averaging; cheap, vectorized)

J-spread runs the sim for `per` steps with J columns vectorized, so `per` sets
the column length (must exceed the largest nperseg) and J sets the number of
Welch averages at ~no extra sim time. Calibration data is generated once per J
and reused across K/nperseg/floor.
"""
import logging
import resource
import time

import numpy as np

logging.disable(logging.CRITICAL)
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import ZeroOrderHold
from cbadc.digital_backend import AdaptiveFIRFilter, decimate
from cbadc import snr as snr_mod

BW = 1e6
warm = 1 << 13
PER = 1 << 22  # raw steps/column -> ~199k decimated rows (>= max nperseg 131072)
af, OSR = AnalogFrontend.leapfrog(N=6, ENOB=20, BW=BW)
DSR = int(OSR)
M = af.M


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def make_data(per, J, seed):
    ref = ZeroOrderHold.uniform_reference_signal(
        af.dt, -np.ones((1, J)), np.ones((1, J)), size=per + warm, seed=seed
    )
    af.analog_signal = ref
    sim = af.simulate(per + warm, domain="discrete")
    dv = decimate(sim["v"][warm:], DSR, method="direct")
    du = decimate(sim["u"][warm:], DSR, method="direct")
    return dv, du


print(f"leapfrog N=6 OSR={OSR:.2f} DSR={DSR} M={M} per={PER}", flush=True)
t0 = time.time()
val_v, val_u = make_data(1 << 21, 16, seed=99)  # independent held-out set
print(f"held-out set ready ({time.time()-t0:.0f}s)", flush=True)


def heldout_snr(fir):
    return snr_mod.snr_residual(fir.convolve(val_v), val_u, trim=fir.K)


print(f"{'J':>4} {'K':>6} {'nperseg':>8} {'floor':>7} {'rows':>7} "
      f"{'SNR_dB':>7} {'t_s':>5} {'GB':>5}", flush=True)

for J in (16, 64):
    dv, du = make_data(PER, J, seed=7)
    rows = dv.shape[0]
    for K in (512, 2048, 8192):
        for nps_mult in (4, 16):
            nperseg = min(1 << int(np.ceil(np.log2(nps_mult * K))), rows)
            for floor in (1e-6, 1e-12):
                fir = AdaptiveFIRFilter(M, K, 1, dt=af.dt)
                t1 = time.time()
                fir.fit_fft(dv, du, nperseg=nperseg, floor=floor)
                dt = time.time() - t1
                s = heldout_snr(fir)
                print(f"{J:>4} {K:>6} {nperseg:>8} {floor:>7.0e} {rows:>7} "
                      f"{s:>7.1f} {dt:>5.1f} {peak_gb():>5.1f}", flush=True)
    del dv, du
