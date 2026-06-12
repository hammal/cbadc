"""Reach 120 dB SNR: find the (N, OSR) leapfrog whose Wiener reconstruction
crosses 120 dB, measured with a coherent tone.

Estimator : analytical Wiener (reaches the design SQNR; data-aided FIR is
            truncation/finite-data limited well below this).
Simulator : dsim (numba, exact ZOH; == simulate_sin for this).
Metric    : coherent snr_tone, trim past the forward-backward transient.
"""
import logging
import time

import numpy as np

logging.disable(logging.CRITICAL)
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import Sinusoidal
from cbadc import snr as snr_mod

BW = 1e6
size = 1 << 19
trim = 1 << 13
A_in = 0.2

print(f"{'N':>2} {'OSR':>4} {'SNR_dB':>8} {'ENOB':>6} {'|x|max':>7} {'eta2':>9} {'t_s':>5}")
for N, OSR in [(6, 22), (6, 32), (6, 48), (8, 32), (8, 48), (8, 64)]:
    t0 = time.time()
    af, OSR_a = AnalogFrontend.leapfrog(OSR=OSR, N=N, BW=BW)
    afd = af.discretize(dt=af.dt)
    fs = 1.0 / af.dt
    f = snr_mod.coherent_frequency(BW / 4, fs, size - 2 * trim)
    afd.analog_signal = Sinusoidal(np.array([[A_in]]), np.array([[f]]))
    sim = afd.simulate_dt(size)
    xmax = np.abs(sim["x"]).max()
    if xmax >= 1.0:
        print(f"{N:>2} {OSR:>4} {'OVERLOAD':>8} {'':>6} {xmax:>7.3f}")
        continue
    try:
        wf = af.wiener_filter(OSR=int(round(OSR_a)))
    except np.linalg.LinAlgError:
        print(f"{N:>2} {OSR:>4} {'CARE-FAIL':>8}  (eta2 too large; >150 dB regime)")
        continue
    u_hat = wf.evaluate(sim["v"])[:, 0, 0]
    s = snr_mod.snr_tone(u_hat, f, fs, trim=trim, band=BW)
    print(f"{N:>2} {OSR:>4} {s:>8.1f} {(s-1.76)/6.02:>6.2f} {xmax:>7.3f} "
          f"{wf.eta2:>9.1e} {time.time()-t0:>5.0f}", flush=True)
