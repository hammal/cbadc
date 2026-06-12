"""Is the DataAided ~80 dB 'cap' real, or a trim/record-length artifact?

Long record + fixed minimal trim + J parallelism (needs a big-RAM machine).
For each estimator we report SNR vs trim; if the SNR is flat in trim and climbs
with K, the earlier 'cap' was the trim=2*K artifact eating the record, not the
fit. Run: uv run python tests/performance_validation/datadriven_high_snr.py
"""

import time

import numpy as np

from cbadc import snr as snr_mod
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import Sinusoidal
from cbadc.fom import snr_to_enob

BW = 1e6
N = 6
warm = 1 << 12
size = 1 << 21  # long record (≈2M samples)

af, OSR = AnalogFrontend.leapfrog(N=N, ENOB=snr_to_enob(120), BW=BW)
fs = 1 / af.dt
DSR = int(OSR)
fs_bb = fs / DSR
f = snr_mod.coherent_frequency(fs / 512, fs, size)
af.analog_signal = Sinusoidal(np.array([[1.0]]), np.array([[f]]))

t0 = time.time()
v = af.simulate(size + warm, domain="discrete")["v"][warm:]
print(f"sim: size={size} OSR={OSR:.1f} ({time.time() - t0:.1f}s)", flush=True)

# --- Wiener (full-rate), fixed trim = warm ---
sw = snr_mod.snr_tone(af.wiener_filter(OSR=OSR)(v)[:, 0, 0], f, fs, trim=warm, band=BW)
print(f"Wiener (trim=warm): {sw:.1f} dB", flush=True)

# --- DataAided, sweep K + J; SNR vs trim (fixed, NOT 2*K-scaled) ---
print(f"{'K':>6} {'J':>3} {'cal':>6} | trim= K     2K    4K   [dB]   (fit time)")
for K_log in (9, 10, 11, 12):
    K = 1 << K_log
    for J in (16,):
        t0 = time.time()
        est = af.calibrate(DSR=DSR, K=K, J=J, sim_size=1 << 20)
        u_hat = est.reconstruct(v)[:, 0, 0]
        ft = time.time() - t0
        row = [snr_mod.snr_tone(u_hat, f, fs_bb, trim=tr, band=BW) for tr in (K, 2 * K, 4 * K)]
        print(
            f"2^{K_log:<4} {J:>3} 2^20 | "
            + " ".join(f"{s:6.1f}" for s in row)
            + f"   ({ft:.0f}s)",
            flush=True,
        )
