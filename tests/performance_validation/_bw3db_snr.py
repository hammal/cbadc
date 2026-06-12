"""Does extending the loop-filter band to BW (bw_3dB) change the SNR ceiling?

Calibrates fit_fft on stock vs bw_3dB leapfrog and reports held-out
snr_residual over the full decimated band and over an inner sub-band
(excluding the band edge, where the earlier probe showed the ceiling lives).
"""
import logging
import time

import numpy as np
from scipy.signal import welch

logging.disable(logging.CRITICAL)
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import ZeroOrderHold
from cbadc.digital_backend import AdaptiveFIRFilter, decimate
from cbadc import snr as snr_mod

BW = 1e6
warm = 1 << 13
PER = 1 << 21
J = 32
K, NPS, FLOOR = 512, 2048, 1e-12


def make(af, DSR, per, Jn, seed):
    ref = ZeroOrderHold.uniform_reference_signal(
        af.dt, -np.ones((1, Jn)), np.ones((1, Jn)), size=per + warm, seed=seed)
    af.analog_signal = ref
    sim = af.simulate(per + warm, domain="discrete")
    return (decimate(sim["v"][warm:], DSR, method="direct"),
            decimate(sim["u"][warm:], DSR, method="direct"))


def subband_snr(u_hat, u_ref, frac):
    """SNR over the inner `frac` of the decimated band (pool over J)."""
    e = (u_hat - u_ref)[:, 0, :]
    s = u_ref[:, 0, :]
    f, Pe = welch(e, axis=0, nperseg=1 << 12)
    _, Ps = welch(s, axis=0, nperseg=1 << 12)
    band = f <= frac * f[-1]
    return 10 * np.log10(Ps[band].sum() / Pe[band].sum())


print(f"{'design':>8} {'OSR':>5} {'full_dB':>8} {'inner0.7_dB':>11} {'t_s':>5}", flush=True)
for label, kw in [("stock", {}), ("bw_3dB", {"bw_3dB": True})]:
    t0 = time.time()
    af, OSR = AnalogFrontend.leapfrog(N=6, ENOB=20, BW=BW, **kw)
    DSR = int(OSR)
    dv, du = make(af, DSR, PER, J, seed=7)
    fir = AdaptiveFIRFilter(af.M, K, 1, dt=af.dt)
    fir.fit_fft(dv, du, nperseg=NPS, floor=FLOOR)
    vv, vu = make(af, DSR, 1 << 20, 16, seed=99)
    uh = fir.convolve(vv)[K:]
    ur = vu[K:]
    full = snr_mod.snr_residual(fir.convolve(vv), vu, trim=K)
    inner = subband_snr(uh, ur, 0.7)
    print(f"{label:>8} {OSR:>5.1f} {full:>8.1f} {inner:>11.1f} {time.time()-t0:>5.0f}",
          flush=True)
