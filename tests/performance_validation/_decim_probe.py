"""Decimation-ceiling probe: is the ~83 dB fit_fft ceiling the anti-alias filter?

Generates raw (full-rate) calibration + held-out sims ONCE, then applies several
decimators to the SAME raw data, fits fit_fft with the best knobs, and reports
held-out snr_residual. If the ceiling moves with the decimator, the anti-alias
filter (not the estimator) is the floor.

Baseline decimator is the library default: single order-9 Chebyshev-I decimate
by DSR=21 -- which scipy flags as too steep (q>13). Variants test multi-stage,
linear-phase FIR, and a sharp Kaiser-windowed FIR.
"""
import logging
import time

import numpy as np
from scipy.signal import decimate as sp_decimate
from scipy.signal import firwin, filtfilt

logging.disable(logging.CRITICAL)
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import ZeroOrderHold
from cbadc.digital_backend import AdaptiveFIRFilter
from cbadc import snr as snr_mod

BW = 1e6
warm = 1 << 13
PER = 1 << 21
J = 32
af, OSR = AnalogFrontend.leapfrog(N=6, ENOB=20, BW=BW)
DSR = int(OSR)  # 21
M = af.M
K, NPERSEG, FLOOR = 512, 2048, 1e-12  # best knobs from the ablation


# ---- decimators (operate on axis 0 of (T, C, J)) ----
def d_iir9(x):  # library default
    return sp_decimate(x, DSR, axis=0, ftype="iir", n=9, zero_phase=True)


def d_iir_2stage(x):  # 21 = 3 * 7, better conditioned than one steep stage
    return sp_decimate(sp_decimate(x, 3, axis=0, ftype="iir", n=8, zero_phase=True),
                       7, axis=0, ftype="iir", n=8, zero_phase=True)


def d_fir(x):  # scipy FIR (linear phase), order 30*DSR
    return sp_decimate(x, DSR, axis=0, ftype="fir", n=30 * DSR, zero_phase=True)


def _kaiser(ntaps, atten_db):
    # sharp lowpass at ~0.9 of the decimated Nyquist, downsample by DSR
    taps = firwin(ntaps, 0.9 / DSR, window=("kaiser", 0.1102 * (atten_db - 8.7)))
    return taps


def d_kaiser(x, ntaps=64 * DSR + 1, atten=140.0):
    taps = _kaiser(ntaps, atten)
    y = filtfilt(taps, [1.0], x, axis=0)
    sl = [slice(None)] * x.ndim
    sl[0] = slice(None, None, DSR)
    return y[tuple(sl)]


DECIMATORS = {
    "iir9 (default)": d_iir9,
    "iir 2-stage": d_iir_2stage,
    "fir 30*DSR": d_fir,
    "kaiser 64*DSR/140dB": d_kaiser,
}


def make_raw(per, J, seed):
    ref = ZeroOrderHold.uniform_reference_signal(
        af.dt, -np.ones((1, J)), np.ones((1, J)), size=per + warm, seed=seed
    )
    af.analog_signal = ref
    sim = af.simulate(per + warm, domain="discrete")
    return sim["v"][warm:], sim["u"][warm:]  # full-rate (per, M, J), (per, L, J)


print(f"leapfrog N=6 OSR={OSR:.2f} DSR={DSR} M={M} J={J} per={PER}", flush=True)
print("generating raw cal + held-out ...", flush=True)
t0 = time.time()
rv, ru = make_raw(PER, J, seed=7)
vv, vu = make_raw(PER, 16, seed=99)
print(f"raw ready ({time.time()-t0:.0f}s)\n", flush=True)

print("=== Part A: fit_fft on IIR/FIR-decimated data ===", flush=True)
print(f"{'decimator':>22} {'rows':>7} {'SNR_dB':>7} {'t_s':>5}", flush=True)
for name, dec in DECIMATORS.items():
    t1 = time.time()
    dv, du = dec(rv), dec(ru)
    val_v, val_u = dec(vv), dec(vu)
    fir = AdaptiveFIRFilter(M, K, 1, dt=af.dt)
    fir.fit_fft(dv, du, nperseg=NPERSEG, floor=FLOOR)
    s = snr_mod.snr_residual(fir.convolve(val_v), val_u, trim=fir.K)
    print(f"{name:>22} {dv.shape[0]:>7} {s:>7.1f} {time.time()-t1:>5.1f}", flush=True)


# ---- Part B: FFT-native -- anti-alias by keeping in-band bins, no IIR ----
def fit_fullrate(vf, uf, nperseg_dec, window_name, floor):
    """Multichannel Wiener at full rate; decimation = keep in-band bins only.

    The K-tap filter lands at the decimated rate by inverse-FFTing just the
    nperseg_dec//2+1 in-band bins of the full-rate (nperseg_dec*DSR) spectra.
    The anti-alias is the window's stopband (sidelobes), not an IIR filter.
    """
    T, M_, Jn = vf.shape
    L = uf.shape[1]
    nps_full = nperseg_dec * DSR
    nps_full -= nps_full % 2
    nf_dec = nperseg_dec // 2 + 1  # in-band bins kept
    if window_name == "kaiser":
        win = np.kaiser(nps_full, 14.0)  # ~ -125 dB sidelobes
    else:
        win = np.hanning(nps_full)
    step = nps_full // 2
    Svv = np.zeros((nf_dec, M_, M_), complex)
    Svu = np.zeros((nf_dec, M_, L), complex)
    for j in range(Jn):
        for s in range(0, T - nps_full + 1, step):
            V = np.fft.rfft(vf[s:s + nps_full, :, j] * win[:, None], axis=0)[:nf_dec]
            U = np.fft.rfft(uf[s:s + nps_full, :, j] * win[:, None], axis=0)[:nf_dec]
            Svv += np.einsum("fm,fn->fmn", V.conj(), V)
            Svu += np.einsum("fm,fl->fml", V.conj(), U)
    reg = floor * np.trace(Svv, axis1=1, axis2=2).real.max() / M_
    H = np.linalg.pinv(Svv + reg * np.eye(M_)) @ Svu
    h = np.fft.fftshift(np.fft.irfft(H, n=nperseg_dec, axis=0), axes=0)
    c = nperseg_dec // 2
    off = (K - 1) // 2
    fir = AdaptiveFIRFilter(M_, K, L, dt=af.dt)
    fir._h[:] = h[c - off:c - off + K]
    fir._offset[:] = 0.0
    return fir


print("\n=== Part B: FFT-native (in-band bin selection, no IIR decimate) ===", flush=True)
print(f"{'window':>22} {'SNR_dB':>7} {'t_s':>5}", flush=True)
# reconstruct/measure on near-ideal (kaiser) decimated v,u for consistency
val_v_id, val_u_id = d_kaiser(vv), d_kaiser(vu)
for wname in ("hanning", "kaiser"):
    t1 = time.time()
    fir = fit_fullrate(rv, ru, NPERSEG, wname, FLOOR)
    s = snr_mod.snr_residual(fir.convolve(val_v_id), val_u_id, trim=fir.K)
    print(f"{wname:>22} {s:>7.1f} {time.time()-t1:>5.1f}", flush=True)
