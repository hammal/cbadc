"""Find the omega_p scale factor that puts the leapfrog loop-filter 3 dB edge at BW.

The leapfrog A and B_u scale linearly with omega_p (beta=-omega_p*2g, alpha=
omega_p/2g), so |G(w)| is invariant in w/omega_p -> scaling omega_p rescales the
frequency axis. Hence the factor is a single ratio r = BW / f_3dB(current),
verified on the open-loop signal transfer function.
"""
import logging
import numpy as np

logging.disable(logging.CRITICAL)
from cbadc.analog_frontend import AnalogFrontend


def loop_filter_mag(af, f):
    """|G(jw)| of the open-loop signal TF (input 0 -> states), per output."""
    jw = 1j * 2 * np.pi * f
    _, H = af.transfer_function(jw, open_loop=True, input_index=0)  # (size, M, 1)
    return np.abs(H[:, :, 0])  # (size, M)


def f_3dB(af, BW, metric="norm"):
    f = np.logspace(np.log10(BW / 1000), np.log10(BW * 20), 200000)
    mag = loop_filter_mag(af, f)
    g = np.linalg.norm(mag, axis=1) if metric == "norm" else mag[:, -1]
    ref = g[0]  # low-frequency passband reference
    # first crossing below -3 dB (1/sqrt(2) of passband ref)
    below = np.where(g < ref / np.sqrt(2))[0]
    return f[below[0]] if len(below) else np.nan, ref, g, f


for N, E in [(4, 12), (6, 20)]:
    BW = 1e6
    af, OSR = AnalogFrontend.leapfrog(N=N, ENOB=E, BW=BW)
    f3, ref, g, f = f_3dB(af, BW)
    r = BW / f3
    print(f"\n=== leapfrog N={N} ENOB={E} BW={BW:.2e} (OSR={OSR:.1f}) ===")
    print(f"current omega_p = omega_BW/2;  f_3dB = {f3:.3e} Hz = {f3/BW:.3f}*BW")
    print(f"  -> scale omega_p by r = BW/f_3dB = {r:.4f}  (beta,alpha *= {r:.4f})")

    # verify: rebuild with beta,alpha scaled by r (== omega_p *= r) and re-measure
    afc = af.analog_filter
    A, B = afc.A.copy(), afc.B.copy()
    # beta on subdiag, alpha on superdiag, B[0,0]=beta0 -- scale all by r
    A2 = np.diag(np.diag(A, 1) * r, 1) + np.diag(np.diag(A, -1) * r, -1) + np.diag(np.diag(A))
    B2 = B.copy(); B2[0, 0] *= r
    from scipy.signal import StateSpace
    from cbadc.digital_control import DigitalControl
    af2 = AnalogFrontend(StateSpace(A2, B2, afc.C.copy(), afc.D.copy()),
                         DigitalControl(N, af.digital_control.dt))
    f3b, _, _, _ = f_3dB(af2, BW)
    print(f"  verify: scaled f_3dB = {f3b:.3e} Hz = {f3b/BW:.4f}*BW  (target 1.000)")
