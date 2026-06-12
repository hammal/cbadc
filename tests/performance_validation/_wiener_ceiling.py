"""Isolate the Wiener ~96 dB ceiling: is it CARE/expm conditioning, sim data,
or sim_size?

Sweeps eta2 (the target knob) on a fixed continuous-time leapfrog, reconstructs
a sinusoid, and reports achieved snr_tone alongside CARE diagnostics
(solution norm, residual, conditioning of V_f+V_b and of the expm argument).
"""
import logging
import numpy as np
from scipy.linalg import solve_continuous_are as care

logging.disable(logging.CRITICAL)
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import Sinusoidal
from cbadc import snr as snr_mod

BW = 1e6
N = 6
af, OSR = AnalogFrontend.leapfrog(N=N, ENOB=20, BW=BW)
afd = af.discretize(dt=af.dt)
dt = af.dt
fs = 1.0 / dt
L = af.L

size = 1 << 18
A_in = 0.2
trim = 1 << 12
# coherent tone: snr_tone is leakage-free only for an integer number of cycles
# in the post-trim window -- a non-coherent tone floors/oscillates the reading.
f_sig = snr_mod.coherent_frequency(BW / 4, 1.0 / af.dt, size - 2 * trim)

# end-to-end: build the (now nondimensionalized) Wiener filter, reconstruct a
# tone, and report achieved SNR vs eta2 -- the cap should now be the optimum,
# not a CARE-failure wall.
afd.analog_signal = Sinusoidal(np.array([[A_in]]), np.array([[f_sig]]))
sim = afd.simulate(size, domain="discrete")
assert np.abs(sim["x"]).max() < 1.0 + 1e-9, "state overloaded -- SNR invalid"
print(f"leapfrog N={N} BW={BW:.0e} OSR={OSR:.1f}  tone={f_sig:.2e}  |x|max={np.abs(sim['x']).max():.3f}")
jw = 1j * np.pi / (OSR * dt)
_, tf = af.transfer_function(np.array([jw]), input_index=0, output_index=-1)
print(f"eta2(OSR) = {float(np.abs(tf[0,0,0])**2):.2e}")
print(f"{'eta2':>8} {'SNR_dB':>8} {'built':>6}")
for eta2 in (1e9, 1e7, 1e5, 1e3, 1e1, 1e0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8):
    try:
        wf = af.wiener_filter(eta2=float(eta2))
        u_hat = wf.evaluate(sim["v"])
        s = snr_mod.snr_tone(u_hat[:, 0, 0], f_sig, fs, trim=trim, band=BW)
        print(f"{eta2:>8.0e} {s:>8.1f} {'ok':>6}", flush=True)
    except Exception as e:
        print(f"{eta2:>8.0e} {'--':>8} {'FAIL':>6}  {type(e).__name__}", flush=True)


