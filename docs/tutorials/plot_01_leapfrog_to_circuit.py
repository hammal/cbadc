"""
From a leapfrog to a circuit
============================

Compare one ideal loop with gm-C and active-RC realizations. Each realization
keeps the same comparator and DAC; its components change the state equations.
Thermal noise is on by default. SNDR uses the Wiener input estimate.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from cbadc import snr
from cbadc.circuit import ActiveRC, GmC
from cbadc.frontend import Frontend, Sinusoid

BW = 1e6
ideal, OSR = Frontend.leapfrog(N=4, ENOB=12, BW=BW)


def sndr(model):
    """Measure one -6 dBFS tone with a coherent Hann-windowed FFT."""
    result = snr.simulate_snr(model, OSR, amp=[-6], full_scale=1.0, k=12, warmup=512)
    return result["snr_u_hat"][0, 0]


# %%
# Fixed transconductors and actual Cint determine gm-C dynamics; finite Ro
# leaks each state. The model adds OTA and Ro thermal noise automatically.
Cint = 2e-12
gmc = GmC(ideal, Cint=Cint, Ro=1e6)

# %%
# Active-RC adds one op-amp pole per integrator. R_int is sized from Cint;
# the op-amp's input noise densities require separate characterization.
gm = 1e-3
rc = ActiveRC(
    ideal,
    Cint=Cint,
    gm=gm,
    Ro=1e6,
    Co=gm / (2 * np.pi * 300e6),  # 300 MHz gain-bandwidth product
    opamp_voltage_noise=5e-9,  # V/sqrt(Hz)
    opamp_current_noise=1e-12,  # A/sqrt(Hz)
    comparator_input_noise_rms=1e-3,  # V rms per decision
)
flicker = rc.with_flicker(node=0, f_corner=3e5)

for name, model in (
    ("ideal", ideal),
    ("gm-C", gmc),
    ("active-RC", rc),
    ("+ flicker", flicker),
):
    print(f"{name:10s} {sndr(model):5.1f} dB")

# %%
# With the input off, the digital-twin PSD shows the added low-frequency
# flicker noise. The same comparator and op-amp noise remain in both curves.
fig, ax = plt.subplots()
for name, model in (("thermal", rc), ("thermal + flicker", flicker)):
    model.input = Sinusoid(0.0, BW / 4)
    y_hat = model.simulate(1 << 15, record="y_hat", seed=7, verbose=False)["y_hat"][
        2048:, 0, 0
    ]
    f, psd = welch(y_hat, fs=model.fs, nperseg=1 << 12)
    ax.loglog(f[1:], psd[1:], label=name)
ax.set(xlabel="f [Hz]", ylabel="digital-twin PSD [1/Hz]", xlim=(f[1], BW))
ax.legend()
