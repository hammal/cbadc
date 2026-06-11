---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Quickstart

Design a control-bounded A/D converter, simulate it, reconstruct the input from
its control signals, and measure the result — end to end in a handful of lines.

```{code-cell} python
import logging

import numpy as np
import matplotlib.pyplot as plt

from cbadc import AnalogFrontend
from cbadc.analog_signal import Sinusoidal
from cbadc.fom import snr_to_enob

logging.disable(logging.WARNING)  # quiet cbadc's progress logging
```

## Design

A 4th-order chain-of-integrators converter targeting 12 effective bits over a
1 MHz band. `OSR` is the resulting oversampling ratio and `af.dt` the sample
period.

```{code-cell} python
BW = 1e6
af, OSR = AnalogFrontend.chain_of_integrators(N=4, ENOB=12, BW=BW)
fs = 1 / af.dt
```

## Simulate

Drive it with a single in-band tone. We pick a *prime* number of cycles over the
record so the tone lands on one FFT bin (coherent sampling → a clean spectrum).

```{code-cell} python
size = 1 << 16
cycles = 523                     # prime → no spectral leakage
f_sig = cycles * fs / size       # ≈ 300 kHz, inside the band
af.analog_signal = Sinusoidal(np.array([[0.5]]), np.array([[f_sig]]))

sim = af.simulate(size)   # dict of t, u, v, x, y
```

## Reconstruct

The Wiener filter inverts the loop, estimating the input `u_hat` from the `M`
binary control signals `v`.

```{code-cell} python
u_hat = af.wiener_filter(OSR=OSR)(sim["v"])[:, 0, 0]
```

## Measure

In-band signal-to-noise ratio and the equivalent number of bits.

```{code-cell} python
spectrum = np.abs(np.fft.rfft(u_hat)) ** 2
in_band = spectrum[1 : round(BW / (fs / size)) + 1]
signal = spectrum[cycles]
snr = 10 * np.log10(signal / (in_band.sum() - signal))
print(f"SNR = {snr:.1f} dB    ENOB = {snr_to_enob(snr):.1f} bits")
```

```{code-cell} python
freq = np.fft.rfftfreq(size, af.dt)
plt.semilogx(freq[1:], 10 * np.log10(spectrum[1:] / spectrum.max()))
plt.axvline(BW, ls="--", color="k", label="bandwidth")
plt.xlabel("frequency [Hz]")
plt.ylabel("PSD [dB]")
plt.legend()
plt.title("Reconstructed spectrum");
```

The tone sits well above a noise floor that is shaped down across the band — the
signature of a control-bounded converter. From here, see the signal-chain and
noise tutorials.
