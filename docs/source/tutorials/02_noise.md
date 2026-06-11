---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Thermal noise

How to give a converter a physical noise specification and see its effect. We
reuse the [Quickstart](01_quickstart) converter and work with
{py:mod}`cbadc.noise`.

```{code-cell} python
import logging

import numpy as np
import matplotlib.pyplot as plt

from cbadc import AnalogFrontend, noise
from cbadc.analog_signal import Sinusoidal
from cbadc.fom import snr_to_enob

logging.disable(logging.WARNING)
BW, size, cycles = 1e6, 1 << 16, 523
```

## A device noise budget

The helpers turn device parameters into input-referred voltage-noise densities
[V/√Hz], which combine by root-sum-square.

```{code-cell} python
e_in = noise.combine_densities(
    noise.resistor_density(R=1e3),     # a 1 kΩ source resistor
    noise.ota_density(gm=2e-3),        # the input transconductor
)
print(f"input-referred noise density ≈ {e_in * 1e9:.1f} nV/√Hz")
```

## Inject it and measure

An input-referred PSD ``S_in`` maps to a state-noise covariance through the
input matrix; assign it to ``state_covariance`` and the simulator does the rest.

```{code-cell} python
def enob(e_in):
    af, OSR = AnalogFrontend.chain_of_integrators(N=4, ENOB=12, BW=BW)
    fs = 1 / af.dt
    if e_in:
        S_in = np.array([[e_in**2]])  # input-referred PSD [V²/Hz]
        af.state_covariance = af.input_referred_covariance_matrix(af, S_in)
    af.analog_signal = Sinusoidal(np.array([[0.5]]), np.array([[cycles * fs / size]]))
    u_hat = af.wiener_filter(OSR=OSR)(af.simulate(size)["v"])[:, 0, 0]
    p = np.abs(np.fft.rfft(u_hat)) ** 2
    band = p[1 : round(BW / (fs / size)) + 1]
    return snr_to_enob(10 * np.log10(p[cycles] / (band.sum() - p[cycles])))


print(f"no noise        : {enob(0):.1f} bits")
print(f"device noise    : {enob(e_in):.1f} bits")
```

The device noise barely moves the result: at a few nV/√Hz it sits well *below*
the quantisation floor — this converter is quantisation-limited.

## Where does noise start to bite?

Sweep the input-referred density to find the crossover into the noise-limited
regime.

```{code-cell} python
densities = np.array([2e-7, 5e-7, 1e-6, 2e-6, 5e-6])
bits = [enob(e) for e in densities]

plt.semilogx(densities * 1e9, bits, "o-")
plt.axhline(enob(0), ls="--", color="gray", label="quantisation floor")
plt.axvline(e_in * 1e9, ls=":", color="green", label="device noise")
plt.xlabel("input-referred noise [nV/√Hz]")
plt.ylabel("ENOB [bits]")
plt.legend()
plt.title("Quantisation- vs noise-limited");
```

The plateau on the left is quantisation-limited; once the input-referred noise
climbs past the floor (here ~0.5 µV/√Hz) every doubling costs about a bit. The
device-noise marker sits far to the left — headroom you could trade for power.
