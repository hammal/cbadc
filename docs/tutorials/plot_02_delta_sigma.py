"""
Delta-sigma: DT, CT, and MASH
=============================

Build loops with :mod:`cbadc.delsig`, then measure their reconstructed-input SNDR
with :func:`cbadc.snr.simulate_snr`. All tests use a -6 dBFS coherent tone;
the MASH example uses OSR 25, and the others use OSR 32.
"""

import warnings

from scipy.signal import BadCoefficients

from cbadc import delsig, snr
from cbadc.frontend import Frontend

warnings.simplefilter("ignore", BadCoefficients)  # toolbox NTF conversion
OSR = 32


def sndr(af, osr=OSR):
    """Measure one amplitude using the default Wiener input estimate."""
    result = snr.simulate_snr(af, osr, amp=[-6], full_scale=1.0, k=12, warmup=512)
    return result["snr_u_hat"][0, 0]


# %%
# Discrete time: the toolbox ABCD matrix can be used directly. ``levels=5``
# changes the quantizer while keeping its controls ``v`` in [-1, 1].
ntf = delsig.synthesizeNTF(order=3, OSR=OSR, opt=1, H_inf=1.5)
a, g, b, c = delsig.realizeNTF(ntf, "CRFB")
ABCD = delsig.stuffABCD(a, g, b, c, "CRFB")
results = {
    "DT, 2 levels": sndr(Frontend.dtsdm(ABCD, levels=2)),
    "DT, 5 levels": sndr(Frontend.dtsdm(ABCD, levels=5)),
}

# %%
# Continuous time: ``tdac`` sets the DAC pulse window in clock periods.
# The bridge adjusts loop coefficients for each pulse shape.
for name, window in (("CT, NRZ", (0, 1)), ("CT, RZ", (0.25, 0.75))):
    af, _ = Frontend.from_delsig_ntf(ntf, BW=1e6, OSR=OSR, form="FB", tdac=window)
    results[name] = sndr(af)

# %%
# MASH: two second-order stages form one frontend with two controls. Its
# Wiener reconstruction combines both controls, including stage cancellation.
mash, mash_osr = Frontend.mash([delsig.synthesizeNTF(2, 25, 1)] * 2, BW=1e6, OSR=25)
results["MASH 2-2, OSR 25"] = sndr(mash, mash_osr)

for name, value in results.items():
    print(f"{name:18s} {value:5.1f} dB")
