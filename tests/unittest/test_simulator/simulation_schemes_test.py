"""The discrete / continuous simulation schemes agree at the reconstruction level.

The raw control bitstream is chaotically sensitive (feedback), so only the
*reconstructed input* is a stable function of the scheme. We therefore reconstruct
a known tone through each scheme and check they agree -- with ``dsim`` (discrete,
the trusted reference) as the anchor. A speed/precision benchmark of the four
schemes lives in the docstring of ``AnalogFrontend.simulate``.
"""

import numpy as np
import pytest

from cbadc import snr as snr_mod
from cbadc.analog_frontend import AnalogFrontend
from cbadc.analog_signal import Sinusoidal


def _recon_snr(af, wf, size, f, fs, **sim_kwargs):
    af.analog_signal = Sinusoidal(np.array([[0.5]]), np.array([[f]]))
    v = af.simulate(size, **sim_kwargs)["v"]
    u_hat = wf(v)[:, 0, 0]
    return snr_mod.snr_tone(u_hat, f, fs, trim=1 << 8, band=fs / (2 * 16))


def test_discrete_and_continuous_schemes_agree():
    af, OSR = AnalogFrontend.chain_of_integrators(N=3, ENOB=10, BW=1e6)
    fs = 1 / af.dt
    size = 1 << 12
    f = snr_mod.coherent_frequency(fs / 64, fs, size)
    wf = af.wiener_filter(OSR=OSR)

    snr_discrete = _recon_snr(af, wf, size, f, fs, domain="discrete")
    snr_fast = _recon_snr(af, wf, size, f, fs, domain="continuous", precision="fast")
    snr_high = _recon_snr(af, wf, size, f, fs, domain="continuous", precision="high")

    # all schemes reconstruct the tone, and agree within a few dB
    assert snr_discrete > 30
    assert snr_fast == pytest.approx(snr_discrete, abs=4.0)
    assert snr_high == pytest.approx(snr_discrete, abs=4.0)


def test_domain_precision_back_compat_and_validation():
    af, _ = AnalogFrontend.chain_of_integrators(N=2, ENOB=8, BW=1e6)
    af.analog_signal = Sinusoidal(np.array([[0.5]]), np.array([[3e5]]))
    # default == discrete domain == legacy method='dsim'
    default = af.simulate(200)["v"]
    assert np.array_equal(default, af.simulate(200, domain="discrete")["v"])
    assert np.array_equal(default, af.simulate(200, method="dsim")["v"])
    # invalid selectors raise
    with pytest.raises(ValueError):
        af.simulate(10, domain="nonsense")
    with pytest.raises(ValueError):
        af.simulate(10, domain="continuous", precision="nonsense")
