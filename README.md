[![Tests](https://github.com/hammal/cbadc/actions/workflows/develop.yml/badge.svg?branch=develop)](https://github.com/hammal/cbadc/actions/workflows/develop.yml)
[![Documentation](https://readthedocs.org/projects/cbadc/badge/?version=latest)](https://cbadc.readthedocs.io/en/latest/)

# cbadc

cbadc models control-bounded A/D converters with one clocked state-space
frontend. Define `A, B, C, D`, attach an input and DAC pulse, then simulate
decisions and the built-in digital twin. The 0.4.0 release narrows the core to
modelling, simulation, reconstruction, and measurement.

## Install

cbadc requires Python 3.11 or later:

```bash
uv add cbadc
```

The delta-sigma synthesis bridge is optional: `uv add 'cbadc[delsig]'` (or
`pip install 'cbadc[delsig]'`). This installs
[cbadc-deltasigma](https://pypi.org/project/cbadc-deltasigma/) from PyPI.

## Example

```python
from cbadc import Frontend, Sinusoid
from cbadc.snr import simulate_snr

af, osr = Frontend.leapfrog(N=4, OSR=16, BW=1e6)
af.input = Sinusoid(0.5, af.fs / 128)
record = af.simulate(1 << 14)  # v, u, and digital-twin y_hat
result = simulate_snr(af, osr, amp=[-12, -6, 0], full_scale=1.0)
sndr, amplitudes = result["snr_u_hat"], result["amp"]
```

The matrices have leading `(J, K)` axes for independent batch members and
cyclic phases. Continuous-time linear dynamics are discretized with matrix
exponentials. `record=` selects additional state and comparator traces.
`GmC` and `ActiveRC` provide physical realizations and ngspice validation;
Wiener reconstruction remains available for estimating the input from
decisions.

## Documentation and source

- [Documentation](https://cbadc.readthedocs.io/en/latest/) for the published
  version. For this development tree, run `make docs` and open
  `docs/build/html/index.html`.
- [Source and issues](https://github.com/hammal/cbadc).
- [Changelog](./CHANGELOG.md).
