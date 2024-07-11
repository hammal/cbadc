![cbadc](https://github.com/hammal/cbadc/actions/workflows/develop.yml/badge.svg?branch=develop)![cbadc](https://github.com/hammal/cbadc/actions/workflows/develop.yml/badge.svg?branch=master)![pypi](https://github.com/hammal/cbadc/actions/workflows/pypi-deployment.yml/badge.svg)[![Documentation Status](https://readthedocs.org/projects/cbadc/badge/?version=latest)](https://cbadc.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Control-Bounded A/D Conversion (cbadc) Toolbox

This package is intended as a design tool for aiding the construction of control-bounded A/D converters.
Specifically, it is capable of:

- **Generating** transfer functions for analog systems and/or digital estimator parametrizations.
- **Estimating** samples from control signals.
- **Simulating** analog system and digital control interactions.

# Documentation

The project's official documentation can be found at [Read the Docs](https://cbadc.readthedocs.io/en/latest/).

<!-- # Background & References

For a in-depth description of the control-bounded conversion concept consider the following publications
- [Control-bounded analog-to-digital conversion, circuits, systems, and signal processing, 2021](https://doi.org/10.1007/s00034-021-01837-z)
- [Control-bounded converters, Ph.D. Thesis, 2020](https://doi.org/10.3929/ethz-b-000469192).
- [Control-bounded analog-to-digital conversion: transfer functions analysis, proof of concept, and digital filter implementation, arXiv:2001.05929, 2020](https://arxiv.org/abs/2001.05929)
- [Control-based analog-to-digital conversion without sampling and quantization, information theory & applications workshop, 2015](https://ieeexplore.ieee.org/document/7308975)
- [Analog-to-digital conversion using unstable filters, information theory & applications workshop, 2011](https://ieeexplore.ieee.org/abstract/document/5743620) -->

# Installation

Install [cbadc](https://pypi.org/project/cbadc/) by typing

```bash
pip install cbadc
```

into your console. Note that, currently cbadc is only supported for Python3.9 and later.

## Develop Version

Alternatively, the latest develop branch can be installed by

```bash
pip install git+https://github.com/hammal/cbadc.git@develop
```

# Source Code

The source code is hosted on [https://github.com/hammal/cbadc](https://github.com/hammal/cbadc).

# Bugs and Issues

Please report problems at [https://github.com/hammal/cbadc/issues](https://github.com/hammal/cbadc/issues)

# Changelog
see [CHANGELOG](./CHANGELOG.md)
