![cbadc](https://github.com/hammal/cbadc/actions/workflows/testing.yml/badge.svg) [![Documentation Status](https://readthedocs.org/projects/cbadc/badge/?version=latest)](https://cbadc.readthedocs.io/en/latest/?badge=latest)[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/hammal/cbadc)

# Control-Bounded A/D Conversion (cbadc) Toolbox

This package is intended as a design tool for aiding the construction of control-bounded A/D converters.
Specifically, it is capable of:

- **Generating** transfer functions for analog systems and/or digital estimator parametrizations.
- **Estimating** samples from control signals.
- **Simulating** analog system and digital control interactions.

# Documentation

The projects official documentation can be found at [Read the Docs](https://cbadc.readthedocs.io/en/latest/).

# Background & References

For a in depth description of the control-bounded conversion concept consider the following publications
- [Control-bounded analog-to-digital conversion, circuits, systems, and signal processing, 2021](https://doi.org/10.1007/s00034-021-01837-z)
- [Control-bounded converters, PhD Thesis, 2020](https://doi.org/10.3929/ethz-b-000469192).
- [Control-bounded analog-to-digital conversion: transfer functions analysis, proof of concept, and digital filter implementation, arXiv:2001.05929, 2020](https://arxiv.org/abs/2001.05929)
- [Control-based analog-to-digital conversion without sampling and quantization, information theory & applications workshop, 2015](https://ieeexplore.ieee.org/document/7308975)
- [Analog-to-digital conversion using unstable filters, information theory & applications workshop, 2011](https://ieeexplore.ieee.org/abstract/document/5743620)

# Installation

Install [cbadc](https://pypi.org/project/cbadc/) by typing
 
```bash
pip install cbadc
```

into your console. Note that, currently cbadc is only supported for Python3.8 and later.

# Changelog

## 0.1.0

- First public release

### 0.1.1

Added support for switched capacitor digital control by adding a new:

- [simulator](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.simulator.SwitchedCapacitorStateSpaceSimulator.html#cbadc.simulator.SwitchedCapacitorStateSpaceSimulator),
- [digital control](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.digital_control.SwitchedCapacitorControl.html#cbadc.digital_control.SwitchedCapacitorControl),
- and modifications to the FIR [digital estimator](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.digital_estimator.FIRFilter.html#cbadc.digital_estimator.FIRFilter) to handle the switch cap case.

### 0.1.2

Added fixed point arithmetics for FIR filter implementation.

### 0.1.5

Added figures of merit (fom) module and MurmannSurvey convenience class.
