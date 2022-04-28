![cbadc](https://github.com/hammal/cbadc/actions/workflows/commit.yml/badge.svg?branch=develop)![cbadc](https://github.com/hammal/cbadc/actions/workflows/pull_request.yml/badge.svg?branch=develop)![pypi](https://github.com/hammal/cbadc/actions/workflows/pypi-deployment.yml/badge.svg)[![Documentation Status](https://readthedocs.org/projects/cbadc/badge/?version=latest)](https://cbadc.readthedocs.io/en/latest/?badge=latest)[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/hammal/cbadc)

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

## Develop Version
Alternatively, the latest develop branch can be installed by
```bash
git clone --branch develop https://github.com/hammal/cbadc.git && cd cbadc &&python -m pip install . && cd .. && rm -rf cbadc;
```

# Source Code
The source code is hosted on [https://github.com/hammal/cbadc](https://github.com/hammal/cbadc).

# Bugs and Issues
Please report problems at [https://github.com/hammal/cbadc/issues](https://github.com/hammal/cbadc/issues)

# Changelog

## 0.2.1

Added first calibration tools.

## 0.2.0

Major structural changes. Mainly motivated by improving simulators and filter coefficient computations to support switch-cap digital control simulations.

Specifically,
- [digital clock](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.analog_signal.clock.Clock.html#cbadc.analog_signal.clock.Clock) to aid the simulator and digital estimator computation
- [digital control](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.digital_control.digital_control.DigitalControl.html#cbadc.digital_control.digital_control.DigitalControl) and derived classes have a new interface to support [digital clock](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.analog_signal.clock.Clock.html#cbadc.analog_signal.clock.Clock), i.e, `DigitalControl(..., clock, ...)`.
- [Simulator](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.simulator.html)
  - [Analytical simulator](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.simulator.analytical_simulator.AnalyticalSimulator.html#cbadc.simulator.analytical_simulator.AnalyticalSimulator) implemented using [SymPy](https://www.sympy.org/en/index.html)
  - [Mpmath simulator](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.simulator.mp_simulator.MPSimulator.html#cbadc.simulator.mp_simulator.MPSimulator) implemented using [mpmath](https://mpmath.org)
  - Two Numerical simulators implemented using [NumPy](https://numpy.org)
    - [Full simulator](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.simulator.numerical_simulator.FullSimulator.html#cbadc.simulator.numerical_simulator.FullSimulator) the pervious default simulator.
    - [pre-computed simulator](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.simulator.numerical_simulator.PreComputedControlSignalsSimulator.html) same as Full simulator with the distinction that the control contributions are pre-computed.
  - The previous default [StateSpaceSimulator class](https://cbadc.readthedocs.io/en/v0.1.0/api/autosummary/cbadc.simulator.StateSpaceSimulator.html#cbadc.simulator.StateSpaceSimulator) has been replaced by the [`get_simulator`](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.simulator.wrapper.get_simulator.html#cbadc.simulator.wrapper.get_simulator) function. The simulation backend is chosen by passing an instance of [SimulatorType]().
  - The simulation clock period Ts is replaced by the [digital clock]() object and thus all simulator classes and the factory function `get_simulation(..., clock, ...)` is now instantiated with a clock determining the sample times.
- [DigitalEstimator](https://cbadc.readthedocs.io/en/v0.1.0/api/autosummary/cbadc.digital_estimator.DigitalEstimator.html#cbadc.digital_estimator.DigitalEstimator)
  - The default DigitalEstimator changes it's name to [BatchEstimator](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.digital_estimator.batch_estimator.BatchEstimator.html#cbadc.digital_estimator.batch_estimator.BatchEstimator)
  - an additional filter computation backend implemented with [mpmath](https://mpmath.org)
- Improved care implementation using [SymPy](https://www.sympy.org/en/index.html) instead of [SciPy](https://scipy.org).
- `cbadc.specification.get_chain_of_integrator` and `cbadc.specification.get_leap_frog` a computation aid to dimension chain-of-integrators and leap-frog analog-frontends to meet ENOB and BW specifications

Added verilog-ams circuit-level submodule

Such that circuit-level implementations can be
- constructed in Verilog-ams
- the resulting filter coefficients can be computed
- the resulting analog frontends can be simulated.

### 0.1.5

Added figures of merit [fom](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.fom.html#module-cbadc.fom) modul, [MurmannSurvey](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.fom.MurmannSurvey.html#cbadc.fom.MurmannSurvey) convenience class, and a new tutorial [The Murmann ADC Survey](https://cbadc.readthedocs.io/en/latest/tutorials/c_further/plot_a_Murmann_ADC_survey.html#sphx-glr-tutorials-c-further-plot-a-murmann-adc-survey-py).

### 0.1.2

Added fixed point arithmetics for FIR filter implementation.

### 0.1.1

Added support for switched capacitor digital control by adding a new:

- [simulator](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.simulator.SwitchedCapacitorStateSpaceSimulator.html#cbadc.simulator.SwitchedCapacitorStateSpaceSimulator),
- [digital control](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.digital_control.SwitchedCapacitorControl.html#cbadc.digital_control.SwitchedCapacitorControl),
- and modifications to the FIR [digital estimator](https://cbadc.readthedocs.io/en/latest/api/autosummary/cbadc.digital_estimator.FIRFilter.html#cbadc.digital_estimator.FIRFilter) to handle the switch cap case.

## 0.1.0

- First public release
