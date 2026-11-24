# Changelog

## 0.4.0 (unreleased)

This release centers cbadc on one state-space frontend and one simulation
loop. Define the model, simulate its decisions, then measure the result.
Circuit realizations and delta-sigma synthesis remain separate modules.

### Core

- One `Frontend(A, B, C, D, dt=...)` replaces the separate analog-system,
  signal, digital-control, and simulator objects. Matrices use `(J, K, ...)`
  for independent batch members and cyclic phases.
- One `simulate` call runs the clocked recursion. Continuous-time linear
  models use one-step maps from matrix exponentials; discrete-time models
  supply their own maps. NumPy and Numba run the same recursion. The twin replays
  decisions; the default records are `v`, `u`, and `y_hat`. State and
  comparator records are requested explicitly.
- Sources, filters, DAC pulses, and predetermined samples share the
  `StateSpace` composition model. The continuous-time DAC default is an
  equal-area pulse active from `T/4` to `3T/4`; discrete-time feedback is
  immediate.
- `simulate_snr` returns decision, raw-twin, and Wiener-input SNDR curves
  with replayable arguments, including batched frontends. Single-comparator
  delta-sigma frontends use the preceding decision as their twin readout.
- `GmC` and `ActiveRC` realize component-level frontends. Nominal component
  sizing stays fixed when actual capacitances vary. Thermal noise is enabled
  by default; comparator and op-amp noise are explicit. Their ngspice netlists
  support circuit validation.
- The delta-sigma bridge is an optional dependency. Its output uses the same
  `Frontend` simulation and measurement path.

### Migration

- The former `AnalogFrontend`, `AnalogSignal`, `DigitalControl`, multiple
  simulator classes, and adaptive estimators are no longer part of the
  public API.
- Polynomial `a2`/`a3` simulation, clock jitter, and the `memory_limit`,
  `record_y`, and `twin` switches were removed. Use `record=` to choose
  stored results.
- Physical netlists belong to `GmC` and `ActiveRC`, rather than a general
  circuit-generation framework.

### Validation

- Unit and doctests cover the state-space and simulation contracts. The
  ngspice suite independently checks circuit dynamics, decisions, and noise
  for supported gm-C and active-RC frontends. Leapfrog resolution tests cover
  orders 2–12, multiple pulse timings, and long records.
- Publishing a GitHub Release automatically builds, validates, and publishes
  the wheel and sdist to PyPI.

## 0.3.3

- Improved adaptive filtering, random signals, and precomputed simulation.

## 0.3.2

- Refactored Python simulators and quadrature analog/digital control.

## 0.2.3

- Added circuit-level ngspice tools.

## 0.2.1

- Added calibration tools.

## 0.2.0

- Reworked the digital clock and control interfaces and added simulation and
  estimator backends.

## 0.1.5

- Added figures of merit and ADC survey tools.

## 0.1.2

- Added fixed-point arithmetic for FIR filters.

## 0.1.1

- Added switched-capacitor simulation and control.

## 0.1.0

- First public release.
