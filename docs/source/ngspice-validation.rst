Validation against ngspice
==========================

:class:`~cbadc.circuit.GmC` and :class:`~cbadc.circuit.ActiveRC` generate
ngspice netlists from their realized components. The tests compare those
transients with cbadc using the same initial state and clock. These measured
results used ngspice 44; deterministic cases set ``thermal_noise=False``.

.. list-table::
   :header-rows: 1

   * - Example circuit
     - Same decisions
     - Maximum state error
   * - One-state gmC, finite ``Ro`` and ``Cp``
     - 64 / 64
     - 3.9e-7
   * - One-stage active-RC, finite ``Ro``
     - 48 / 48
     - 1.6e-6
   * - One-stage active-RC, infinite ``Ro``
     - 48 / 48
     - 1.6e-6
   * - N=3 gmC leapfrog
     - 4096 / 4096
     - 3.9e-4

The N=3 leapfrog also gave 71.26 dB reconstructed SNR in **both** simulators.
Other leapfrog tests cover N=4 and N=7. A decision near ``y=0`` can eventually
split their feedback paths, so those tests compare states only up to the first
different decision and compare reconstructed SNR over the full record.

Noise runs use 4096 periods and discard the first 128. The state-noise case
uses ``state_covariance = [[4000]]``; the comparator case uses
``comparator_input_noise_rms = 0.25`` V. The independent random draws are
compared by state spread and decision switching rate, not sample by sample.

================= ============================ =============================
Noise source      State std (cbadc / ngspice)   Switching rate (cbadc / ngspice)
================= ============================ =============================
State             0.3228 / 0.3231              0.5105 / 0.5155
Comparator        0.3099 / 0.3102              0.5452 / 0.5563
================= ============================ =============================

These runs check the simulator's noise mapping from the same covariance.
Separate thermal-noise tests inject resistor, OTA, and op-amp noise sources
independently, comparing variance with theory and ngspice, then checking
their sum, temperature/capacitance scaling, clock-step changes, and closed-loop
PSD and SNDR across seeds. Supported netlists have ``J = K = 1``, binary
comparators, and constant DAC pulses; batch tests select members separately.

Run ``make test-ngspice-docker`` with Docker, ``make test-ngspice-podman`` with
Podman, or ``make test-ngspice`` with a local ngspice that includes XSPICE.
