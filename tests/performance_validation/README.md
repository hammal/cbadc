# Performance Validation

Validating the cbadc is done in several steps.

## Digital Estimators

Here we compare the different filter computation backends by comparing the resulting
offline matrices: Af, Ab, Bf, Bb, W.

The reference filter computation method is the mpmath version.

Note that since sympy is not currently able to solve the analytical expressions
we skip sympy for N > 1.

Swept parameters:
- N
- ENOB
- BW
- analog system
- digital control
- computation method
- eta2

## Simulators

Here the different simulators are compared to the analytical version for a number of cycles.

Currently the leap-frog is not possible to simulate with sympy for N > 2 and therefore are these tests skipped

Swept parameters:
- N
- ENOB
- BW
- analog system
- digital control
- simulation method
- excess delay

## Full Simulators

As complexity grows quickly we rely on the two previous comparison. Specifically, we fix

- simulation method -> FullSimulator
- reconstruction method -> BatchEstimator
- filter computation method -> mpmath
- eta2 -> ENOB
- excess delay -> 0.1


Swept parameters:
- N
- ENOB
- BW
- analog system
- digital control
