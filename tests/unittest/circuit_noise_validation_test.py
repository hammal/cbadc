"""Check circuit noise against independent linear stochastic models.

For ``dx = A x dt + dW`` with ``E[dW dW.T] = Q dt``, the stationary
covariance solves ``A P + P A.T + Q = 0``. These tests form ``Q`` from
component noise densities, independently of the implementation under test.
"""

import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov

from cbadc.circuit import ActiveRC, GmC
from cbadc.frontend import Frontend
from cbadc.noise import k_B


def _open_loop(dt=2e-6):
    """One stable state; the comparator and DAC have no path back to it."""
    return Frontend([[-5e4]], [[0.0, 0.0]], [[1.0]], dt=dt)


@pytest.mark.parametrize("temperature", [250.0, 400.0])
@pytest.mark.parametrize("capacitance", [1e-12, 2e-12])
def test_gmc_component_noise_matches_ou_variance(temperature, capacitance):
    """OTA and output-resistor currents add; variance follows T and C."""
    design, pole, gamma, ro = 1e-12, 5e4, 1.5, 1e7
    af = _open_loop()
    cases = (
        (gamma, np.inf),  # OTA only
        (0.0, ro),  # output resistor only
        (gamma, ro),  # independent sum
    )
    for noise_factor, resistance in cases:
        circuit = GmC(
            af,
            Cint=capacitance,
            C_design=design,
            Ro=resistance,
            gamma=noise_factor,
            temperature=temperature,
        )
        gm = design * pole
        conductance = noise_factor * gm + 1 / resistance
        q = 2 * k_B * temperature * conductance / capacitance**2
        decay = (gm + 1 / resistance) / capacitance
        expected_variance = q / (2 * decay)
        np.testing.assert_allclose(circuit.state_covariance[0, 0, 0, 0], q)
        np.testing.assert_allclose(
            solve_continuous_lyapunov(circuit.A[0, 0], -np.array([[q]]))[0, 0],
            expected_variance,
        )


@pytest.mark.parametrize("capacitance", [1e-12, 2e-12])
def test_active_rc_independent_noise_sources(capacitance):
    """Rint, input current and input voltage noise enter their own states."""
    af = _open_loop()
    design, gm, co, omega_u = 1e-12, 1e-4, 1e-10, 5e4
    temperature, current_density, voltage_density = 350.0, 5e-14, 2e-7
    base = dict(
        Cint=capacitance,
        C_design=design,
        gm=gm,
        Ro=np.inf,
        Co=co,
        omega_u=omega_u,
        temperature=temperature,
    )
    resistor = ActiveRC(af, **base)
    current = ActiveRC(
        af, **base, thermal_noise=False, opamp_current_noise=current_density
    )
    voltage = ActiveRC(
        af, **base, thermal_noise=False, opamp_voltage_noise=voltage_density
    )
    combined = ActiveRC(
        af,
        **base,
        opamp_current_noise=current_density,
        opamp_voltage_noise=voltage_density,
    )
    r_int = 1 / (omega_u * design)
    q_resistor = 2 * k_B * temperature / (r_int * capacitance**2)
    q_current = current_density**2 / (2 * capacitance**2)
    q_voltage = (gm * voltage_density / co) ** 2 / 2
    for circuit, diagonal in (
        (resistor, [q_resistor, 0]),
        (current, [q_current, 0]),
        (voltage, [0, q_voltage]),
        (combined, [q_resistor + q_current, q_voltage]),
    ):
        q = np.diag(diagonal)
        np.testing.assert_allclose(circuit.state_covariance[0, 0], q)
        # The two-pole response mixes both channels in the state variance.
        p = solve_continuous_lyapunov(circuit.A[0, 0], -q)
        assert p[0, 0] > 0
    np.testing.assert_allclose(
        combined.state_covariance,
        resistor.state_covariance + current.state_covariance + voltage.state_covariance,
    )


def test_inherited_noise_is_added_once_and_clock_step_preserves_variance():
    """A preexisting source adds once; exact discretization preserves OU P."""
    temperature, capacitance = 300.0, 1e-12
    af = _open_loop()
    af.state_covariance = [[2e-4]]
    circuit = GmC(
        af,
        Cint=capacitance,
        C_design=capacitance,
        Ro=np.inf,
        temperature=temperature,
        gamma=1.5,
    )
    thermal_q = 2 * k_B * temperature * 1.5 * 5e4 / capacitance
    np.testing.assert_allclose(circuit.state_covariance[0, 0, 0, 0], 2e-4 + thermal_q)
    expected_variance = (2e-4 + thermal_q) / (2 * 5e4)
    for dt in (1e-6, 4e-6):
        af = _open_loop(dt)
        af.state_covariance = [[2e-4]]
        circuit = GmC(af, Cint=capacitance, temperature=temperature)
        x = circuit.simulate(
            1 << 15, record="x", seed=15, engine="numpy", verbose=False
        )["x"][1024:, 0, 0]
        assert np.var(x) == pytest.approx(expected_variance, rel=0.15)
