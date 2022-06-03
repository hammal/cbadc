import numpy as np
import cbadc.circuit_level.noise_models as nm


def test_resistor_voltage_noise_formula():
    # For a 1KOhm resistor at room temperature
    R = 1e3
    T = 3e2
    np.testing.assert_allclose(
        nm.resistor_noise_voltage_source_model(R, T), 4.07e-9**2, rtol=1e-3
    )


def test_resistor_current_noise_formula():
    # For a 1KOhm resistor at room temperature
    R = 1e3
    T = 3e2
    np.testing.assert_allclose(
        nm.resistor_noise_current_source_model(R, T), 4.07e-9**2 / R**2, rtol=1e-3
    )


def test_KTC_formula():
    # For a 1nF capacitor at 300K temperature
    C = 1e-9
    T = 3e2
    np.testing.assert_allclose(nm.kTC_noise_voltage_source(C, T), 2e-6**2, rtol=1e-1)
