"""Tests for the quadrature modulators in :mod:`cbadc.analog_filter.modulator`."""

import numpy as np
import pytest

from cbadc.analog_filter import AnalogSystem
from cbadc.analog_filter.modulator import SineWaveModulator


def _quadrature_system(N=2, M=2, L=1):
    """A minimal even-sized analog system to wrap in a modulator."""
    A = np.zeros((N, N))
    # a simple rotation generator on the first 2x2 block
    A[0, 1], A[1, 0] = -1.0, 1.0
    B = np.zeros((N, L))
    B[0, 0] = 1.0
    CT = np.eye(N)
    Gamma = np.eye(N)
    Gamma_tildeT = np.eye(N)
    return AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)


def test_sine_modulator_requires_even_state_dimension():
    odd = AnalogSystem(
        np.zeros((3, 3)), np.zeros((3, 1)), np.eye(3), np.eye(3), np.eye(3)
    )
    with pytest.raises(ValueError):
        SineWaveModulator(odd, modulation_frequency=1e3)


def test_sine_rotation_matrix_is_orthogonal():
    m = SineWaveModulator(_quadrature_system(), modulation_frequency=1e3)
    for phi in (0.0, 0.3, 1.7, -2.1):
        R = m._rotation_matrix(phi)
        np.testing.assert_allclose(R @ R.T, np.eye(m.N), atol=1e-12)


def test_sine_modulate_demodulate_are_inverse():
    m = SineWaveModulator(_quadrature_system(), modulation_frequency=1e3)
    for t in (1e-4, 3.2e-4, 7.5e-4):
        # ``_rotation_matrix`` reuses one internal buffer, so copy each result
        # before holding both at once.
        up = m.modulate(t).copy()
        down = m.demodulate(t).copy()
        np.testing.assert_allclose(down @ up, np.eye(m.N), atol=1e-12)


def test_sine_modulate_at_zero_is_identity():
    m = SineWaveModulator(_quadrature_system(), modulation_frequency=1e3)
    np.testing.assert_allclose(m.modulate(0.0), np.eye(m.N), atol=1e-12)
