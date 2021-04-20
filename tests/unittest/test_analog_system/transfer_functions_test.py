from tests.fixture.chain_of_integrators import chain_of_integrators
import numpy as np
import os
import sys


def test_analog_transfer_function_matrix(chain_of_integrators):
    length = 1000
    omega = np.arange(length) * 2 * np.pi * 2 / chain_of_integrators["beta"]
    analogSystem = chain_of_integrators["system"]
    tf = analogSystem.analog_transfer_function_matrix(omega)
    assert tf.shape[2] == length
    assert tf.shape[0] == analogSystem.N
    assert tf.shape[1] == analogSystem.L


def test_transfer_function(chain_of_integrators):
    length = 1000
    omega = np.arange(length) * 2 * np.pi * 2 / chain_of_integrators["beta"]
    analogSystem = chain_of_integrators["system"]
    tf = analogSystem.transfer_function(omega)
    print(tf, tf.shape)
    assert tf.shape[2] == length
    assert tf.shape[0] == analogSystem.N_tilde
    assert tf.shape[1] == analogSystem.L
    assert isinstance(tf, (np.ndarray, np.generic))
    assert tf.dtype == np.complex128