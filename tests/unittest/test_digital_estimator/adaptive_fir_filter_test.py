import cbadc
from scipy.signal import firwin
import pytest


@pytest.fixture()
def setup():
    L = 2
    M = 5
    K = [1 << (10 - m) for m in range(M)]
    h0 = [firwin(K[0], 0.1), firwin(K[0], 0.2)]
    reference_index = [0, 0]
    return L, M, K, h0, reference_index


def test_init_filter(setup):
    L, M, K, h0, reference_index = setup
    h = cbadc.digital_estimator.initial_filter(h0, K, reference_index)
    assert len(h) == L
    assert len(h[0]) == len(K)
    assert h[0][0].size == K[0]


def test_initializer(setup):
    L, M, K, h0, reference_index = setup
    h = cbadc.digital_estimator.initial_filter(h0, K, reference_index)
    filter = cbadc.digital_estimator.AdaptiveFIRFilter(h, K, reference_index)
    assert filter.analog_system.L == L
    assert filter.analog_system.M == M
    assert filter.K == K
    assert filter.reference_index == reference_index


def test_number_of_filter_coefficients(setup):
    L, M, K, h0, reference_index = setup
    h = cbadc.digital_estimator.initial_filter(h0, K, reference_index)
    filter = cbadc.digital_estimator.AdaptiveFIRFilter(h, K, reference_index)
    assert filter.number_of_filter_coefficients() == L * K[0]
    epsilon = 1e-2
    filter.h = [[hm + epsilon for hm in hl] for hl in filter.h]
    assert filter.number_of_filter_coefficients() == L * sum(K)


def test_impulse_response(setup):
    L, M, K, h0, reference_index = setup
    h = cbadc.digital_estimator.initial_filter(h0, K, reference_index)
    filter = cbadc.digital_estimator.AdaptiveFIRFilter(h, K, reference_index)
    impulse_response = filter.impulse_response()
    assert impulse_response


def test_frequency_response(setup):
    L, M, K, h0, reference_index = setup
    h = cbadc.digital_estimator.initial_filter(h0, K, reference_index)
    filter = cbadc.digital_estimator.AdaptiveFIRFilter(h, K, reference_index)
    frequencies, frequency_response = filter.fir_filter_transfer_function()
    assert frequency_response