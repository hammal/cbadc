import pytest
import numpy as np
from cbadc.digital_estimator import DigitalEstimator
from cbadc.digital_estimator._filter_coefficients import FilterComputationBackend
from cbadc.fom import snr_from_dB, enob_to_snr
from .fixtures import setup_filter


@pytest.mark.parametrize(
    "N",
    [
        pytest.param(2, id="N=2"),
        pytest.param(12, id="N=12"),
        # pytest.param(8, id="N=8"),
        # pytest.param(10, id="N=10"),
    ],
)
@pytest.mark.parametrize(
    "ENOB",
    [
        pytest.param(12, id="ENOB=12"),
        # pytest.param(16, id="ENOB=16"),
        pytest.param(23, id="ENOB=22"),
    ],
)
@pytest.mark.parametrize(
    "BW",
    [pytest.param(1e0, id="BW=1Hz"), pytest.param(1e7, id="BW=10MHz")],
)
@pytest.mark.parametrize(
    "analog_system",
    [
        pytest.param('chain-of-integrators', id="chain_of_integrators_as"),
        pytest.param('leap_frog', id="leap_frog_as"),
    ],
)
@pytest.mark.parametrize(
    "digital_control",
    [
        pytest.param('default', id="default_dc"),
        pytest.param('switch-cap', id="switch_cap_dc"),
    ],
)
@pytest.mark.parametrize(
    'computation_method',
    [
        pytest.param(FilterComputationBackend.numpy, id="numpy"),
        pytest.param(FilterComputationBackend.sympy, id="sympy"),
        pytest.param(FilterComputationBackend.mpmath, id="mpmath"),
    ],
)
@pytest.mark.parametrize(
    'eta2', [pytest.param(1.0, id="eta2=1"), pytest.param('snr', id="eta2=ENOB")]
)
def test_filter(N, ENOB, BW, analog_system, digital_control, computation_method, eta2):
    res = setup_filter(N, ENOB, BW, analog_system, digital_control)
    K1 = 1 << 8
    K2 = 1 << 8

    if N < 5 and ENOB > 12:
        pytest.skip("Can't compute care. I'll conditioned")

    if N > 1 and computation_method == FilterComputationBackend.sympy:
        pytest.skip("Sympy don't work for filter orders > 1")

    if eta2 == 'snr':
        eta2 = snr_from_dB(enob_to_snr(ENOB))
    ref = DigitalEstimator(
        res['analog_system'],
        res['digital_control'],
        eta2,
        K1,
        K2,
        solver_type=FilterComputationBackend.mpmath,
    )
    filter = DigitalEstimator(
        res['analog_system'],
        res['digital_control'],
        eta2,
        K1,
        K2,
        solver_type=computation_method,
    )
    atol = 1e-12
    rtol = 1e-7
    np.testing.assert_allclose(filter.Af, ref.Af, rtol=rtol, atol=atol)
    np.testing.assert_allclose(filter.Ab, ref.Ab, rtol=rtol, atol=atol)
    np.testing.assert_allclose(filter.Bf, ref.Bf, rtol=rtol, atol=atol)
    np.testing.assert_allclose(filter.Bb, ref.Bb, rtol=rtol, atol=atol)
    np.testing.assert_allclose(filter.WT, ref.WT, rtol=rtol, atol=atol)
