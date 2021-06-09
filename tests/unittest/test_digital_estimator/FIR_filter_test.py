import cbadc
import numpy as np
import scipy as sp
from tests.fixture.chain_of_integrators import chain_of_integrators


def test_write_C_header(chain_of_integrators):
    eta2 = 1.0
    K1 = 4
    K2 = 3
    digital_control = cbadc.digital_control.DigitalControl(
        1.0 / (2 * chain_of_integrators['beta']), 5)
    filter = cbadc.digital_estimator.FIRFilter(
        chain_of_integrators['system'], digital_control, eta2, K1, K2)
    filter.write_C_header('unittest')
