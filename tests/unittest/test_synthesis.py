from cbadc.synthesis import get_chain_of_integrator, get_leap_frog
from cbadc.digital_estimator import BatchEstimator
import cbadc
import cbadc.fom
import numpy as np
from tests.fixture.chain_of_integrators import chain_of_integrators

N = 6
ENOB = 14
BW = 1e5
xi = 1
eta2 = 1.0
K1 = 1 << 9
K2 = 1 << 9


def test_get_chain_of_integrator():
    analog_frontend = get_chain_of_integrator(N=N, ENOB=ENOB, BW=BW, xi=xi)


def test_get_leap_frog():
    analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW, xi=xi)
