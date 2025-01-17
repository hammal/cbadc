from cbadc import AnalogFrontend

N = 6
ENOB = 14
BW = 1e5
xi = 1
eta2 = 1.0
K1 = 1 << 9
K2 = 1 << 9


def test_get_leap_frog():
    analog_frontend = AnalogFrontend.leapfrog(N=N, ENOB=ENOB, BW=BW, xi=xi)
