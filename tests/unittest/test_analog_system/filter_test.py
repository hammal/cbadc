import numpy as np
import cbadc


def test_Butterworth():
    N = 5
    # Set critical frequency
    Wn = 2 * np.pi * 44e3

    cbadc.analog_system.ButterWorth(N, Wn)


def test_ChebyshevI():
    N = 5
    # Set critical frequency
    Wn = 2 * np.pi * 44e3
    # Set maximum ripple
    rp = 1.0

    cbadc.analog_system.ChebyshevI(N, Wn, rp)


def test_ChebyshevII():
    N = 5
    # Set critical frequency
    Wn = 2 * np.pi * 44e3
    # Set maximum ripple
    rs = 80
    cbadc.analog_system.ChebyshevII(N, Wn, rs)


def test_Cauer():
    N = 5
    # Set critical frequency
    Wn = 2 * np.pi * 44e3
    # Set maximum ripple
    rp = 0.1
    rs = 80

    cbadc.analog_system.Cauer(N, Wn, rp, rs)
