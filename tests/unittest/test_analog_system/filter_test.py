import numpy as np
import cbadc
import scipy.signal


def test_Butterworth():
    N = 2
    # Set critical frequency
    Wn = 1.0

    cbadc.analog_system.ButterWorth(N, Wn)


def test_ButterWorth_transferfunction():
    N = 4
    Wn = 1e3
    butter_worth_system = cbadc.analog_system.ButterWorth(N, Wn)

    b, a = scipy.signal.butter(N, Wn, btype='low', analog=True)
    print("b, a\n")
    print(b)
    print(a)

    w, h = scipy.signal.freqs(b, a)

    tf = butter_worth_system.transfer_function_matrix(w)[:, 0, :].flatten()

    if not np.allclose(h, tf):
        print(h, tf)
        print(h - tf)
        raise BaseException("Filter mismatch")


def test_ChebyshevI():
    N = 4
    # Set critical frequency
    Wn = 2 * np.pi * 44e3
    # Set maximum ripple
    rp = 1.0

    cbadc.analog_system.ChebyshevI(N, Wn, rp)


def test_ChebyshevI_transferfunction():
    N = 4
    Wn = 1e3
    rp = np.sqrt(2)
    chebyshevI_worth_system = cbadc.analog_system.ChebyshevI(N, Wn, rp)

    b, a = scipy.signal.cheby1(N, rp, Wn, btype='low', analog=True)
    print("b, a\n")
    print(b)
    print(a)

    w, h = scipy.signal.freqs(b, a)

    tf = chebyshevI_worth_system.transfer_function_matrix(w)

    if not np.allclose(h, tf):
        print(h, tf)
        print(h - tf)
        raise BaseException("Filter mismatch")


def test_ChebyshevII():
    N = 5
    # Set critical frequency
    Wn = 2 * np.pi * 44e3
    # Set maximum ripple
    rs = 80
    cbadc.analog_system.ChebyshevII(N, Wn, rs)


def test_ChebyshevII_transferfunction():
    N = 4
    Wn = 1e3
    rs = 60
    chebyshevII_worth_system = cbadc.analog_system.ChebyshevII(N, Wn, rs)

    b, a = scipy.signal.cheby2(N, rs, Wn, btype='low', analog=True)
    print("b, a\n")
    print(b)
    print(a)

    w, h = scipy.signal.freqs(b, a)

    tf = chebyshevII_worth_system.transfer_function_matrix(w)

    if not np.allclose(h, tf):
        print(h, tf)
        print(h - tf)
        raise BaseException("Filter mismatch")


def test_Cauer():
    N = 5
    # Set critical frequency
    Wn = 2 * np.pi * 44e3
    # Set maximum ripple
    rp = 0.1
    rs = 80

    cbadc.analog_system.Cauer(N, Wn, rp, rs)


def test_Cauer_transferfunction():
    N = 4
    Wn = 1e3
    rp = np.sqrt(2)
    rs = 60
    cauer_worth_system = cbadc.analog_system.Cauer(N, Wn, rp, rs)

    b, a = scipy.signal.ellip(N, rp, rs, Wn, btype='low', analog=True)
    print("b, a\n")
    print(b)
    print(a)

    w, h = scipy.signal.freqs(b, a)

    tf = cauer_worth_system.transfer_function_matrix(w)

    if not np.allclose(h, tf):
        print(h, tf)
        print(h - tf)
        raise BaseException("Filter mismatch")
