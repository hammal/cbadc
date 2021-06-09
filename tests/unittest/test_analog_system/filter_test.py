import numpy as np
import cbadc
import scipy.signal

from cbadc.analog_system import zpk2abcd


def test_Butterworth():
    N = 2
    # Set critical frequency
    Wn = 1.0

    cbadc.analog_system.ButterWorth(N, Wn)


def test_ButterWorth_transferfunction():
    N = 7
    Wn = 1e3
    butter_worth_system = cbadc.analog_system.ButterWorth(N, Wn)
    print(butter_worth_system)
    b, a = scipy.signal.butter(N, Wn, btype='low', analog=True)
    print("b, a\n")
    print(b)
    print(a)

    z, p, k = scipy.signal.butter(
        N, Wn, btype='low', output="zpk", analog=True)

    print("z,p,k")
    print(z)
    print(np.sort_complex(p))
    print(k)

    print(zpk2abcd(z, p, k))

    w, h = scipy.signal.freqs(b, a)

    tf = butter_worth_system.transfer_function_matrix(w)[:, 0, :].flatten()

    if not np.allclose(h, tf):
        # print(h, tf)
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

    z, p, k = scipy.signal.cheby1(
        N, rp, Wn, btype='low', analog=True, output="zpk")
    print("z,p,k")
    print(z)
    print(np.sort_complex(p))
    print(k)

    w, h = scipy.signal.freqs(b, a)

    tf = chebyshevI_worth_system.transfer_function_matrix(w)

    if not np.allclose(h, tf):
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
    N = 15
    Wn = 1e3
    rs = 60
    chebyshevII_worth_system = cbadc.analog_system.ChebyshevII(N, Wn, rs)

    b, a = scipy.signal.cheby2(N, rs, Wn, btype='low', analog=True)
    print("b, a\n")
    print(b)
    print(a)

    z, p, k = scipy.signal.cheby2(
        N, rs, Wn, btype='low', analog=True, output='zpk')
    print("z,p,k")
    print(z)
    print(np.sort_complex(p))
    print(k)

    ss_z, ss_p, ss_k = chebyshevII_worth_system.zpk()
    print("State Space model as zpk")
    print(ss_z)
    print(ss_p)
    print(ss_k)

    w, h = scipy.signal.freqs(b, a)

    tf = chebyshevII_worth_system.transfer_function_matrix(w)
    print(chebyshevII_worth_system)
    if not np.allclose(h, tf):
        print(h - tf)
        raise BaseException("Filter mismatch")


def test_Cauer():
    N = 4
    # Set critical frequency
    Wn = 2 * np.pi * 44e3
    # Set maximum ripple
    rp = 0.1
    rs = 80

    cbadc.analog_system.Cauer(N, Wn, rp, rs)


def test_Cauer_transferfunction():
    N = 11
    Wn = 1e3
    rp = np.sqrt(2)
    rs = 60
    cauer_worth_system = cbadc.analog_system.Cauer(N, Wn, rp, rs)

    b, a = scipy.signal.ellip(N, rp, rs, Wn, btype='low', analog=True)
    print("b, a\n")
    print(b)
    print(a)
    print(cauer_worth_system)
    w, h = scipy.signal.freqs(b, a)

    z, p, k = scipy.signal.ellip(
        N, rp, rs, Wn, btype='low', analog=True, output='zpk')
    print("z,p,k")
    print(z)
    print(np.sort_complex(p))
    print(k)

    tf = cauer_worth_system.transfer_function_matrix(w)
    print(cauer_worth_system)
    if not np.allclose(h, tf):
        print(h - tf)
        raise BaseException("Filter mismatch")
