from numpy.core.numeric import allclose
import cbadc as cb
import numpy as np
import scipy.signal
from cbadc import analog_system


def test_sos2abcd():
    # zero = 123 + 1j * 75
    # pole = 1.2 + 1j * 0.999232
    # z = [zero, np.conjugate(zero)]
    # p = [pole, np.conjugate(pole)]
    # k = np.array([10.1])
    # b, a = scipy.signal.zpk2tf(z, p, k)
    # sos = np.array([[b[-1], b[-2], b[-3], a[-1], a[-2], a[-3]]])

    # A, B, C, D = analog_system.sos2abcd(sos)

    # ana_sys = analog_system.AnalogSystem(A, B, C, None, None, D)
    # z_new, p_new, k_new = ana_sys.zpk()

    # print(z, z_new)
    # print(p, p_new)
    # print(k, k_new)

    # assert(np.allclose(z, z_new))
    # assert(np.allclose(p, p_new))
    # assert(np.allclose(k, k_new))
    pass


def test_zpk2abcd_single_pole():
    no_zero = np.array([])
    zero = np.array([123.54112312331])
    pole = np.array([-4.0])
    poles = np.array([-np.pi, -3, -1.1])
    k = 2.31
    print(f"no_zero = {no_zero}")
    print(f"zero = {zero}")
    print(f"pole = {pole}")
    print(f"poles = {poles}")
    print(f"k = {k}")

    A, B, C, D = analog_system.zpk2abcd(no_zero, pole, k)
    z_new, p_new, k_new = analog_system.AnalogSystem(
        A, B, C, None, None, D).zpk()
    print(z_new, p_new, k_new)
    assert(np.allclose(p_new, pole))
    assert(np.allclose(k_new, k))

    A, B, C, D = analog_system.zpk2abcd(zero, pole, k)
    z_new, p_new, k_new = analog_system.AnalogSystem(
        A, B, C, None, None, D).zpk()
    print(z_new, p_new, k_new)
    assert(np.allclose(p_new, pole))
    assert(np.allclose(z_new, zero))
    assert(np.allclose(k_new, k))

    # A, B, C, D = analog_system.zpk2abcd(no_zero, poles, k)
    # print(A, B, C, D)
    # z_new, p_new, k_new = analog_system.AnalogSystem(
    #     A, B, C, None, None, D).zpk()
    # print(z_new, p_new, k_new)
    # assert(np.allclose(p_new, poles))
    # assert(np.allclose(k_new, k))

    # A, B, C, D = analog_system.zpk2abcd(zero, poles, k)
    # print(A, B, C, D)
    # z_new, p_new, k_new = analog_system.AnalogSystem(
    #     A, B, C, None, None, D).zpk()
    # print(z_new, p_new, k_new)
    # assert(np.allclose(p_new, poles))
    # assert(np.allclose(z_new, zero))
    # assert(np.allclose(k_new, k))


def test_zpk2abcd_double_pole():
    no_zero = np.array([])
    single_zero = np.array([123.54112312331])
    zero_conjugate = 120 + 1j * 3213.
    zero_conjugate = np.array(
        [zero_conjugate, np.conjugate(zero_conjugate)])

    pole_conjugate = 42 + 1j * 42 * 1.67854
    pole_conjugate = np.array(
        [pole_conjugate, np.conjugate(pole_conjugate)])
    double_real_pole = np.array([-123., -93.0])
    k = 2.3133311122
    print(f"no_zero = {no_zero}")
    print(f"single_zero = {single_zero}")
    print(f"zero_conjugate = {zero_conjugate}")
    print(f"pole_conjugate = {pole_conjugate}")
    print(f"double_real_pole = {double_real_pole}")
    print(f"k = {k}")

    A, B, C, D = analog_system.zpk2abcd(no_zero, pole_conjugate, k)
    z_new, p_new, k_new = analog_system.AnalogSystem(
        A, B, C, None, None, D).zpk()
    print(z_new, p_new, k_new)
    assert(np.allclose(p_new, pole_conjugate))
    assert(np.allclose(k_new, k))

    A, B, C, D = analog_system.zpk2abcd(single_zero, pole_conjugate, k)
    z_new, p_new, k_new = analog_system.AnalogSystem(
        A, B, C, None, None, D).zpk()
    print(z_new, p_new, k_new)
    assert(np.allclose(p_new, pole_conjugate))
    assert(np.allclose(z_new, single_zero))
    assert(np.allclose(k_new, k))

    A, B, C, D = analog_system.zpk2abcd(zero_conjugate, pole_conjugate, k)
    z_new, p_new, k_new = analog_system.AnalogSystem(
        A, B, C, None, None, D).zpk()
    print(z_new, p_new, k_new)
    assert(np.allclose(p_new, pole_conjugate))
    assert(np.allclose(z_new, zero_conjugate))
    assert(np.allclose(k_new, k))

    A, B, C, D = analog_system.zpk2abcd(no_zero, double_real_pole, k)
    z_new, p_new, k_new = analog_system.AnalogSystem(
        A, B, C, None, None, D).zpk()
    print(z_new, p_new, k_new)
    assert(np.allclose(p_new, double_real_pole))
    assert(np.allclose(k_new, k))

    A, B, C, D = analog_system.zpk2abcd(single_zero, double_real_pole, k)
    z_new, p_new, k_new = analog_system.AnalogSystem(
        A, B, C, None, None, D).zpk()
    print(z_new, p_new, k_new)
    assert(np.allclose(p_new, double_real_pole))
    assert(np.allclose(z_new, single_zero))
    assert(np.allclose(k_new, k))

    A, B, C, D = analog_system.zpk2abcd(zero_conjugate, double_real_pole, k)
    z_new, p_new, k_new = analog_system.AnalogSystem(
        A, B, C, None, None, D).zpk()
    print(z_new, p_new, k_new)
    assert(np.allclose(p_new, double_real_pole))
    assert(np.allclose(z_new, zero_conjugate))
    assert(np.allclose(k_new, k))


def test_tf2abcd():
    temp = np.arange(9)
    a = temp[:4]
    b = temp[4:]

    print(a, b)

    try:
        cb.analog_system.tf2abcd(temp, temp)
    except BaseException:
        pass

    A, B, CT, D = cb.analog_system.tf2abcd(b, a)

    assert(np.allclose(A, np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [-a[-1], -a[-2], -a[-3], -a[-4]]
    ])))
    assert(np.allclose(B, np.array([
        [0],
        [0],
        [0],
        [1]
    ])))
    assert(np.allclose(CT, np.array([
        [b[-1] - b[0]*a[-1], b[-2] - b[0]*a[-2],
            b[-3] - b[0]*a[-3], b[-4]-b[0]*a[-4]]
    ])))
    assert(np.allclose(D, np.array([
        [b[0]]
    ])))


def test_chain():
    A1 = np.array([[0, -52], [520, 0]])
    B1 = np.array([[1.0], [2.0]])
    C1 = np.array([[0, 1.0]])
    Gamma1 = 31.3 * np.eye(2)
    Gamma_tilde1 = np.eye(2)
    D1 = np.array([[1.2]])
    system1 = cb.analog_system.AnalogSystem(
        A1, B1, C1, Gamma1, Gamma_tilde1, D1)

    A2 = np.array([[0, 0], [1.23, 0]])
    B2 = np.array([[1.0], [0.0]])
    C2 = np.eye(2)
    Gamma2 = None
    Gamma_tilde2 = None
    D2 = np.array([[1.0], [2.0]])
    system2 = cb.analog_system.AnalogSystem(
        A2, B2, C2, Gamma2, Gamma_tilde2, D2)

    BCT2 = np.dot(B2, C1)

    A3 = np.array([[-5.2, 0], [520, 0]])
    B3 = np.array([[1.0, 0.0], [0.0, -3.0]])
    C3 = -np.eye(2)
    Gamma3 = np.arange(4).reshape((2, 2)) * 6260.0
    Gamma_tilde3 = np.eye(2)
    D3 = np.array([[1, 2], [3, 4]])
    system3 = cb.analog_system.AnalogSystem(
        A3, B3, C3, Gamma3, Gamma_tilde3, D3)

    BCT3 = np.dot(B3, C2)
    B3D2C1 = np.dot(B3, np.dot(D2, C1))

    chained_system = cb.analog_system.chain([system1, system2])
    chained_system = cb.analog_system.chain([system1, system2, system3])
    print(chained_system)

    assert(np.allclose(chained_system.A,
                       np.array([
                           [A1[0, 0], A1[0, 1], 0, 0, 0, 0],
                           [A1[1, 0], A1[1, 1], 0, 0, 0, 0],
                           [BCT2[0, 0], BCT2[0, 1], A2[0, 0], A2[0, 1], 0, 0],
                           [BCT2[1, 0], BCT2[1, 1], A2[1, 0], A2[1, 1], 0, 0],
                           [B3D2C1[0, 0], B3D2C1[0, 1], BCT3[0, 0],
                               BCT3[0, 1], A3[0, 0], A3[0, 1]],
                           [B3D2C1[1, 0], B3D2C1[1, 1], BCT3[1, 0],
                               BCT3[1, 1], A3[1, 0], A3[1, 1]]
                       ])))
    B2D1 = np.dot(B2, D1)
    B3D2D1 = np.dot(B3, np.dot(D2, D1))
    assert(np.allclose(chained_system.B,
                       np.array([
                           [B1[0, 0]],
                           [B1[1, 0]],
                           [B2D1[0, 0]],
                           [B2D1[1, 0]],
                           [B3D2D1[0, 0]],
                           [B3D2D1[1, 0]]
                       ])))
    D3C2 = np.dot(D3, C2)
    D3D2C1 = np.dot(D3, np.dot(D2, C1))
    assert(np.allclose(chained_system.CT,
                       np.array([
                           [D3D2C1[0, 0], D3D2C1[0, 1], D3C2[0, 0],
                               D3C2[0, 1], C3[0, 0], C3[0, 1]],
                           [D3D2C1[1, 0], D3D2C1[1, 1], D3C2[1, 0],
                               D3C2[1, 1], C3[1, 0], C3[1, 1]]
                       ])))
    assert(np.allclose(chained_system.Gamma,
                       np.array([
                           [Gamma1[0, 0], Gamma1[0, 1], 0, 0],
                           [Gamma1[1, 0], Gamma1[1, 1], 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, Gamma3[0, 0], Gamma3[0, 1]],
                           [0, 0, Gamma3[1, 0], Gamma3[1, 1]]
                       ])))
    assert(np.allclose(chained_system.Gamma_tildeT,
                       np.array([
                           [Gamma_tilde1[0, 0], Gamma_tilde1[0, 1], 0, 0, 0, 0],
                           [Gamma_tilde1[1, 0], Gamma_tilde1[1, 1], 0, 0, 0, 0],
                           [0, 0, 0, 0, Gamma_tilde3[0, 0], Gamma_tilde3[0, 1]],
                           [0, 0, 0, 0, Gamma_tilde3[1, 0], Gamma_tilde3[1, 1]]
                       ])))
    assert(np.allclose(chained_system.D, np.dot(D3, np.dot(D2, D1))))


def test_stack():
    A1 = np.array([[0, -52], [520, 0]])
    B1 = np.array([[1.0], [2.0]])
    C1 = np.array([[0, 1.0]])
    D1 = np.array([[1.2]])
    Gamma1 = 31.3 * np.eye(2)
    Gamma_tilde1 = np.eye(2)
    system1 = cb.analog_system.AnalogSystem(
        A1, B1, C1, Gamma1, Gamma_tilde1, D1)

    A2 = np.array([[0, 0], [1.23, 0]])
    B2 = np.array([[1.0], [0.0]])
    C2 = np.eye(2)
    Gamma2 = None
    Gamma_tilde2 = None
    D2 = np.array([[1.0], [2.0]])
    system2 = cb.analog_system.AnalogSystem(
        A2, B2, C2, Gamma2, Gamma_tilde2, D2)

    A3 = np.array([[-5.2, 0], [520, 0]])
    B3 = np.array([[1.0, 0.0], [0.0, -3.0]])
    C3 = -np.eye(2)
    D3 = np.array([[1, 2], [3, 4]])
    Gamma3 = np.arange(4).reshape((2, 2)) * 6260.0
    Gamma_tilde3 = np.eye(2)
    system3 = cb.analog_system.AnalogSystem(
        A3, B3, C3, Gamma3, Gamma_tilde3, D3)

    stacked_system = cb.analog_system.stack([system1, system2, system3])
    print(stacked_system)

    assert(np.allclose(stacked_system.A,
                       np.array([
                           [A1[0, 0], A1[0, 1], 0, 0, 0, 0],
                           [A1[1, 0], A1[1, 1], 0, 0, 0, 0],
                           [0, 0, A2[0, 0], A2[0, 1], 0, 0],
                           [0, 0, A2[1, 0], A2[1, 1], 0, 0],
                           [0, 0, 0, 0, A3[0, 0], A3[0, 1]],
                           [0, 0, 0, 0, A3[1, 0], A3[1, 1]]
                       ])))
    assert(np.allclose(stacked_system.B,
                       np.array([
                           [B1[0, 0], 0, 0, 0],
                           [B1[1, 0], 0, 0, 0],
                           [0, B2[0, 0], 0, 0],
                           [0, B2[1, 0], 0, 0],
                           [0, 0, B3[0, 0], B3[0, 1]],
                           [0, 0, B3[1, 0], B3[1, 1]]
                       ])))
    assert(np.allclose(stacked_system.CT,
                       np.array([
                           [C1[0, 0], C1[0, 1], 0, 0, 0, 0],
                           [0, 0, C2[0, 0], C2[0, 1], 0, 0],
                           [0, 0, C2[1, 0], C2[1, 1], 0, 0],
                           [0, 0, 0, 0, C3[0, 0], C3[0, 1]],
                           [0, 0, 0, 0, C3[1, 0], C3[1, 1]]
                       ])))
    assert(np.allclose(stacked_system.Gamma,
                       np.array([
                           [Gamma1[0, 0], Gamma1[0, 1], 0, 0],
                           [Gamma1[1, 0], Gamma1[1, 1], 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, Gamma3[0, 0], Gamma3[0, 1]],
                           [0, 0, Gamma3[1, 0], Gamma3[1, 1]]
                       ])))
    assert(np.allclose(stacked_system.Gamma_tildeT,
                       np.array([
                           [Gamma_tilde1[0, 0], Gamma_tilde1[0, 1], 0, 0, 0, 0],
                           [Gamma_tilde1[1, 0], Gamma_tilde1[1, 1], 0, 0, 0, 0],
                           [0, 0, 0, 0, Gamma_tilde3[0, 0], Gamma_tilde3[0, 1]],
                           [0, 0, 0, 0, Gamma_tilde3[1, 0], Gamma_tilde3[1, 1]]
                       ])))
    assert(np.allclose(stacked_system.D, np.array([
        [D1[0, 0], 0, 0, 0],
        [0, D2[0, 0], 0, 0],
        [0, D2[1, 0], 0, 0],
        [0, 0, D3[0, 0], D3[0, 1]],
        [0, 0, D3[1, 0], D3[1, 1]]
    ])))
