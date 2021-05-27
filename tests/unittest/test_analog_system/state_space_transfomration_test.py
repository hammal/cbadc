import cbadc as cb
import numpy as np


def test_abcd2abc():
    A = np.array([[1.0, 2.0], [3.0, 3.0]])
    B = np.array([[5.0], [0.0]])
    C = np.array([[0.0, 23.0]])
    D = np.array([[np.pi]])

    A_tilde, B_tilde, C_tilde = cb.analog_system.abcd2abc(A, B, C, D)
    print(A_tilde, B_tilde, C_tilde)

    assert(np.allclose(A_tilde,
                       np.array(
                           [
                               [0, 0, 0],
                               [B[0, 0], A[0, 0], A[0, 1]],
                               [B[1, 0], A[1, 0], A[1, 1]]
                           ])))
    assert(np.allclose(B_tilde,
                       np.array(
                           [
                               [1],
                               [0],
                               [0]
                           ])))
    assert(np.allclose(C_tilde,
                       np.array(
                           [
                               [D[0, 0], C[0, 0], C[0, 1]]
                           ])))


def test_chain():
    A1 = np.array([[0, -52], [520, 0]])
    B1 = np.array([[1.0], [2.0]])
    C1 = np.array([[0, 1.0]])
    Gamma1 = 31.3 * np.eye(2)
    Gamma_tilde1 = np.eye(2)
    system1 = cb.analog_system.AnalogSystem(A1, B1, C1, Gamma1, Gamma_tilde1)

    A2 = np.array([[0, 0], [1.23, 0]])
    B2 = np.array([[1.0], [0.0]])
    C2 = np.eye(2)
    Gamma2 = None
    Gamma_tilde2 = None
    system2 = cb.analog_system.AnalogSystem(A2, B2, C2, Gamma2, Gamma_tilde2)

    BCT2 = np.dot(B2, C1)

    A3 = np.array([[-5.2, 0], [520, 0]])
    B3 = np.array([[1.0, 0.0], [0.0, -3.0]])
    C3 = -np.eye(2)
    Gamma3 = np.arange(4).reshape((2, 2)) * 6260.0
    Gamma_tilde3 = np.eye(2)
    system3 = cb.analog_system.AnalogSystem(A3, B3, C3, Gamma3, Gamma_tilde3)

    BCT3 = np.dot(B3, C2)

    chained_system = cb.analog_system.chain([system1, system2, system3])
    print(chained_system)

    assert(np.allclose(chained_system.A,
                       np.array([
                           [A1[0, 0], A1[0, 1], 0, 0, 0, 0],
                           [A1[1, 0], A1[1, 1], 0, 0, 0, 0],
                           [BCT2[0, 0], BCT2[0, 1], A2[0, 0], A2[0, 1], 0, 0],
                           [BCT2[1, 0], BCT2[1, 1], A2[1, 0], A2[1, 1], 0, 0],
                           [0, 0, BCT3[0, 0], BCT3[0, 1], A3[0, 0], A3[0, 1]],
                           [0, 0, BCT3[1, 0], BCT3[1, 1], A3[1, 0], A3[1, 1]]
                       ])))
    assert(np.allclose(chained_system.B,
                       np.array([
                           [B1[0, 0]],
                           [B1[1, 0]],
                           [0],
                           [0],
                           [0],
                           [0]
                       ])))
    assert(np.allclose(chained_system.CT,
                       np.array([
                           [0, 0, 0, 0, C3[0, 0], C3[0, 1]],
                           [0, 0, 0, 0, C3[1, 0], C3[1, 1]]
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


def test_stack():
    A1 = np.array([[0, -52], [520, 0]])
    B1 = np.array([[1.0], [2.0]])
    C1 = np.array([[0, 1.0]])
    Gamma1 = 31.3 * np.eye(2)
    Gamma_tilde1 = np.eye(2)
    system1 = cb.analog_system.AnalogSystem(A1, B1, C1, Gamma1, Gamma_tilde1)

    A2 = np.array([[0, 0], [1.23, 0]])
    B2 = np.array([[1.0], [0.0]])
    C2 = np.eye(2)
    Gamma2 = None
    Gamma_tilde2 = None
    system2 = cb.analog_system.AnalogSystem(A2, B2, C2, Gamma2, Gamma_tilde2)

    A3 = np.array([[-5.2, 0], [520, 0]])
    B3 = np.array([[1.0, 0.0], [0.0, -3.0]])
    C3 = -np.eye(2)
    Gamma3 = np.arange(4).reshape((2, 2)) * 6260.0
    Gamma_tilde3 = np.eye(2)
    system3 = cb.analog_system.AnalogSystem(A3, B3, C3, Gamma3, Gamma_tilde3)

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
