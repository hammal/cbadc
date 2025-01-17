import cbadc
import numpy as np


def test_ds_demo3():
    order = 5
    R = 42
    opt = 1

    H = cbadc.delsig.synthesizeNTF(order, R, opt)
    a, g, b, c = cbadc.delsig.realizeNTF(H)
    b[1:] = 0.0

    print("Unscaled modulator")
    print("DAC feedback coefficients = ", a)
    print("Resonator feedback coefficients = ", g)

    print("Calculate the state maxima")
    ABCD = cbadc.delsig.stuffABCD(a, g, b, c)
    u = np.linspace(0, 0.6, 30)
    N = 10000
    T = np.ones((1, N))
    maxima = np.zeros((order, u.size))
    for i in range(u.size):
        ui = u[i]
        v, xn, xmax, _ = cbadc.delsig.simulateDSM(ui * T, ABCD)
        maxima[:, i] = xmax.flatten()
        if (xmax > 1e2).any():
            umax = ui
            u = u[:i]
            maxima = maxima[:, :i]
            break

    print("The state maxima are: ", maxima)

    print("Calculate the scaled coefficients")

    ABCDs, umax, _ = cbadc.delsig.scaleABCD(ABCD)
    a_s, g_s, b_s, c_s = cbadc.delsig.mapABCD(ABCDs)
    print("Scaled modulator")

    print("DAC feedback coefficients = ", a_s)
    print("Resonator feedback coefficients = ", g_s)
    print("Interstage coefficients", c_s)
    print("Feed-in coefficients", b_s)

    print("Calculate the state maxima")

    u = np.linspace(0, umax, 30)
    for i in range(u.size):
        ui = u[i]
        v, xn, xmax, _ = cbadc.delsig.simulateDSM(ui * T, ABCD)
        maxima[:, i] = xmax.flatten()
        if (xmax > 1e2).any():
            umax = ui
            u = u[:i]
            maxima = maxima[:, :i]
            break
