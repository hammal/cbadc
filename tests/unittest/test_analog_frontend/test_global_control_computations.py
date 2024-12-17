from cbadc.analog_frontend import get_global_control, _analog_filter_matrix_exponential
from cbadc.synthesis.leap_frog import get_leap_frog
import numpy as np
import scipy.integrate
import scipy.linalg

N = 4
ENOB = 15
BW = 1e6

analog_frontend = get_leap_frog(N=N, ENOB=ENOB, BW=BW, xi=1e-2)


def test_exponential_integral():
    def homogeneous_exponential_integral(t, x):
        return np.dot(analog_frontend.analog_filter.A, x)

    x_sol = np.zeros((N, N))

    for n in range(N):
        initital_state = np.zeros(N)
        initital_state[n] = 1.0

        sol = scipy.integrate.solve_ivp(
            homogeneous_exponential_integral,
            (0, analog_frontend.digital_control.clock.T),
            initital_state,
        )
        x_sol[:, n] = sol.y[:, -1]

    direct_solution = _analog_filter_matrix_exponential(
        analog_frontend.analog_filter.A, analog_frontend.digital_control.clock.T
    )
    print(x_sol)
    assert np.allclose(x_sol, direct_solution)


def test_magnitude_of_control():
    x_sol = np.zeros((N, N))

    for m in range(N):

        def derivative(t, x):
            return np.dot(analog_frontend.analog_filter.A, x) + np.dot(
                analog_frontend.analog_filter.Gamma,
                analog_frontend.digital_control.impulse_response(m, t),
            )

        sol = scipy.integrate.solve_ivp(
            derivative, (0, analog_frontend.digital_control.clock.T), np.zeros(N)
        )

        x_sol[:, m] = sol.y[:, -1]
    print(x_sol)
    assert True


def test_minimum_energy_control():
    def controlability_gramian(t, x):
        exp_a = _analog_filter_matrix_exponential(analog_frontend.analog_filter.A, t)
        exp_a_b = np.dot(exp_a, analog_frontend.analog_filter.B)
        return np.dot(exp_a_b, exp_a_b.T).flatten()

    CA = (
        scipy.integrate.solve_ivp(
            controlability_gramian,
            (0, analog_frontend.digital_control.clock.T),
            np.zeros(N * N),
        )
        .y[:, -1]
        .reshape((N, N))
    )

    exp_A = _analog_filter_matrix_exponential(
        analog_frontend.analog_filter.A, analog_frontend.digital_control.clock.T
    )
    exp_A_Gamma_transpose = np.dot(
        exp_A, analog_frontend.analog_filter.Gamma
    ).transpose()

    Gamma = -np.dot(np.dot(exp_A_Gamma_transpose, np.linalg.inv(CA)), exp_A)

    print(Gamma)
    assert True


def test_gramian():
    T = 1 / (2 * BW) / 16

    def controlability_gramian(t, x):
        exp_a = _analog_filter_matrix_exponential(analog_frontend.analog_filter.A, t)
        exp_a_b = np.dot(exp_a, analog_frontend.analog_filter.B)
        return np.dot(exp_a_b, exp_a_b.T).flatten()

    CA = (
        scipy.integrate.solve_ivp(
            controlability_gramian,
            (0, T),
            np.zeros(N * N),
        )
        .y[:, -1]
        .reshape((N, N))
    )
    exp_A = np.zeros((N, N))
    for n in range(N):
        gamma_vec = np.zeros(N)
        gamma_vec[n] = 1.0

        def derivative(t, x):
            return np.dot(analog_frontend.analog_filter.A, x) + gamma_vec

        exp_A[:, n] = scipy.integrate.solve_ivp(derivative, (0, T), np.zeros(N)).y[
            :, -1
        ]
    # exp_A = _analog_filter_matrix_exponential(analog_frontend.analog_filter.A, T)
    L = np.linalg.cholesky(CA)
    # sol = np.dot(L, np.linalg.inv(exp_A))
    sol = np.linalg.lstsq(exp_A, L, rcond=None)[0]

    print(CA)
    print(L)
    print(sol)
    assert True


def test_linspace_sin_cos():
    size = 1000
    omega = 2 * np.pi * np.linspace(BW / size, BW, size)
    T = 1 / (2 * BW) / 16

    Y = np.zeros((N, len(omega)))
    for i, w in enumerate(omega):

        def derivative(t, x):
            return np.dot(
                analog_frontend.analog_filter.A, x
            ) + analog_frontend.analog_filter.B.flatten() * np.sin(w * t)

        Y[:, i] = scipy.integrate.solve_ivp(
            derivative,
            (0, T),
            np.zeros(N),
        ).y[:, -1]

    print(Y)

    A = _analog_filter_matrix_exponential(analog_frontend.analog_filter.A, T)

    sol = np.linalg.lstsq(
        np.vstack([A for _ in range(len(omega))]), Y.flatten(), rcond=None
    )[0]
    print(f"T = {T}, diag(Gamma) = \n{sol}")
    assert True
