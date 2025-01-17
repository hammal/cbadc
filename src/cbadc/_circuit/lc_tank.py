from ..analog_frontend import AnalogFrontend
from ..analog_filter import AnalogSystem
from ..digital_control import DigitalControl
from ..analog_signal import Clock
from .analog_frontend import CircuitAnalogFrontend
from ..analog_frontend import get_global_control, _analog_filter_matrix_exponential
from .components.passives import Resistor, Capacitor, Inductor
from .components.opamp import OTA
import numpy as np
import scipy.integrate


def _non_homogenious_weights(
    analog_frontend: AnalogFrontend, input_function
) -> np.ndarray:
    analog_filter = analog_frontend.analog_filter
    digital_control = analog_frontend.digital_control

    def input_derivative(t: float, x: np.ndarray):
        return np.dot(analog_filter.A, x) + analog_filter.B.flatten() * input_function(
            t
        )

    y = scipy.integrate.solve_ivp(
        input_derivative,
        (0, digital_control.clock.T),
        np.zeros(analog_filter.N),
    ).y[:, -1]

    A = np.zeros((analog_filter.N, analog_filter.M))

    for m in range(analog_filter.M):
        identity_vector = np.zeros(analog_filter.N)
        identity_vector[m] = 1.0

        def control_derivative(t: float, x: np.ndarray):
            return np.dot(analog_filter.A, x) + identity_vector

        A[:, m] = scipy.integrate.solve_ivp(
            control_derivative,
            (0, digital_control.clock.T),
            np.zeros(analog_filter.N),
        ).y[:, -1]

    return np.linalg.lstsq(A, y)[0]


def _non_homogenious_spectrum_weights(analog_frontend: AnalogFrontend, BW: float):
    size = 1000
    omega = 2 * np.pi * np.linspace(0, BW, size)
    Y = np.zeros((analog_frontend.analog_filter.N, len(omega)))
    for i, w in enumerate(omega):

        def derivative(t, x):
            return np.dot(
                analog_frontend.analog_filter.A, x
            ) + analog_frontend.analog_filter.B.flatten() * np.cos(w * t)

        Y[:, i] = scipy.integrate.solve_ivp(
            derivative,
            (0, analog_frontend.digital_control.clock.T),
            np.zeros(analog_frontend.analog_filter.N),
        ).y[:, -1]

    A = np.zeros((analog_frontend.analog_filter.N, analog_frontend.analog_filter.M))

    for m in range(analog_frontend.analog_filter.M):
        identity_vector = np.zeros(analog_frontend.analog_filter.N)
        identity_vector[m] = 1.0

        def control_derivative(t: float, x: np.ndarray):
            return np.dot(analog_frontend.analog_filter.A, x) + identity_vector

        A[:, m] = scipy.integrate.solve_ivp(
            control_derivative,
            (0, analog_frontend.digital_control.clock.T),
            np.zeros(analog_frontend.analog_filter.N),
        ).y[:, -1]

    return np.linalg.lstsq(
        np.vstack([A for _ in range(len(omega))]), Y.flatten(), rcond=None
    )[0]


def _non_homogenious_initial_state_weights(analog_frontend: AnalogFrontend):
    A = np.zeros((analog_frontend.analog_filter.N, analog_frontend.analog_filter.M))

    for m in range(analog_frontend.analog_filter.M):
        identity_vector = np.zeros(analog_frontend.analog_filter.N)
        identity_vector[m] = 1.0

        def control_derivative(t: float, x: np.ndarray):
            return np.dot(analog_frontend.analog_filter.A, x) + identity_vector

        A[:, m] = scipy.integrate.solve_ivp(
            control_derivative,
            (0, analog_frontend.digital_control.clock.T),
            np.zeros(analog_frontend.analog_filter.N),
        ).y[:, -1]

    x0 = np.array([0.5 / (1 << n) for n in np.arange(analog_frontend.analog_filter.N)])
    y = np.dot(
        _analog_filter_matrix_exponential(
            analog_frontend.analog_filter.A, analog_frontend.digital_control.clock.T
        ),
        x0,
    )

    return np.linalg.lstsq(A, y, rcond=None)[0]


def _exponential_decay(analog_frontend: AnalogFrontend, input_function) -> np.ndarray:
    analog_filter = analog_frontend.analog_filter
    digital_control = analog_frontend.digital_control

    def input_derivative(t: float, x: np.ndarray):
        return np.dot(analog_filter.A, x) + analog_filter.B.flatten() * input_function(
            t
        )

    y = scipy.integrate.solve_ivp(
        input_derivative,
        (0, digital_control.clock.T),
        np.zeros(analog_filter.N),
    ).y[:, -1]

    A = np.zeros((analog_filter.N, analog_filter.M))

    for m in range(analog_filter.M):
        identity_vector = np.zeros(analog_filter.N)
        identity_vector[m] = 1.0

        def control_derivative(t: float, x: np.ndarray):
            return np.dot(analog_filter.A, x) + identity_vector

        A[:, m] = scipy.integrate.solve_ivp(
            control_derivative,
            (0, digital_control.clock.T),
            np.zeros(analog_filter.N),
        ).y[:, -1]

    return np.array(
        [np.inner(y, A[:, m]) / np.inner(A[:, m], A[:, m]) for m in range(A.shape[1])]
    )


class LCFrontend(CircuitAnalogFrontend):
    L: float
    C: float
    Rs: float

    def __init__(
        self,
        M: int,
        L: float,
        C: float,
        gm: float,
        fc: float,
        Rin: float = 1e1,
        vdd_voltage: float = 1.2,
        in_high=0.0,
        in_low=0.0,
    ):
        kappa_scale: np.ndarray = np.array([gm * (0.5**m) for m in range(M)])

        N = 2 * M - 1
        A = np.zeros((N, N))
        B = np.zeros((N, 1))
        Gamma = np.zeros((N, M))

        # The local feedback
        # A[:M, :M] = -np.eye(M) / (R * C)
        A[0, 0] = -1 / (Rin * C)

        # The resonance elements
        A[M:, :M] = (np.eye(M) - np.eye(M, k=1))[: (N - M), :] / (2 * L)
        A[:M, M:] = (-np.eye(M) + np.eye(M, k=-1))[:, : (N - M)] / C

        B[0, 0] = 1 / (Rin * C)

        Gamma[:M, :M] = np.eye(M) / C
        for m in range(M):
            Gamma[m, m] *= kappa_scale[m]

        # experimental
        def step(t: float):
            return 1.0 if t > 0 else 0.0

        self.extended_analog_frontend = AnalogFrontend(
            AnalogSystem(A, B, np.eye(N), Gamma, -np.eye(N)),
            DigitalControl(Clock(1 / fc), M),
        )

        # Gamma[:M, :M] = np.diag(
        #     _non_homogenious_weights(self.extended_analog_frontend, step) * 2
        # )

        # BW = 1e8
        # Gamma[:M, :M] = np.diag(_non_homogenious_spectrum_weights(self.extended_analog_frontend, BW))
        # Gamma[:M, :M] = np.diag(_non_homogenious_weights(self.extended_analog_frontend, step))[0,0]

        # Gamma[:M, :M] = np.diag(_non_homogenious_initial_state_weights(self.extended_analog_frontend))
        Gamma[:M, :M] = np.diag(_exponential_decay(self.extended_analog_frontend, step))

        self.extended_analog_frontend = AnalogFrontend(
            AnalogSystem(A, B, np.eye(N), Gamma, -Gamma.transpose()),
            DigitalControl(Clock(1 / fc), M),
        )

        self.extended_analog_frontend = get_global_control(
            AnalogSystem(A, B, np.eye(N), Gamma, -Gamma.transpose()),
            DigitalControl(Clock(1 / fc), M),
            np.linspace(0, 1 / fc, M, endpoint=False),
            # atol=1e-6,
            # rtol=1e-5,
        )

        super().__init__(
            AnalogFrontend(
                AnalogSystem(
                    self.extended_analog_frontend.analog_filter.A[:M, :M],
                    self.extended_analog_frontend.analog_filter.B[:M, :],
                    np.eye(M),
                    self.extended_analog_frontend.analog_filter.Gamma[:M, :M],
                    self.extended_analog_frontend.analog_filter.Gamma_tildeT[:M, :M],
                    B_tilde=self.extended_analog_frontend.analog_filter.B_tilde,
                    A_tilde=self.extended_analog_frontend.analog_filter.A_tilde,
                ),
                self.extended_analog_frontend.digital_control,
            ),
            vdd_voltage,
            in_high,
            in_low,
            subckt_name="LC_frontend",
            instance_name="Xaf",
        )

        inductors = [Inductor(f"Lp_{i}", L) for i in range(M - 1)] + [
            Inductor(f"Ln_{i}", L) for i in range(M - 1)
        ]
        capacitors = [Capacitor(f"C_{i}", C) for i in range(M)]
        resistors = [
            Resistor("Rin_p", Rin),
            Resistor("Rin_n", Rin),
        ]

        otas = [
            OTA(
                f"gm{m}_p",
                "ota",
                np.abs(self.extended_analog_frontend.analog_filter.Gamma[m, m] * C / 2),
            )
            for m in range(M)
        ] + [
            OTA(
                f"gm{m}_n",
                "ota",
                np.abs(self.extended_analog_frontend.analog_filter.Gamma[m, m] * C / 2),
            )
            for m in range(M)
        ]

        self.add(*inductors)
        self.add(*capacitors)
        self.add(*resistors)
        self.add(*otas)

        for m in range(M):
            self.connects(
                (self["VDD"], otas[m]["VDD"]),
                (self["VSS"], otas[m]["VSS"]),
                (self["VDD"], otas[m]["OUT_P"]),
                (self.xp[m], otas[m]["OUT_N"]),
                (self[f"OUT{m}_P"], otas[m]["IN_P"]),
                (self[f"OUT{m}_N"], otas[m]["IN_N"]),
            )

            self.connects(
                (self["VDD"], otas[m + M]["VDD"]),
                (self["VSS"], otas[m + M]["VSS"]),
                (self["VSS"], otas[m + M]["OUT_N"]),
                (self.xn[m], otas[m + M]["OUT_P"]),
                (self[f"OUT{m}_P"], otas[m + M]["IN_P"]),
                (self[f"OUT{m}_N"], otas[m + M]["IN_N"]),
            )

            self.connects(
                (self.xp[m], capacitors[m][0]),
                (self.xn[m], capacitors[m][1]),
            )

            # Input
            if m == 0:
                self.connects(
                    (self.xp[m], resistors[0][1]),
                    (self.xn[m], resistors[1][1]),
                    (self["IN0_P"], resistors[0][0]),
                    (self["IN0_N"], resistors[1][0]),
                )
            # controls
            else:
                self.connects(
                    (self.xp[m - 1], inductors[m - 1][0]),
                    (self.xp[m], inductors[m - 1][1]),
                    (self.xn[m - 1], inductors[M + m - 2][0]),
                    (self.xn[m], inductors[M + m - 2][1]),
                )
