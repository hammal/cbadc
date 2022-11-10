"""Analog systems implemented with op-amps"""
from typing import Dict, List, Tuple
from cbadc.circuit.module import Module, Wire, SubModules
from cbadc.analog_system.analog_system import AnalogSystem
from cbadc.analog_system.topology import chain
from cbadc.circuit.op_amp.resistor_network import ResistorNetwork
from cbadc.circuit.op_amp.amplifier_configurations import (
    InvertingAmplifierCapacitiveFeedback,
)
from cbadc.circuit.op_amp.op_amp import FirstOrderPoleOpAmp, IdealOpAmp
from cbadc.circuit.state_space_equations import StateSpaceLinearSystem
from cbadc.circuit.noise_models import resistor_sizing_voltage_source
import logging
import numpy as np


logger = logging.getLogger(__name__)


class _AnalogSystemOpAmpWithoutIntegrators(Module):

    analog_system: AnalogSystem

    def __init__(
        self,
        analog_system: AnalogSystem,
        **kwargs,
    ) -> None:
        if analog_system.Gamma is None or analog_system.Gamma_tildeT is None:
            raise Exception("both Gammas must be defined.")
        self.analog_system = analog_system
        self._vdd = Wire("vdd", True, False, True, comment="positive supply")
        self._gnd = Wire("vgd", True, False, True, comment="ground")
        self._sgd = Wire("vsgd", True, False, True, comment="signal ground")
        # virtual ground of op_amp
        self._vgd = [
            Wire(f"vgd_{n}", False, False, True, comment='op-amps virtual ground')
            for n in range(analog_system.N)
        ]

        self._u = [
            Wire(f"u_{l}", True, False, True, comment=f"input channel {l}")
            for l in range(analog_system.L)
        ]
        self._s = [
            Wire(f"s_{m}", True, False, True, comment=f"control signal {m}")
            for m in range(analog_system.M)
        ]
        self.inputs = [
            self._vdd,
            self._gnd,
            self._sgd,
            *self._u,
            *self._s,
        ]

        self._x = [
            Wire(f"x_{n}", False, True, True, comment=f"state variable number {n}")
            for n in range(analog_system.N)
        ]

        self._s_tilde = [
            Wire(
                f"s_tilde_{m_tilde}",
                False,
                True,
                True,
                comment=f"control observation {m_tilde}",
            )
            for m_tilde in range(analog_system.M_tilde)
        ]
        self.outputs = [
            *self._s_tilde,
        ]
        ports = [*self.inputs, *self.outputs]
        nets = [*ports, *self._x, *self._vgd]
        super().__init__(
            "analog_system",
            nets,
            ports,
            **kwargs,
        )

    def _module_comment(self) -> List[str]:
        return [
            *super()._module_comment(),
            "",
            "Functional Description",
            "",
            "An analog system enforcing the differential equations.",
            "",
            "ddt(x(t)) = A x(t) + B u(t) + Gamma s(t)",
            "s_tilde(t) = Gamma_tildeT x(t)",
            "",
            "where",
            "",
            f"x(t) = [{', '.join([f'{v.name}' for v in self._x])}]^T",
            f"u(t) = [{', '.join([f'{v.name}' for v in self._u])}]^T",
            f"s(t) = [{', '.join([f'{v.name}' for v in self._s])}]^T",
            f"s_tilde(t) = [{', '.join([f'{v.name}' for v in self._s_tilde])}]^T",
            "",
            "A \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.A[i, :]])}]"
                for i in range(self.analog_system.N)
            ],
            "",
            "B \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.B[i, :]])}]"
                for i in range(self.analog_system.N)
            ],
            "",
            "Gamma \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.Gamma[i, :]])}]"
                for i in range(self.analog_system.N)
            ],
            "",
            "Gamma_tildeT \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.Gamma_tildeT[i, :]])}]"
                for i in range(self.analog_system.M_tilde)
            ],
            "CT \u2248",
            *[
                f"[{', '.join([f'{a:.2e}' for a in self.analog_system.CT[i, :]])}]"
                for i in range(self.analog_system.N_tilde)
            ],
        ]


def _power_or_fixed(
    kwargs: Dict,
) -> Tuple[AnalogSystem, np.ndarray]:
    if "C" in kwargs and "analog_system" in kwargs:
        analog_system = kwargs.pop('analog_system')
        C_diag = np.eye(analog_system.N)
        C = float(kwargs.pop("C"))
        C_diag *= C
    elif "target_snr" in kwargs and "digital_estimator" in kwargs and "BW" in kwargs:

        target_SNR = kwargs.pop("target_snr")
        digital_estimator = kwargs.pop("digital_estimator")
        analog_system = digital_estimator.analog_system
        BW = kwargs.pop("BW")
        C_diag = np.eye(analog_system.N)
        P_Hz = np.max(
            digital_estimator.white_noise_sensitivities(
                np.array(BW), target_SNR, spectrum=True
            ),
            axis=0,
        )
        tau_sum = 1.0 / (
            np.linalg.norm(analog_system.A, axis=1, ord=1)
            + np.linalg.norm(analog_system.B, axis=1, ord=1)
            + np.linalg.norm(analog_system.Gamma, axis=1, ord=1)
        )
        for n, p_Hz in enumerate(P_Hz):
            R_tot = resistor_sizing_voltage_source(p_Hz)
            C_diag[n, n] = tau_sum[n] / R_tot
    else:
        raise Exception("Either the capacitance C or the target_snr must be specified.")
    return analog_system, C_diag


class AnalogSystemIdealOpAmp(_AnalogSystemOpAmpWithoutIntegrators):
    """Analog system implementation using op-amps and RC-networks.

    We model the analog system where the integrators are realized
    using ideal op-amp with capacitive feedback together and interconnections
    are manifested using resistive networks.

    The a single integrator implementation is shown in the figure below.

    .. image:: ../../images/ideal_op_amp_2.svg
        :width: 350
        :align: center
        :alt: Analog system using ideal op-amp with negative capacitive feedback.

    In this configuration the analog state is represented as the voltage
    falling over the capacitor, i.e., :math:`V_{\mathbf{x}_k}(t)`.

    Specifically, for an inverting op-amp configuration with capacitive feedback
    is of the form

    :math:`\dot{V}_{\mathbf{x}_{k}}(t) = -\sum_{\\ell=n}^{N} \\frac{G_{\mathbf{A}_{k,n}}}{C} V_{\mathbf{x}_{n}}(t) - \sum_{\\ell=1}^{L} \\frac{G_{\mathbf{B}_{k,\\ell}}}{C} V_{\mathbf{u}_{\\ell}}(t) - \sum_{m=1}^{M} \\frac{G_{\mathbf{\Gamma}_{k,m}}}{C} V_{\mathbf{s}_{m}}(t)`

    where the capacitance :math:`C` is a specified parameter, and the admittances

    :math:`G_{\mathbf{A}_{k,n}} = - \mathbf{A}_{k, n} \\cdot C`

    :math:`G_{\mathbf{B}_{k,\\ell}} = - \mathbf{B}_{k, \\ell} \\cdot C`

    :math:`G_{\mathbf{\Gamma}_{k,m}} = - \mathbf{\Gamma}_{k, m} \\cdot C`.

    Can the resistor values be negative? Well not exactly. However, for our purposes we assume
    the resulting circuit architecture to be implemented differenitally where negative
    resistors are trivially implemented.

    Additionally, the control observations are realized by a number of
    voltage dividers

    :math:`V_{\\tilde{\mathbf{s}}_\\tilde{m}}(t)= G_{\\tilde{\mathbf{s}}_\\tilde{m}} \\left( \sum_{n=1}^{N} R_{\\tilde{\mathbf{\Gamma}}_{\\tilde{m},n}} V_{\mathbf{x}_n}(t) \\right)`

    where

    :math:`R_{\\tilde{\mathbf{\Gamma}}_{\\tilde{m},n}} = \\frac{1}{\\tilde{\mathbf{\Gamma}}^{\mathsf{T}}_{\\tilde{m},n} C}`

    :math:`G_{\\tilde{\mathbf{s}}_{\\tilde{m}}} = \sum_{n=1}^{N} \\frac{1}{R_{\\tilde{\mathbf{\Gamma}}_{\\tilde{m},n}}}`

    and we have assumed that the digital control has high-impedance inputs and :math:`R_{\\tilde{\mathbf{\Gamma}}_{\\tilde{m},n}}`
    is zero whenever :math:`\\tilde{\mathbf{\Gamma}}^{\mathsf{T}}_{\\tilde{m},n}` is zero.

    Note
    ----
    For the ideal op-amp implementation, the analog-system function is unaltered therefore the
    resulting implementation should, nominally, exactly represent the target analog system.

    Parameters
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`, `optional`
        the ideal analog system specification.
    C: `float`, `optional`
        the capacitance for all integrators.
    BW: `[float, float]`, `optional`
        the bandwidth range [min_BW, max_BW].
    target_snr: `float`, `optional`
        the target SNR.
    digital_estimator: :py:class:`cbdac.digital_estimator.BatchEstimator`
        a digital estimator from which to design
    """

    analog_system: AnalogSystem

    def __init__(
        self,
        **kwargs,
    ) -> None:
        analog_system, self.C_diag = _power_or_fixed(kwargs)
        if analog_system.Gamma is None or analog_system.Gamma_tildeT is None:
            raise Exception("both Gammas must be defined.")
        super().__init__(analog_system, **kwargs)

        # Feedback resistor for summation of s_tilde

        self._A_G_matrix = ResistorNetwork(
            "resistor_network_a",
            "A",
            -np.dot(self.C_diag, analog_system.A),
        )
        self._B_G_matrix = ResistorNetwork(
            "resistor_network_b",
            "B",
            -np.dot(self.C_diag, analog_system.B),
        )
        self._Gamma_G_matrix = ResistorNetwork(
            "resistor_network_gamma",
            "Gamma",
            -np.dot(self.C_diag, analog_system.Gamma),
        )
        self._Gamma_tilde_G_matrix = ResistorNetwork(
            "resistor_network_gamma_tildeT",
            "Gamma_tildeT",
            np.dot(self.C_diag, analog_system.Gamma_tildeT),
        )

        integrators = [
            InvertingAmplifierCapacitiveFeedback(
                f"int_{n}", self.C_diag[n, n], IdealOpAmp
            )
            for n in range(analog_system.N)
        ]
        submodules = [
            *[
                SubModules(
                    integrator,
                    [self._vdd, self._gnd, self._sgd, self._vgd[index], self._x[index]],
                )
                for index, integrator in enumerate(integrators)
            ],
            SubModules(self._A_G_matrix, [*self._x, *self._vgd]),
            SubModules(self._B_G_matrix, [*self._u, *self._vgd]),
            SubModules(self._Gamma_G_matrix, [*self._s, *self._vgd]),
            SubModules(self._Gamma_tilde_G_matrix, [*self._x, *self._s_tilde]),
        ]

        super().__init__(analog_system, submodules=submodules, **kwargs)


# class AnalogSystemFiniteGainOpAmp(_AnalogSystemOpAmpWithoutIntegrators):
#     """Analog system implementation using finite gain op-amps and RC-networks.

#     We model the analog system where the integrators are realized
#     using ideal op-amp with capacitive feedback together and interconnections
#     are manifested using resistive networks.

#     Specifically, the integrators will be as modeled in the figure below.

#     .. image:: ../../images/finite_gain_op_amp_2.svg
#         :width: 350
#         :align: center
#         :alt: Analog system using ideal op-amp with negative capacitive feedback.

#     In this configuration the analog state is represented as the voltage
#     falling over the capacitor, i.e., :math:`V_{\mathbf{x}_k}(t)`.

#     Specifically, the governing differential equation for an inverting op-amp
#     configuration with capacitive feedback and a finite op-amp gain
#     :math:`\\text{A}_{\\text{DC}}` follows as

#     :math:`\\xi \\cdot C \\cdot \dot{V}_{\mathbf{x}_{k}}(t) = -\sum_{\\ell=n}^{N} G_{\mathbf{A}_{k,n}} V_{\mathbf{x}_{n}}(t) - \sum_{\\ell=1}^{L} G_{\mathbf{B}_{k,\\ell}} V_{\mathbf{u}_{\\ell}}(t) - \sum_{m=1}^{M} G_{\mathbf{\Gamma}_{k,m}} V_{\mathbf{s}_{m}}(t) - \\frac{G_{\mathbf{x}_k}}{\\text{A}_{\\text{DC}}} V_{\mathbf{x}_k}(t)`

#     where

#     where the capacitance :math:`C` is a specified parameter, and the resistances

#     :math:`G_{\mathbf{A}_{k,n}} = - \mathbf{A}_{k, n} \\cdot \\xi \\cdot C`

#     :math:`G_{\mathbf{B}_{k,\\ell}} = - \mathbf{B}_{k, \\ell} \\cdot \\xi \\cdot C`

#     :math:`G_{\mathbf{\Gamma}_{k,m}} = - \mathbf{\Gamma}_{k, m} \\cdot \\xi \\cdot C`.

#     Additionally, we see the effect of the finite gain through the terms

#     :math:`\\xi = 1 + 1 / \\text{A}_{\\text{DC}}`

#     :math:`\\frac{G_{\mathbf{x}_k}}{\\text{A}_{\\text{DC}}} = \\frac{\sum_{n=1}^N G_{\mathbf{A}_{k,n}} + \sum_{l=1}^{L} G_{\mathbf{B}_{k,\\ell}} + \sum_{l=m}^{M} G_{\mathbf{\Gamma}_{k,m}}}{\\text{A}_{\\text{DC}}}`

#     This will effectively change the analog system state matrix as

#     :math:`\mathbf{A}_{k, k} \\gets \mathbf{A}_{k, k} + \\frac{G_{\mathbf{x}_k}}{\\xi \\cdot C \\cdot \\text{A}_{\\text{DC}}}`

#     Parameters
#     ----------
#     analog_system: :py:class:`cbadc.analog_system.AnalogSystem`, `optional`
#         the ideal analog system specification.
#     C: `float`, `optional`
#         the capacitance for all integrators.
#     BW: `[float, float]`, `optional`
#         the bandwidth range [min_BW, max_BW].
#     target_snr: `float`, `optional`
#         the target SNR.
#     digital_estimator: :py:class:`cbdac.digital_estimator.BatchEstimator`
#         a digital estimator from which to design
#     A_DC: `float`, `optional`
#         the finite gain, defaults to 1e9.
#     """

#     def __init__(
#         self,
#         **kwargs,
#     ) -> None:
#         analog_system, self.C_diag = _power_or_fixed(kwargs)
#         if analog_system.Gamma is None or analog_system.Gamma_tildeT is None:
#             raise Exception("both Gammas must be defined.")
#         # Modify system to account for finite gain
#         if 'A_DC' not in kwargs:
#             raise NotImplementedError("A_DC must be specified")
#         self.A_DC = kwargs.pop("A_DC", 1e9)
#         super().__init__(analog_system, **kwargs)

#         # xi = 1 + 1 / self.A_DC

#         # No compensation for finite gain
#         x1 = 1.0

#         self._A_G_matrix = ResistorNetwork(
#             "resistor_network_a", "A", -np.dot(self.C_diag, analog_system.A) * xi
#         )
#         self._B_G_matrix = ResistorNetwork(
#             "resistor_network_b",
#             "B",
#             -np.dot(self.C_diag, analog_system.B) * xi,
#         )
#         self._Gamma_G_matrix = ResistorNetwork(
#             "resistor_network_gamma",
#             "Gamma",
#             -np.dot(self.C_diag, analog_system.Gamma) * xi,
#         )
#         self._Gamma_tilde_G_matrix = ResistorNetwork(
#             "resistor_network_gamma_tildeT",
#             "Gamma_tildeT",
#             np.dot(self.C_diag, analog_system.Gamma_tildeT),
#         )

#         G_x = (
#             np.sum(self._A_G_matrix.G, axis=1)
#             + np.sum(self._B_G_matrix.G, axis=1)
#             + np.sum(self._Gamma_G_matrix.G, axis=1)
#         ) / (self.A_DC * xi * np.sum(self.C_diag, axis=1))

#         # Update the analog system.
#         analog_system_new = AnalogSystem(
#             analog_system.A - np.diag(G_x),
#             analog_system.B,
#             analog_system.CT,
#             analog_system.Gamma,
#             analog_system.Gamma_tildeT,
#         )

#         integrators = [
#             InvertingAmplifierCapacitiveFeedback(
#                 f"int_{n}",
#                 self.C_diag[n, n],
#                 FiniteGainOpAmp,
#                 A_DC=self.A_DC,
#             )
#             for n in range(analog_system.N)
#         ]
#         submodules = [
#             *[
#                 SubModules(
#                     integrator,
#                     [self._vdd, self._gnd, self._sgd, self._vgd[index], self._x[index]],
#                 )
#                 for index, integrator in enumerate(integrators)
#             ],
#             SubModules(self._A_G_matrix, [*self._x, *self._vgd]),
#             SubModules(self._B_G_matrix, [*self._u, *self._vgd]),
#             SubModules(self._Gamma_G_matrix, [*self._s, *self._vgd]),
#             SubModules(self._Gamma_tilde_G_matrix, [*self._x, *self._s_tilde]),
#         ]

#         super().__init__(analog_system_new, submodules=submodules, **kwargs)

#     def _module_comment(self) -> List[str]:
#         return [
#             *super()._module_comment(),
#         ]


class AnalogSystemFirstOrderPoleOpAmp(_AnalogSystemOpAmpWithoutIntegrators):
    """

    We model the analog system where the integrators are realized
    using op-amps with capacitive feedback were the internal pole
    is additionally modeled. The interconnections between different
    states are manifested using resistive networks.

    Specifically, the integrators will be as modeled in the figure below.

    .. image:: ../../images/first_order_pole_op_amp_2.svg
        :width: 450
        :align: center
        :alt: The general analog system

    resulting in the two governing differential equations

    :math:`\dot{V}_{\mathbf{x}_{k}}(t) = - \omega_p \\cdot V_{\mathbf{x}_{k}}(t) - \omega_p \\cdot \\text{A}_{\\text{DC}} \\cdot V_{g_k}(t)`

    and

    :math:`C \\cdot \dot{V}_{g_k}(t)=\sum_{\\ell=n}^{N} G_{\mathbf{A}_{k,n}} V_{\mathbf{x}_{n}}(t)+\sum_{\\ell=1}^{L}G_{\mathbf{B}_{k,\\ell}}V_{\mathbf{u}_{\\ell}}(t)+\sum_{m=1}^{M}G_{\mathbf{\Gamma}_{k,m}}V_{\mathbf{s}_{m}}(t)-G_{g_{k}}V_{g_k}(t)+ C \\cdot \dot{V}_{\mathbf{x}_{k}}(t)`

    :math:`=\sum_{\\ell=n}^{N} G_{\mathbf{A}_{k,n}} V_{\mathbf{x}_{n}}(t)+\sum_{\\ell=1}^{L}G_{\mathbf{B}_{k,\\ell}}V_{\mathbf{u}_{\\ell}}(t)+\sum_{m=1}^{M}G_{\mathbf{\Gamma}_{k,m}}V_{\mathbf{s}_{m}}(t)-\\left(G_{g_{k}} + C \\cdot \omega_p \\cdot \\text{A}_{\\text{DC}} \\right)V_{g_k}(t)- C \\cdot \omega_p \\cdot V_{\mathbf{x}_{k}}(t)`

    where

    :math:`\omega_p = \\frac{1}{R_p C_p}`,

    :math:`\\text{A}_{\\text{DC}} = \\text{gm}_k \\cdot R_p`,

    and

    :math:`G_{g_k} = \sum_{n=1}^N G_{\mathbf{A}_{k,n}} + \sum_{l=1}^{L} G_{\mathbf{B}_{k,\\ell}} + \sum_{l=m}^{M} G_{\mathbf{\Gamma}_{k,m}}`.

    To match the ideal analog system specification we follow the steps made in :py:class:`cbadc.circuit_level.op_amp.AnalogSystemFiniteGainOpAmp`.
    Namely, we match the integration slope at DC, i.e., we fix

    :math:`V_{g_k}(t) = - \\frac{1}{\\text{A}_{\\text{DC}}} V_{\mathbf{x}_k}(t)`

    and thereby end up with the equations

    :math:`G_{\mathbf{A}_{k,n}} = - \mathbf{A}_{k, n} \\cdot \\xi \\cdot C`

    :math:`G_{\mathbf{B}_{k,\\ell}} = - \mathbf{B}_{k, \\ell} \\cdot \\xi \\cdot C`

    :math:`G_{\mathbf{\Gamma}_{k,m}} = - \mathbf{\Gamma}_{k, m} \\cdot \\xi \\cdot C`.

    where

    :math:`\\xi = 1 + 1 / \\text{A}_{\\text{DC}}`

    from comparing the resulting expression
    to the ideal analog system differential equations.

    The resulting analog system contains twice as many states as
    we now additionally model the differential equation of the virtual ground.

    Furthermore, we now have extended the states such that

    :math:`\mathbf{x}_{\mathrm{new}}(t) = \\begin{pmatrix} \mathbf{x}_g(t) \\\ \mathbf{x}(t) \\end{pmatrix}`

    where :math:`\mathbf{x}(t)` is the state vector as before and :math:`\mathbf{x}_g(t)` is the states
    corresponding to the virtual ground of the op-amp.

    Parameters
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`, `optional`
        the ideal analog system specification.
    C: `float`, `optional`
        the capacitance for all integrators.
    BW: `[float, float]`, `optional`
        the bandwidth range [min_BW, max_BW].
    target_snr: `float`, `optional`
        the target SNR.
    digital_estimator: :py:class:`cbdac.digital_estimator.BatchEstimator`
        a digital estimator from which to design
    A_DC: `float`, `optional`
        the DC gain of the amplifier, defaults to 1e9.
    omega_p: `float`, `optional`
        the angular pole frequency of the amplifier, defaults to 2 * np.pi * 1e7.
    GBWP: `float`, `optional`
        the gain bandwidth product of the amplifier, defaults to 1e9.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        analog_system, self.C_diag = _power_or_fixed(kwargs)
        if 'A_DC' in kwargs and 'omega_p' in kwargs:
            self.A_DC = kwargs.pop("A_DC", 1e9)
            self.omega_p = kwargs.pop("omega_p", 2 * np.pi * 1e7)
            self.GBWP = self.A_DC * self.omega_p
        elif 'GBWP' in kwargs and 'A_DC' in kwargs:
            self.A_DC = kwargs.pop("A_DC", 1e9)
            self.GBWP = kwargs.pop("GBWP", 2 * np.pi * 1e7)
            self.omega_p = self.GBWP / self.A_DC
        elif 'GBWP' in kwargs and 'omega_p' in kwargs:
            self.omega_p = kwargs.pop("omega_p", 2 * np.pi * 1e7)
            self.GBWP = kwargs.pop("GBWP", 2 * np.pi * 1e7)
            self.A_DC = self.GBWP / self.omega_p
        else:
            raise ValueError(
                "Either A_DC and omega_p or GBWP and omega_p or GBWP and A_DC must be specified."
            )
        G_gnd = (
            np.sum(-analog_system.A, axis=1)
            + np.sum(-analog_system.B, axis=1)
            + np.sum(-analog_system.Gamma, axis=1)
        )

        # Compensation
        # xi = 1 + G_gnd / (self.A_DC * self.omega_p)
        # logger.info(f"gain correction in op-amp: xi={xi}")

        # No compensation
        xi = np.ones(analog_system.N)

        self._A_G_matrix = ResistorNetwork(
            "resistor_network_a",
            "A",
            -np.dot(np.diag(xi), np.dot(self.C_diag, analog_system.A)),
        )
        self._B_G_matrix = ResistorNetwork(
            "resistor_network_b",
            "B",
            -np.dot(np.diag(xi), np.dot(self.C_diag, analog_system.B)),
        )
        self._Gamma_G_matrix = ResistorNetwork(
            "resistor_network_gamma",
            "Gamma",
            -np.dot(np.diag(xi), np.dot(self.C_diag, analog_system.Gamma)),
        )
        self._Gamma_tilde_G_matrix = ResistorNetwork(
            "resistor_network_gamma_tildeT",
            "Gamma_tildeT",
            np.dot(self.C_diag, analog_system.Gamma_tildeT),
        )

        G_gnd = (
            np.sum(self._A_G_matrix.G, axis=1)
            + np.sum(self._B_G_matrix.G, axis=1)
            + np.sum(self._Gamma_G_matrix.G, axis=1)
        )

        # Update the analog system.
        N_new = analog_system.N * 2
        N_old = analog_system.N
        A_new = np.zeros((N_new, N_new))
        B_new = np.zeros((N_new, analog_system.L))
        Gamma_new = np.zeros((N_new, analog_system.M))
        Gamma_tildeT_new = np.zeros((analog_system.M_tilde, N_new))
        # CT_new = np.zeros((analog_system.N_tilde, N_new))
        CT_new = np.eye(N_new)
        A_new[:N_old, :N_old] = -np.eye(
            analog_system.N
        ) * self.omega_p * self.A_DC - np.diag(G_gnd / np.diag(self.C_diag))
        A_new[:N_old, N_old:] = -np.dot(
            np.diag(xi), analog_system.A
        ) - self.omega_p * np.eye(N_old)
        A_new[N_old:, :N_old] = -self.omega_p * self.A_DC * np.eye(N_old)
        A_new[N_old:, N_old:] = -self.omega_p * np.eye(N_old)

        B_new[:N_old, :] = -np.dot(np.diag(xi), analog_system.B)

        Gamma_new[:N_old, :] = -np.dot(np.diag(xi), analog_system.Gamma)

        Gamma_tildeT_new[:, N_old:] = -analog_system.Gamma_tildeT

        # CT_new[:, N_old:] = -analog_system.CT
        CT_new[N_old:, N_old:] = -analog_system.CT

        analog_system_new = AnalogSystem(
            A_new, B_new, CT_new, Gamma_new, Gamma_tildeT_new
        )

        super().__init__(analog_system, **kwargs)

        integrators = [
            InvertingAmplifierCapacitiveFeedback(
                f"int_{n}",
                self.C_diag[n, n],
                FirstOrderPoleOpAmp,
                A_DC=self.A_DC,
                omega_p=self.omega_p,
            )
            for n in range(analog_system.N)
        ]
        submodules = [
            *[
                SubModules(
                    integrator,
                    [self._vdd, self._gnd, self._sgd, self._vgd[index], self._x[index]],
                )
                for index, integrator in enumerate(integrators)
            ],
            SubModules(self._A_G_matrix, [*self._x, *self._vgd]),
            SubModules(self._B_G_matrix, [*self._u, *self._vgd]),
            SubModules(self._Gamma_G_matrix, [*self._s, *self._vgd]),
            SubModules(self._Gamma_tilde_G_matrix, [*self._x, *self._s_tilde]),
        ]

        super().__init__(analog_system, submodules=submodules, **kwargs)
        self.analog_system = analog_system_new

    def _module_comment(self) -> List[str]:
        return [
            *super()._module_comment(),
        ]


# class AnalogSystemHigherOrderOpAmp(_AnalogSystemOpAmpWithoutIntegrators):
#     """

#     We model the analog system where the integrators are realized
#     using op-amps with capacitive feedback were the internals of the
#     op-amp are described with polynomial transfer function.
#     The interconnections between different states are manifested using
#     resistive networks. This is essentially a generic
#     extension of the concepts demonstrated in :py:class:`cbadc.circuit_level.op_amp.AnalogSystemFirstOrderPoleOpAmp`

#     Specifically, the integrators will be as modeled in the figure below.

#     .. image:: ../../images/n_th_order_pole_op_amp_2.svg
#         :width: 500
#         :align: center
#         :alt: The general analog system

#     resulting in the relation

#     :math:`V_{\mathbf{x}_{k}}(s) = T(s) \\cdot V_{g_k}(s)`

#     were :math:`T(s)` is the transfer function of the op-amp and

#     :math:`C \\cdot \dot{V}_{g_k}(t)=\sum_{\\ell=n}^{N} G_{\mathbf{A}_{k,n}} V_{\mathbf{x}_{n}}(t)+\sum_{\\ell=1}^{L}G_{\mathbf{B}_{k,\\ell}}V_{\mathbf{u}_{\\ell}}(t)+\sum_{m=1}^{M}G_{\mathbf{\Gamma}_{k,m}}V_{\mathbf{s}_{m}}(t)-G_{g_{k}}V_{g_k}(t)+ C \\cdot \dot{V}_{\mathbf{x}_{k}}(t)`

#     where

#     :math:`G_{g_k} = \sum_{n=1}^N G_{\mathbf{A}_{k,n}} + \sum_{l=1}^{L} G_{\mathbf{B}_{k,\\ell}} + \sum_{l=m}^{M} G_{\mathbf{\Gamma}_{k,m}}`.

#     To match the ideal analog system specification we follow the steps made in :py:class:`cbadc.circuit_level.op_amp.AnalogSystemFiniteGainOpAmp`.
#     Namely, we match the integration slope at DC, i.e., :math:`s = 0`


#     and thereby end up with the equations

#     :math:`G_{\mathbf{A}_{k,n}} = - \mathbf{A}_{k, n} \\cdot \\xi \\cdot C`

#     :math:`G_{\mathbf{B}_{k,\\ell}} = - \mathbf{B}_{k, \\ell} \\cdot \\xi \\cdot C`

#     :math:`G_{\mathbf{\Gamma}_{k,m}} = - \mathbf{\Gamma}_{k, m} \\cdot \\xi \\cdot C`.

#     where

#     :math:`\\xi = 1 + 1 / T(0)`


#     Parameters
#     ----------
#     analog_system: :py:class:`AnalogSystem`
#         the ideal analog system specification.
#     C: `float`
#         the capacitance of the feedback capacitor.
#     A_DC: `float`
#         the DC gain of the amplifier.
#     amplifiers: List[ :py:class:`cbadc.circuit_level.state_space_equations.StateSpaceLinearSystem ]
#         a list of amplifier models. If the list only contains one element, this system specification will be
#         used for all amplifiers.
#     """

#     def __init__(
#         self,
#         analog_system: AnalogSystem,
#         C: float,
#         amplifiers: List[AnalogSystem],
#         **kwargs,
#     ) -> None:
#         if analog_system.Gamma is None or analog_system.Gamma_tildeT is None:
#             raise Exception("both Gammas must be defined.")

#         if len(amplifiers) == 1:
#             logger.info(
#                 "Only one zpk specified. Will assume all amplifiers are of the same specification."
#             )
#             self.amplifiers = [amplifiers[0] for n in range(analog_system.N)]
#         elif len(amplifiers) != analog_system.N:
#             raise Exception(
#                 "list of amplifier systems must be either a single element (same specification for all) alternatively one specifcation per state, i.e., amplifier."
#             )
#         else:
#             self.amplifiers = amplifiers

#         for amp in self.amplifiers:
#             if amp.L != 1:
#                 raise Exception("The amplifier system must be single input systems.")
#             if amp.N_tilde != 1:
#                 raise Exception("The amplifiers must have precisely a single output")
#             if (amp.Gamma is not None) or (amp.Gamma_tildeT is not None):
#                 raise Exception(
#                     "The amplifiers must be simple systems. No Gamma or Gamma_tildeTs allowed"
#                 )

#         # Compensation
#         # xi = np.array(
#         #     [
#         #         1 - 1 / np.real(amp.transfer_function_matrix(np.array([0])).flatten())
#         #         for amp in self.amplifiers
#         #     ]
#         # ).flatten()
#         # # logger.info(f"gain correction in op-amp: xi={xi}")

#         # No compensation
#         xi = np.ones(analog_system.N)

#         self._A_G_matrix = ResistorNetwork(
#             "resistor_network_a",
#             "A",
#             -np.dot(np.diag(xi), analog_system.A) * C,
#         )
#         self._B_G_matrix = ResistorNetwork(
#             "resistor_network_b",
#             "B",
#             -np.dot(np.diag(xi), analog_system.B) * C,
#         )
#         self._Gamma_G_matrix = ResistorNetwork(
#             "resistor_network_gamma",
#             "Gamma",
#             -np.dot(np.diag(xi), analog_system.Gamma) * C,
#         )
#         self._Gamma_tilde_G_matrix = ResistorNetwork(
#             "resistor_network_gamma_tildeT",
#             "Gamma_tildeT",
#             analog_system.Gamma_tildeT * C,
#         )

#         G_gnd = (
#             np.sum(self._A_G_matrix.G, axis=1)
#             + np.sum(self._B_G_matrix.G, axis=1)
#             + np.sum(self._Gamma_G_matrix.G, axis=1)
#         )

#         new_systems = []
#         N_new = 0

#         for n in range(analog_system.N):
#             A_gnd = np.array([[analog_system.A[n, n] - G_gnd[n] / C]])
#             B_gnd = analog_system.B[n, :].reshape((1, analog_system.L))
#             C_gnd = np.array([[1.0]])
#             Gamma_gnd = analog_system.Gamma[n, :].reshape((1, analog_system.M))
#             Gamma_tildeT_gnd = analog_system.Gamma_tildeT[:, n].reshape(
#                 (analog_system.M_tilde, 1)
#             )
#             local_analog_system = chain(
#                 [
#                     AnalogSystem(A_gnd, B_gnd, C_gnd, Gamma_gnd, Gamma_tildeT_gnd),
#                     self.amplifiers[n],
#                 ]
#             )
#             # include feedback path to vgnd
#             local_analog_system.A[0, 1:] = (
#                 local_analog_system.A[0, 1:] + local_analog_system.A[-1, 1:]
#             )
#             N_new += local_analog_system.N
#             new_systems.append(local_analog_system)
#         A_new = np.zeros((N_new, N_new))
#         B_new = np.zeros((N_new, analog_system.L))
#         Gamma_new = np.zeros((N_new, analog_system.M))
#         Gamma_tildeT_new = np.zeros((analog_system.M_tilde, N_new))
#         CT_new = np.zeros((analog_system.N_tilde, N_new))
#         for index, sys in enumerate(new_systems):
#             A_new[
#                 index * sys.N : (index + 1) * sys.N, index * sys.N : (index + 1) * sys.N
#             ] = sys.A
#             for n in range(analog_system.N):
#                 if n != index:
#                     other_system = new_systems[index]
#                     input_vector = np.zeros(sys.N)
#                     input_vector[0] = 1.0
#                     output_vector = np.zeros(other_system.N)
#                     output_vector[-1] = 1.0
#                     outer_map_matrix = np.outer(input_vector, output_vector)
#                     A_new[
#                         index * sys.N : (index + 1) * sys.N,
#                         n * other_system.N : (n + 1) * other_system.N,
#                     ] = (
#                         outer_map_matrix * analog_system.A[index, n]
#                     )
#             B_new[index * sys.N : (index + 1) * sys.N, :] = sys.B
#             Gamma_new[index * sys.N : (index + 1) * sys.N, :] = sys.Gamma
#             Gamma_tildeT_new[:, index * sys.N : (index + 1) * sys.N] = -sys.Gamma_tildeT
#             CT_new[:, index * sys.N : (index + 1) * sys.N] = -sys.CT
#         analog_system_new = AnalogSystem(
#             A_new, B_new, CT_new, Gamma_new, Gamma_tildeT_new
#         )

#         super().__init__(analog_system, **kwargs)

#         integrators = [
#             InvertingAmplifierCapacitiveFeedback(
#                 f"int_{n}",
#                 C,
#                 StateSpaceLinearSystem,
#                 amplifier=self.amplifiers[n],
#             )
#             for n in range(analog_system.N)
#         ]
#         submodules = [
#             *[
#                 SubModules(
#                     integrator,
#                     [self._vdd, self._gnd, self._sgd, self._vgd[index], self._x[index]],
#                 )
#                 for index, integrator in enumerate(integrators)
#             ],
#             SubModules(self._A_G_matrix, [*self._x, *self._vgd]),
#             SubModules(self._B_G_matrix, [*self._u, *self._vgd]),
#             SubModules(self._Gamma_G_matrix, [*self._s, *self._vgd]),
#             SubModules(self._Gamma_tilde_G_matrix, [*self._x, *self._s_tilde]),
#         ]

#         super().__init__(analog_system, submodules=submodules, **kwargs)
#         self.analog_system = analog_system_new

#     def _module_comment(self) -> List[str]:
#         return [
#             *super()._module_comment(),
#         ]
