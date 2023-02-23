from . import Terminal, SubCircuitElement, Ground
from .components.integrator import Integrator
from .components.summer import Summer
from ..analog_frontend import AnalogFrontend
from ..analog_system import AnalogSystem
from ..digital_control import DigitalControl as NominalDigitalControl
from ..digital_control import DitherControl as NominalDitherControl
from .digital_control import DigitalControl, DitherControl
from .testbench import CircuitAnalogFrontend

# class IntegratorAnalogSystem(SubCircuitElement):
#     def __init__(self, analog_system: AnalogSystem, vdd: float, vgnd: float):
#         # bound_half = vdd - vgnd  # / 2
#         # vsgnd = bound_half / 2 + vgnd
#         self.nominal_analog_system = analog_system
#         # self.nominal_analog_system.A = self.nominal_analog_system.A / bound_half
#         # self.nominal_analog_system.B = self.nominal_analog_system.B / bound_half
#         # self.nominal_analog_system.Gamma = self.nominal_analog_system.Gamma / bound_half

#         super().__init__(
#             [Terminal('vgnd')]
#             + [Terminal(f'U{index}') for index in range(self.nominal_analog_system.L)]
#             + [Terminal(f'S{index}') for index in range(self.nominal_analog_system.M)]
#             + [
#                 Terminal(f'STILDE{index}')
#                 for index in range(self.nominal_analog_system.M_tilde)
#             ]
#             + [Terminal(f'X{index}') for index in range(self.nominal_analog_system.N)],
#             subckt_name='analog_system',
#             instance_name='AS',
#         )

#         self._integrators = [
#             Integrator(
#                 'int',
#                 input_offset=-vgnd,
#                 out_initial_condition=vgnd,
#                 out_lower_limit=-vdd,
#                 out_upper_limit=2 * vdd,
#             )
#             for index in range(self.nominal_analog_system.N)
#         ]
#         self._pre_sum = [
#             Summer(
#                 f'sum_pre_int{index}',
#                 number_of_inputs=self.nominal_analog_system.N
#                 + self.nominal_analog_system.L
#                 + self.nominal_analog_system.M,
#                 input_offset=[
#                     0.0
#                     for _ in range(
#                         self.nominal_analog_system.N
#                         + self.nominal_analog_system.L
#                         + self.nominal_analog_system.M
#                     )
#                 ],
#                 input_gain=[a for a in self.nominal_analog_system.A[index, :]]
#                 + [b for b in self.nominal_analog_system.B[index, :]]
#                 + [gamma for gamma in self.nominal_analog_system.Gamma[index, :]],
#                 output_offset=vgnd,
#             )
#             for index in range(self.nominal_analog_system.N)
#         ]
#         self._post_sum = [
#             Summer(
#                 f'sum_post_int{index}',
#                 number_of_inputs=self.nominal_analog_system.N
#                 + self.nominal_analog_system.L
#                 + self.nominal_analog_system.M,
#                 input_offset=[
#                     0.0
#                     for _ in range(
#                         self.nominal_analog_system.N
#                         + self.nominal_analog_system.L
#                         + self.nominal_analog_system.M
#                     )
#                 ],
#                 input_gain=[
#                     gamma for gamma in self.nominal_analog_system.Gamma_tildeT[index, :]
#                 ]
#                 + [b for b in self.nominal_analog_system.B_tilde[index, :]]
#                 + [a for a in self.nominal_analog_system.A_tilde[index, :]],
#                 output_offset=vgnd,
#             )
#             for index in range(self.nominal_analog_system.M_tilde)
#         ]

#         self.add(*self._integrators)
#         self.add(*self._pre_sum)
#         self.add(*self._post_sum)

#         # Connect vgnd
#         for pre_sum in self._pre_sum:
#             self.connect(self.terminals[0], pre_sum._terminals[-1])
#         for post_sum in self._post_sum:
#             self.connect(self.terminals[0], post_sum._terminals[-1])

#         # Connect input to pre sum
#         for index in range(self.nominal_analog_system.N):
#             # connect states
#             for n in range(self.nominal_analog_system.N):
#                 self.connect(
#                     self.terminals[
#                         1
#                         + self.nominal_analog_system.L
#                         + self.nominal_analog_system.M
#                         + self.nominal_analog_system.M_tilde
#                         + n
#                     ],
#                     self._pre_sum[index]._terminals[n],
#                 )
#             # Connect the input
#             for l in range(self.nominal_analog_system.L):
#                 self.connect(
#                     self.terminals[1 + l],
#                     self._pre_sum[index]._terminals[self.nominal_analog_system.N + l],
#                 )
#             # Connect the control signals
#             for m in range(self.nominal_analog_system.M):
#                 self.connect(
#                     self.terminals[1 + m + self.nominal_analog_system.L],
#                     self._pre_sum[index]._terminals[
#                         self.nominal_analog_system.N + self.nominal_analog_system.L + m
#                     ],
#                 )

#         # Connect pre_sum outputs to integrators
#         for n in range(self.nominal_analog_system.N):
#             self.connect(
#                 self._integrators[n]._terminals[0],
#                 self._pre_sum[n]._terminals[
#                     self.nominal_analog_system.L
#                     + self.nominal_analog_system.M
#                     + self.nominal_analog_system.M_tilde
#                 ],
#             )
#             # connect integrator output to post_sum
#             for m_tilde in range(self.nominal_analog_system.M_tilde):
#                 self.connect(
#                     self.terminals[
#                         1
#                         + self.nominal_analog_system.L
#                         + self.nominal_analog_system.M
#                         + self.nominal_analog_system.M_tilde
#                         + n
#                     ],
#                     self._post_sum[m_tilde]._terminals[n],
#                 )
#         for m_tilde in range(self.nominal_analog_system.M_tilde):
#             # connect control_inputs to post_sum
#             for m in range(self.nominal_analog_system.M):
#                 self.connect(
#                     self.terminals[1 + self.nominal_analog_system.L + m],
#                     self._post_sum[m_tilde]._terminals[
#                         self.nominal_analog_system.N + self.nominal_analog_system.L + m
#                     ],
#                 )
#             self.connect(
#                 self.terminals[
#                     1
#                     + self.nominal_analog_system.L
#                     + self.nominal_analog_system.M
#                     + m_tilde
#                 ],
#                 self._post_sum[m_tilde]._terminals[
#                     self.nominal_analog_system.N
#                     + self.nominal_analog_system.L
#                     + self.nominal_analog_system.M
#                 ],
#             )
#             for l in range(self.nominal_analog_system.L):
#                 self.connect(
#                     self.terminals[1 + l],
#                     self._post_sum[m_tilde]._terminals[
#                         self.nominal_analog_system.N + l
#                     ],
#                 )

#         # Connect integrator
#         for n in range(self.nominal_analog_system.N):
#             self.connect(
#                 self.terminals[
#                     1
#                     + self.nominal_analog_system.L
#                     + self.nominal_analog_system.M
#                     + self.nominal_analog_system.M_tilde
#                     + n
#                 ],
#                 self._integrators[n]._terminals[1],
#             )

# class MultiInputIntegrator(SubCircuitElement):

#     def __init__(
#         self,
#         instance_name: str,
#         analog_system: AnalogSystem,
#         index: int,
#         C: float,
#         R: float,
#     ):
#         super().__init__(
#             instance_name,
#             f'multi_input_integrator_{index}',
#             [Terminal('VSS'), Terminal('VDD'), Terminal("CMV")]
#             + [Terminal(f'X{i}_P') for i in range(analog_system.N)]
#             + [Terminal(f'X{i}_N') for i in range(analog_system.N)]
#             + [Terminal(f'IN{i}_P') for i in range(analog_system.L)]
#             + [Terminal(f'IN{i}_N') for i in range(analog_system.L)]
#             + [Terminal(f'S{i}_P') for i in range(analog_system.M)]
#             + [Terminal(f'S{i}_N') for i in range(analog_system.M)],
#         )


class StateSpaceFrontend(CircuitAnalogFrontend):
    def __init__(
        self,
        analog_frontend: AnalogFrontend,
        vdd_voltage: float = 1.2,
        in_high=0.0,
        in_low=0.0,
    ):
        self.analog_frontend = analog_frontend
        vcm = vdd_voltage / 2.0
        super().__init__(
            analog_frontend,
            vdd_voltage,
            in_high,
            in_low,
            subckt_name='state_space_analog_frontend',
            instance_name='Xaf',
        )

        # self.Aint = Integrator(
        #     'Aint',
        #         'int',
        #         input_offset=0.0,
        #         out_initial_condition=vcm,
        #         out_lower_limit=-vdd,
        #         out_upper_limit=2 * vdd,
        #     )

        # self.add(
        #     IntegratorAnalogSystem(
        #         self.nominal_analog_frontend.analog_system,
        #         vdd_voltage,
        #         self._values_vgnd,
        #     ),
        #     DigitalControl(
        #         self.nominal_analog_frontend.digital_control,
        #         in_high,
        #         in_low,
        #         out_high,
        #         out_low,
        #     ),
        # )

        # # Connect inputs and outputs analog system
        # for l in range(self.nominal_analog_frontend.analog_system.L):
        #     self.connect(
        #         self.terminals[l + 4],
        #         self.subckt_components[0]._terminals[1 + l],
        #     )
        # for m in range(self.nominal_analog_frontend.analog_system.M):
        #     self.connect(
        #         self.terminals[m + 4 + self.nominal_analog_frontend.analog_system.L],
        #         self.subckt_components[0]._terminals[
        #             1 + self.nominal_analog_frontend.analog_system.L + m
        #         ],
        #     )
        # # connect vgnd
        # self.connect(self.terminals[1], self.subckt_components[0]._terminals[0])
        # self.connect(self.terminals[1], self.subckt_components[1]._terminals[1])

        # # Connect clock
        # self.connect(self.terminals[3], self.subckt_components[1]._terminals[0])

        # # Connect analog system and digital control
        # for m in range(self.nominal_analog_frontend.analog_system.M):
        #     self.connect(
        #         self.subckt_components[0]._terminals[
        #             1 + self.nominal_analog_frontend.analog_system.L + m
        #         ],
        #         self.subckt_components[1]._terminals[
        #             2 + self.nominal_analog_frontend.analog_system.M_tilde + m
        #         ],
        #     )

        # for m_tilde in range(self.nominal_analog_frontend.analog_system.M_tilde):
        #     self.connect(
        #         self.subckt_components[0]._terminals[
        #             1
        #             + self.nominal_analog_frontend.analog_system.L
        #             + self.nominal_analog_frontend.analog_system.M
        #             + m_tilde
        #         ],
        #         self.subckt_components[1]._terminals[2 + m_tilde],
        #     )
