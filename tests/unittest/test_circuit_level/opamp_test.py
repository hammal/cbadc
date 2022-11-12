from tests.fixture.chain_of_integrators import (
    get_simulator,
    chain_of_integrators,
    chain_of_integrators_op_amp,
    chain_of_integrators_op_amp_small,
)

# import cbadc.circuit
import cbadc
import numpy as np
import cbadc.analog_system
import scipy.signal


def test_ideal_op_amp():
    op_amp = cbadc.circuit.op_amp.op_amp.IdealOpAmp("my_op_amp")
    op_amp.render()


def test_op_amp_with_first_order_pole():

    A_DC = 1e3
    omega_0 = 2 * np.pi * 1e3
    op_amp = cbadc.circuit.op_amp.op_amp.FirstOrderPoleOpAmp("my_op_amp", A_DC, omega_0)
    op_amp.render()


# def test_op_amp_with_finite_gain():

#     A_DC = 1e3
#     op_amp = cbadc.circuit.op_amp.op_amp.FiniteGainOpAmp("my_op_amp", A_DC)
#     op_amp.render()


def test_op_amp_integrator_ideal():

    C = 1e-12
    op_amp = cbadc.circuit.op_amp.amplifier_configurations.InvertingAmplifierCapacitiveFeedback(
        "my_op_amp", C, cbadc.circuit.op_amp.op_amp.IdealOpAmp
    )
    op_amp.render()


# def test_op_amp_integrator_finite_gain():

#     A_DC = 1e3
#     C = 1e-12
#     op_amp = cbadc.circuit.op_amp.amplifier_configurations.InvertingAmplifierCapacitiveFeedback(
#         "my_op_amp", C, cbadc.circuit.op_amp.op_amp.FiniteGainOpAmp, A_DC=A_DC
#     )
#     op_amp.render()


def test_op_amp_integrator_first_order_pole():

    A_DC = 1e3
    omega_p = 2 * np.pi * 1e3
    C = 1e-12
    op_amp = cbadc.circuit.op_amp.amplifier_configurations.InvertingAmplifierCapacitiveFeedback(
        "my_op_amp",
        C,
        cbadc.circuit.op_amp.op_amp.FirstOrderPoleOpAmp,
        A_DC=A_DC,
        omega_p=omega_p,
    )
    op_amp.render()


def test_resistor_network(chain_of_integrators):
    C = 1e-12
    G = chain_of_integrators['system'].A * C
    resistor_network_module = cbadc.circuit.op_amp.resistor_network.ResistorNetwork(
        "module_name", "instance_name", G
    )
    resistor_network_module.render()


def test_analog_system_ideal_op_amp(chain_of_integrators_op_amp):
    C = 1e-12
    analog_system_module = cbadc.circuit.AnalogSystemIdealOpAmp(
        analog_system=chain_of_integrators_op_amp['system'], C=C
    )
    analog_system_module.render()


# def test_analog_system_final_gain_op_amp(chain_of_integrators_op_amp):
#     C = 1e-12
#     A_DC = 1e2
#     analog_system = chain_of_integrators_op_amp['system']
#     analog_system_module = cbadc.circuit.op_amp.AnalogSystemFiniteGainOpAmp(
#         analog_system=chain_of_integrators_op_amp['system'], C=C, A_DC=A_DC
#     )
#     analog_system_module.render()


def test_analog_system_first_order_pole_op_amp(chain_of_integrators_op_amp):
    C = 1e-12
    A_DC = 1e2
    omega_p = 1e4
    analog_system_module = cbadc.circuit.AnalogSystemFirstOrderPoleOpAmp(
        analog_system=chain_of_integrators_op_amp['system'],
        C=C,
        A_DC=A_DC,
        omega_p=omega_p,
    )
    analog_system_module.render()


# def test_analog_system_n_th_order_pole_op_amp(chain_of_integrators_op_amp_small):
#     C = 1e-12
#     N = 3
#     Wn = 1e3

#     amplifier = cbadc.analog_system.filters.ButterWorth(N, Wn)
#     amplifier.B = -1e3 * amplifier.B

#     analog_system_module = cbadc.circuit.AnalogSystemStateSpaceEquations(
#         chain_of_integrators_op_amp_small['system'],
#         C,
#         [amplifier],
#     )
#     analog_system_module.render()
