from cbadc.circuit.components.passives import Resistor, Capacitor, Inductor
from cbadc.circuit.components.sources import (
    DCVoltageSource,
    PulseVoltageSource,
    SinusoidalVoltageSource,
)
from cbadc.circuit import Terminal, SubCircuitElement

three_terminals = [Terminal('p'), Terminal('n'), Terminal('g')]


def test_subckt():
    SubCircuitElement('Xsub', 'subckt', three_terminals)


def test_resistor():
    subckt = SubCircuitElement('Xsub', 'subckt', three_terminals)
    subckt.add(Resistor('R2', 1e3))
    subckt.add(Resistor('R3', 1e6))
    subckt.R4 = Resistor('R4', 1e2)

    subckt.connects(
        (subckt[0], subckt.R2[0]),
        (subckt[1], subckt.R2[1]),
        (subckt[0], subckt.R3[0]),
        (subckt[1], subckt.R3[1]),
        (subckt[0], subckt.R4[0]),
        (subckt[1], subckt.R4[1]),
    )

    assert subckt.R2.get_ngspice(subckt._internal_connections) == 'R2 P N r=1000.0'
    assert subckt.R3.get_ngspice(subckt._internal_connections) == 'R3 P N r=1000000.0'
    assert subckt.R4.get_ngspice(subckt._internal_connections) == 'R4 P N r=100.0'

    # # Spectre
    # assert (
    #     subckt.subckt_components[0].get_spectre(subckt._internal_connections)
    #     == 'R2 P N r=1000.0'
    # )
    # assert (
    #     subckt.subckt_components[1].get_spectre(subckt._internal_connections)
    #     == 'R1 P N r=1k'
    # )
    # assert (
    #     subckt.subckt_components[2].get_spectre(subckt._internal_connections)
    #     == 'R4 P N r=100'
    # )


def test_capacitor():
    subckt = SubCircuitElement('Xsub', 'subckt', three_terminals)
    subckt.add(Capacitor('C2', 1e-12))
    subckt.connect(subckt[0], subckt.C2[0])
    subckt.connect(subckt[1], subckt.C2[1])

    assert subckt.C2.get_ngspice(subckt._internal_connections) == 'C2 P N 1e-12'
    # assert (
    #     subckt.subckt_components[0].get_spectre(subckt._internal_connections)
    #     == 'C2 P N c=1e-12'
    # )


def test_inductor():
    subckt = SubCircuitElement('Xsub', 'subckt', three_terminals)
    subckt.add(Inductor('L3', 1e-9))
    subckt.connects(
        (subckt[0], subckt.L3[0]),
        (subckt[1], subckt.L3[1]),
    )
    assert subckt.L3.get_ngspice(subckt._internal_connections) == 'L3 P N 1e-09'
    # assert (
    #     subckt.subckt_components[0].get_spectre(subckt._internal_connections)
    #     == 'L2 P N l=1e-09'
    # )


def test_DC_voltage_source():
    subckt = SubCircuitElement('Xsub', 'subckt', three_terminals)
    subckt.add(DCVoltageSource('Vdd', 1))

    subckt.connects(
        (subckt[0], subckt.Vdd[0]),
        (subckt[1], subckt.Vdd[1]),
    )
    assert subckt.Vdd.get_ngspice(subckt._internal_connections) == 'Vdd P N DC 1'
    # assert (
    #     subckt.subckt_components[0].get_spectre(subckt._internal_connections)
    #     == 'V2 P N vsource dc=1 type=dc'
    # )


def test_PULSE_voltage_source_ngspice():
    low = -1
    high = 1
    delay = 1e-3
    rise_time = 1e-4
    fall_time = 1e-5
    period = 1e-2
    subckt = SubCircuitElement('Xsub', 'subckt', three_terminals)
    subckt.add(PulseVoltageSource('V2', low, high, period, rise_time, fall_time))

    subckt.connects(
        (subckt[0], subckt.V2[0]),
        (subckt[1], subckt.V2[1]),
    )

    assert (
        subckt.V2.get_ngspice(subckt._internal_connections)
        == 'V2 P N PULSE(-1 1 0.0 0.0001 1e-05 0.0049900000000000005 0.01) DC 0.0'
    )


# def test_PULSE_voltage_source_spectre():
#     low = -1
#     high = 1
#     delay = 1e-3
#     rise_time = 1e-4
#     fall_time = 1e-5
#     period = 1e-2
#     subckt = SubCircuitElement('Xsub', 'subckt', three_terminals)
#     subckt.add(
#         PulseVoltageSource(
#             low, high, period, rise_time, fall_time, delay, instance_name='2'
#         )
#     )
#     subckt.connect(three_terminals[0], subckt.subckt_components[0]._terminals[0])
#     subckt.connect(three_terminals[1], subckt.subckt_components[0]._terminals[1])
#     assert (
#         subckt.subckt_components[0].get_spectre(subckt._internal_connections)
#         == 'V2 P N vsource type=pulse -1 1 1e-02 1e-04 1e-05 1e-03'
#     )


def test_SIN_voltage_source_ngspice():
    amplitude = 1
    frequency = 1e3
    delay = 1e-3
    offset = 1e-12
    damping_factor = -1
    phase = 3.14
    subckt = SubCircuitElement('Xsub', 'subckt', three_terminals)
    subckt.add(
        SinusoidalVoltageSource(
            'V2',
            offset=offset,
            amplitude=amplitude,
            frequency=frequency,
            delay_time=delay,
            phase=phase,
            damping_factor=damping_factor,
        )
    )

    subckt.connects(
        (subckt[0], subckt.V2[0]),
        (subckt[1], subckt.V2[1]),
    )

    assert (
        subckt.V2.get_ngspice(subckt._internal_connections)
        == 'V2 P N SIN(1e-12 1 1000.0 0.001 -1 3.14) DC 0.0 AC 1'
    )


# def test_SIN_voltage_source_spectre():
#     amplitude = 1
#     frequency = 1e3
#     delay = 1e-3
#     offset = 1e-12
#     subckt = SubCircuitElement('Xsub', 'subckt', three_terminals)
#     subckt.add(
#         SinusoidalVoltageSource(
#             offset=offset,
#             amplitude=amplitude,
#             frequency=frequency,
#             delay_time=delay,
#             instance_name='2',
#         )
#     )
#     subckt.connect(three_terminals[0], subckt.subckt_components[0]._terminals[0])
#     subckt.connect(three_terminals[1], subckt.subckt_components[0]._terminals[1])
#     assert (
#         subckt.subckt_components[0].get_spectre(subckt._internal_connections)
#         == 'V2 P N vsource type=sine sinedc=1e-12 ampl=1 freq=1000.0 sinephase=0'
#     )
