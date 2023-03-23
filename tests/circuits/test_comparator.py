from cbadc.circuit import Terminal, SubCircuitElement
from cbadc.circuit.components.comparator import ClockedComparator


def test_comparator_ngspice():
    vdd = 1.2
    vgnd = 0.0
    in_low = vdd / 3
    in_high = 2 * vdd / 3
    out_low = vgnd
    out_high = vdd
    out_undef = vdd / 2

    terminals = [Terminal('clk'), Terminal('in'), Terminal('out'), Terminal('VCM')]
    subckt = SubCircuitElement('Xsub', 'subckt', terminals)
    subckt.add(
        ClockedComparator(
            'Xcc',
            'comp',
            in_low,
            in_high,
            out_low,
            out_high,
            out_undef,
        )
    )
    subckt.connects(
        (subckt['CLK'], subckt.Xcc['CLK']),
        (subckt['IN'], subckt.Xcc['IN']),
        (subckt['OUT'], subckt.Xcc['OUT']),
        (subckt['VCM'], subckt.Xcc['VCM']),
    )

    print(subckt.Xcc.get_ngspice(subckt._internal_connections))
    assert (
        subckt.Xcc.get_ngspice(subckt._internal_connections)
        == 'Xcc CLK VCM IN OUT comp'
    )
    print(subckt.get_ngspice(subckt._internal_connections))
    assert (
        subckt.get_ngspice(subckt._internal_connections) == 'Xsub CLK IN OUT VCM subckt'
    )
    print([model.get_ngspice() for model in subckt._get_model_set()][0])
    print('\n\n')
    for subckt_definition in subckt.get_sub_circuit_definitions():
        print(subckt_definition)
        print('\n\n')
    assert True


# def test_comparator_spectre():
#     vdd = 1.2
#     vgnd = 0.0
#     in_low = vdd / 3
#     in_high = 2 * vdd / 3
#     out_low = vgnd
#     out_high = vdd
#     out_undef = vdd / 2

#     terminals = [Terminal('clk'), Terminal('in'), Terminal('out')]
#     subckt = SubCircuitElement(terminals, 'subckt', instance_name='sub')
#     subckt.add(
#         ClockedComparator(
#             'comp',
#             in_low,
#             in_high,
#             out_low,
#             out_high,
#             out_undef,
#         )
#     )

#     subckt.connect(subckt.terminals[0], subckt.subckt_components[0]._terminals[0])
#     subckt.connect(subckt.terminals[1], subckt.subckt_components[0]._terminals[1])
#     subckt.connect(subckt.terminals[2], subckt.subckt_components[0]._terminals[2])

#     print(subckt.subckt_components[0].get_spectre(subckt._internal_connections))
#     assert subckt.subckt_components[0].get_spectre(subckt._internal_connections) == 'verilog_1 CLK IN OUT comp'
#     print(subckt.get_spectre(subckt._internal_connections))
#     assert subckt.get_spectre(subckt._internal_connections) == 'Xsub CLK IN OUT subckt'
#     print([model.get_verilog_ams() for model in subckt._get_model_set()][0])
#     print('\n\n')
#     for subckt_definition in subckt._get_spectre_sub_circuit_definition():
#         print(subckt_definition)
#         print('\n\n')
#     assert True
