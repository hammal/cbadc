from cbadc.circuit import Terminal, SubCircuitElement
from cbadc.circuit.components.reference_source import ReferenceSource


def test_comparator_ngspice():
    terminals = [Terminal('clk'), Terminal('in'), Terminal('out')]
    subckt = SubCircuitElement('Xsub', 'subckt', terminals)
    subckt.add(
        ReferenceSource(
            instance_name='Ars',
            input_filename='input.txt',
            number_of_sources=3,
            model_name='ref',
        )
    )

    subckt.connects(
        (subckt[0], subckt.Ars[0]),
        (subckt[1], subckt.Ars[1]),
        (subckt[2], subckt.Ars[2]),
    )


    print(subckt.Ars.get_ngspice(subckt._internal_connections))
    assert (
        subckt.Ars.get_ngspice(subckt._internal_connections)
        == 'Ars [CLK IN OUT] ref'
    )
    print(subckt.get_ngspice(subckt._internal_connections))
    assert subckt.get_ngspice(subckt._internal_connections) == 'Xsub CLK IN OUT subckt'
    print([model.get_ngspice() for model in subckt._get_model_set()][0])
    print('\n\n')
    for subckt_definition in subckt.get_sub_circuit_definitions():
        print(subckt_definition)
        print('\n\n')
    assert True
