from cbadc.circuit import Terminal, SubCircuitElement
from cbadc.circuit.components.reference_source import ReferenceSource
import numpy as np


def test_comparator_ngspice():
    terminals = [Terminal("clk"), Terminal("in"), Terminal("out")]
    subckt = SubCircuitElement("Xsub", "subckt", terminals)
    rng = np.random.default_rng()
    random_sequence_length = 1 << 10
    number_of_sources = 3
    subckt.add(
        ReferenceSource(
            instance_name="Ars",
            input_filename="input.txt",
            number_of_sources=number_of_sources,
            model_name="ref",
            pseudo_random_sequence=rng.integers(
                2, size=(random_sequence_length, number_of_sources)
            ),
        )
    )

    subckt.connects(
        (subckt[0], subckt.Ars[0]),
        (subckt[1], subckt.Ars[1]),
        (subckt[2], subckt.Ars[2]),
    )

    print(subckt.Ars.get_ngspice(subckt._internal_connections))
    assert (
        subckt.Ars.get_ngspice(subckt._internal_connections) == "Ars [CLK IN OUT] ref"
    )
    print(subckt.get_ngspice(subckt._internal_connections))
    assert subckt.get_ngspice(subckt._internal_connections) == "Xsub CLK IN OUT subckt"
    print([model.get_ngspice() for model in subckt._get_model_set()][0])
    print("\n\n")
    for subckt_definition in subckt.get_sub_circuit_definitions():
        print(subckt_definition)
        print("\n\n")
    assert True
