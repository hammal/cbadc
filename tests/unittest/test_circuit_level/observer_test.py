import cbadc.circuit


def test_observer_initialization():
    inputs = [cbadc.circuit.Wire(f"in_{i}", True, False) for i in range(2)]
    outputs = [cbadc.circuit.Wire(f"out_{i}", False, True) for i in range(2)]
    inouts = [cbadc.circuit.Wire(f"io_{i}", True, True) for i in range(3)]
    local = [cbadc.circuit.Wire(f"n_{i}", False, False) for i in range(2)]
    ports = [*inputs, *outputs, *inouts]
    nets = [*local, *ports]

    observer = cbadc.circuit.Observer(ports, filename="observations.csv")
    observer.to_file('test_observer.vams')
    # assert False
