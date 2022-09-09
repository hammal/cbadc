import cbadc.circuit


def test_Module_inputs():
    inputs = [
        cbadc.circuit.Wire(f"in_{i}", True, False, comment=f"Comment number {i + 1}")
        for i in range(2)
    ]
    ports = [*inputs]
    nets = [*ports]

    module = cbadc.circuit.Module(
        "op-amp",
        nets,
        ports,
    )
    module.render()


def test_Module_ports():
    inputs = [cbadc.circuit.Wire(f"in_{i}", True, False) for i in range(2)]
    outputs = [cbadc.circuit.Wire(f"out_{i}", False, True) for i in range(2)]
    inouts = [cbadc.circuit.Wire(f"io_{i}", True, True) for i in range(3)]
    local = [cbadc.circuit.Wire(f"n_{i}", False, False) for i in range(2)]
    ports = [*inputs, *outputs, *inouts]
    nets = [*local, *ports]

    module = cbadc.circuit.Module(
        "op-amp",
        nets,
        ports,
    )
    module.render()


def test_Module_ports_and_parameters():
    inputs = [cbadc.circuit.Wire(f"in_{i}", True, False) for i in range(2)]
    outputs = [cbadc.circuit.Wire(f"out_{i}", False, True) for i in range(2)]
    inouts = [
        cbadc.circuit.Wire(f"io_{i}", True, True, comment=f"Small comment {i}")
        for i in range(3)
    ]
    local = [cbadc.circuit.Wire(f"n_{i}", False, False) for i in range(2)]
    ports = [*inputs, *outputs, *inouts]
    nets = [*local, *ports]

    parameters = [
        *[cbadc.circuit.Parameter(f"r_{i}", real=True) for i in range(2)],
        *[
            cbadc.circuit.Parameter(f"r_{i}", real=True, initial_value=0.01)
            for i in range(2, 4)
        ],
        *[
            cbadc.circuit.Parameter(f"r_{i}", real=True, initial_value="10p")
            for i in range(4, 6)
        ],
        *[
            cbadc.circuit.Parameter(
                f"int_{i}",
                real=False,
            )
            for i in range(6, 8)
        ],
        *[
            cbadc.circuit.Parameter(f"int_{i}", real=False, initial_value=2)
            for i in range(8, 10)
        ],
        *[
            cbadc.circuit.Parameter(f"int_{i}", real=False, initial_value=3)
            for i in range(10, 12)
        ],
    ]

    module = cbadc.circuit.Module("op-amp", nets, ports, parameters=parameters)
    module.render()


def test_Module_ports_and_parameters_and_analog_statements():
    inputs = [cbadc.circuit.Wire(f"in_{i}", True, False) for i in range(2)]
    outputs = [cbadc.circuit.Wire(f"out_{i}", False, True) for i in range(2)]
    inouts = [cbadc.circuit.Wire(f"io_{i}", True, True) for i in range(3)]
    local = [cbadc.circuit.Wire(f"n_{i}", False, False) for i in range(2)]
    ports = [*inputs, *outputs, *inouts]
    nets = [*local, *ports]

    parameters = [
        *[cbadc.circuit.Parameter(f"r_{i}", real=True, analog=False) for i in range(2)],
        *[
            cbadc.circuit.Parameter(
                f"r_{i}", real=True, analog=False, initial_value=0.01
            )
            for i in range(2, 4)
        ],
        *[
            cbadc.circuit.Parameter(
                f"r_{i}", real=True, analog=True, initial_value="10p"
            )
            for i in range(4, 6)
        ],
        *[
            cbadc.circuit.Parameter(f"int_{i}", real=False, analog=False)
            for i in range(6, 8)
        ],
        *[
            cbadc.circuit.Parameter(
                f"int_{i}", real=False, analog=False, initial_value=2
            )
            for i in range(8, 10)
        ],
        *[
            cbadc.circuit.Parameter(
                f"int_{i}", real=False, analog=True, initial_value=3
            )
            for i in range(10, 12)
        ],
    ]

#     analog_statements = [
#         f"V({nets[0].name}) <+ {parameters[3].name} * V({nets[1].name})",
#         f"V({nets[4].name}) <+ {parameters[6].name} * V({nets[7].name})",
#     ]

    module = cbadc.circuit.Module(
        "op-amp",
        nets,
        ports,
        parameters=parameters,
        analog_statements=analog_statements,
    )
    module.render()


def test_Module_ports_and_parameters_and_analog_statements():
    inputs = [cbadc.circuit.Wire(f"in_{i}", True, False) for i in range(2)]
    outputs = [cbadc.circuit.Wire(f"out_{i}", False, True) for i in range(2)]
    inouts = [cbadc.circuit.Wire(f"io_{i}", True, True) for i in range(3)]
    local = [cbadc.circuit.Wire(f"n_{i}", False, False) for i in range(2)]
    ports = [*inputs, *outputs, *inouts]
    nets = [*local, *ports]

    parameters = [
        *[
            cbadc.circuit.Parameter(
                f"r_{i}",
                real=True,
            )
            for i in range(2)
        ],
        *[
            cbadc.circuit.Parameter(f"r_{i}", real=True, initial_value=0.01)
            for i in range(2, 4)
        ],
        *[
            cbadc.circuit.Parameter(f"r_{i}", real=True, initial_value="10p")
            for i in range(4, 6)
        ],
        *[
            cbadc.circuit.Parameter(
                f"int_{i}",
                real=False,
            )
            for i in range(6, 8)
        ],
        *[
            cbadc.circuit.Parameter(f"int_{i}", real=False, initial_value=2)
            for i in range(8, 10)
        ],
        *[
            cbadc.circuit.Parameter(f"int_{i}", real=False, initial_value=3)
            for i in range(10, 12)
        ],
    ]

#     analog_statements = [
#         f"V({nets[0].name}) <+ {parameters[3].name} * V({nets[1].name})",
#         f"V({nets[4].name}) <+ {parameters[6].name} * V({nets[7].name})",
#     ]

    module = cbadc.circuit.Module(
        "op-amp",
        nets,
        ports,
        parameters=parameters,
        analog_statements=analog_statements,
    )
    module.render()


def test_Module_ports_and_parameters_and_submodules():
    inputs = [cbadc.circuit.Wire(f"in_{i}", True, False) for i in range(2)]
    outputs = [cbadc.circuit.Wire(f"out_{i}", False, True) for i in range(2)]
    inouts = [cbadc.circuit.Wire(f"io_{i}", True, True) for i in range(3)]
    local = [cbadc.circuit.Wire(f"n_{i}", False, False) for i in range(2)]
    ports = [*inputs, *outputs, *inouts]
    nets = [*local, *ports]

    parameters = [
        *[
            cbadc.circuit.Parameter(f"r_{i}", real=True, initial_value="10p")
            for i in range(4, 6)
        ],
    ]
    analog_statements = [
        f"V({nets[0].name}) <+ {parameters[0].name} * V({nets[1].name})",
        f"V({nets[4].name}) <+ {parameters[1].name} * V({nets[7].name})",
    ]

    modules = [
        cbadc.circuit.module.SubModules(
            cbadc.circuit.Module(
                f"submodule_{i}",
                nets,
                ports[i : 1 + i * 2],
                parameters=parameters,
                instance_name=f"instance_{i}",
                analog_statements=analog_statements,
            ),
            ports[: len(range(i, 1 + i * 2))],
        )
        for i in range(0, 3)
    ]
    super_module = cbadc.circuit.Module(
        "super_module", nets, ports, parameters=parameters, submodules=modules
    )
    super_module.render()
