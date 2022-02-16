import cbadc.circuit_level


def test_Module_inputs():
    inputs = [
        cbadc.circuit_level.Wire(
            f"in_{i}", True, False, comment=f"Comment number {i + 1}"
        )
        for i in range(2)
    ]
    ports = [*inputs]
    nets = [*ports]

    module = cbadc.circuit_level.Module(
        "op-amp",
        nets,
        ports,
    )
    assert (
        module.render()[0][-1]
        == """// op-amp
//
// Ports: in_0, in_1
//
// Parameters:
//
module op-amp(in_0, in_1);

    input in_0; // Comment number 1
    input in_1; // Comment number 2


endmodule"""
    )


def test_Module_ports():
    inputs = [cbadc.circuit_level.Wire(f"in_{i}", True, False) for i in range(2)]
    outputs = [cbadc.circuit_level.Wire(f"out_{i}", False, True) for i in range(2)]
    inouts = [cbadc.circuit_level.Wire(f"io_{i}", True, True) for i in range(3)]
    local = [cbadc.circuit_level.Wire(f"n_{i}", False, False) for i in range(2)]
    ports = [*inputs, *outputs, *inouts]
    nets = [*local, *ports]

    module = cbadc.circuit_level.Module(
        "op-amp",
        nets,
        ports,
    )
    assert (
        module.render()[0][-1]
        == """// op-amp
//
// Ports: in_0, in_1, out_0, out_1, io_0, io_1, io_2
//
// Parameters:
//
module op-amp(in_0, in_1, out_0, out_1, io_0, io_1, io_2);

    input in_0;
    input in_1;

    output out_0;
    output out_1;

    inout io_0;
    inout io_1;
    inout io_2;


endmodule"""
    )


def test_Module_ports_and_parameters():
    inputs = [cbadc.circuit_level.Wire(f"in_{i}", True, False) for i in range(2)]
    outputs = [cbadc.circuit_level.Wire(f"out_{i}", False, True) for i in range(2)]
    inouts = [
        cbadc.circuit_level.Wire(f"io_{i}", True, True, comment=f"Small comment {i}")
        for i in range(3)
    ]
    local = [cbadc.circuit_level.Wire(f"n_{i}", False, False) for i in range(2)]
    ports = [*inputs, *outputs, *inouts]
    nets = [*local, *ports]

    parameters = [
        *[cbadc.circuit_level.Parameter(f"r_{i}", real=True) for i in range(2)],
        *[
            cbadc.circuit_level.Parameter(f"r_{i}", real=True, initial_value=0.01)
            for i in range(2, 4)
        ],
        *[
            cbadc.circuit_level.Parameter(f"r_{i}", real=True, initial_value="10p")
            for i in range(4, 6)
        ],
        *[
            cbadc.circuit_level.Parameter(
                f"int_{i}",
                real=False,
            )
            for i in range(6, 8)
        ],
        *[
            cbadc.circuit_level.Parameter(f"int_{i}", real=False, initial_value=2)
            for i in range(8, 10)
        ],
        *[
            cbadc.circuit_level.Parameter(f"int_{i}", real=False, initial_value=3)
            for i in range(10, 12)
        ],
    ]

    module = cbadc.circuit_level.Module("op-amp", nets, ports, parameters=parameters)
    assert (
        module.render()[0][-1]
        == """// op-amp
//
// Ports: in_0, in_1, out_0, out_1, io_0, io_1, io_2
//
// Parameters: r_0, r_1, r_2, r_3, r_4, r_5, int_6, int_7, int_8, int_9, int_10, int_11
//
module op-amp(in_0, in_1, out_0, out_1, io_0, io_1, io_2);

    input in_0;
    input in_1;

    output out_0;
    output out_1;

    inout io_0; // Small comment 0
    inout io_1; // Small comment 1
    inout io_2; // Small comment 2

    parameter real r_0;
    parameter real r_1;
    parameter real r_2 = 0.01;
    parameter real r_3 = 0.01;
    parameter real r_4 = 10p;
    parameter real r_5 = 10p;

    parameter int int_6;
    parameter int int_7;
    parameter int int_8 = 2;
    parameter int int_9 = 2;
    parameter int int_10 = 3;
    parameter int int_11 = 3;

    electrical int_6;
    electrical int_7;
    electrical int_8;
    electrical int_9;
    electrical int_10;
    electrical int_11;

endmodule"""
    )


def test_Module_ports_and_parameters_and_analog_statements():
    inputs = [cbadc.circuit_level.Wire(f"in_{i}", True, False) for i in range(2)]
    outputs = [cbadc.circuit_level.Wire(f"out_{i}", False, True) for i in range(2)]
    inouts = [cbadc.circuit_level.Wire(f"io_{i}", True, True) for i in range(3)]
    local = [cbadc.circuit_level.Wire(f"n_{i}", False, False) for i in range(2)]
    ports = [*inputs, *outputs, *inouts]
    nets = [*local, *ports]

    parameters = [
        *[
            cbadc.circuit_level.Parameter(f"r_{i}", real=True, analog=False)
            for i in range(2)
        ],
        *[
            cbadc.circuit_level.Parameter(
                f"r_{i}", real=True, analog=False, initial_value=0.01
            )
            for i in range(2, 4)
        ],
        *[
            cbadc.circuit_level.Parameter(
                f"r_{i}", real=True, analog=True, initial_value="10p"
            )
            for i in range(4, 6)
        ],
        *[
            cbadc.circuit_level.Parameter(f"int_{i}", real=False, analog=False)
            for i in range(6, 8)
        ],
        *[
            cbadc.circuit_level.Parameter(
                f"int_{i}", real=False, analog=False, initial_value=2
            )
            for i in range(8, 10)
        ],
        *[
            cbadc.circuit_level.Parameter(
                f"int_{i}", real=False, analog=True, initial_value=3
            )
            for i in range(10, 12)
        ],
    ]

    analog_statements = [
        f"V({nets[0].name}) <+ {parameters[3].name} * V({nets[1].name})",
        f"V({nets[4].name}) <+ {parameters[6].name} * V({nets[7].name})",
    ]

    module = cbadc.circuit_level.Module(
        "op-amp",
        nets,
        ports,
        parameters=parameters,
        analog_statements=analog_statements,
    )
    assert (
        module.render()[0][-1]
        == """module op-amp(in_0, in_1, out_0, out_1, io_0, io_1, io_2);

    input in_0, in_1;
    output out_0, out_1;
    inout io_0, io_1, io_2;

    parameters real r_0, r_1, r_2 = 0.01, r_3 = 0.01, r_4, r_5;
    parameters int int_6, int_7, int_8 = 2, int_9 = 2, int_10, int_11;

    electrical n_0, n_1, in_0, in_1, out_0, out_1, io_0, io_1, io_2;

    analog initial begin
        r_4 = 10p;
        r_5 = 10p;
        int_10 = 3;
        int_11 = 3;
    end

    analog begin
        V(n_0) <+ r_3 * V(n_1);
        V(out_0) <+ int_6 * V(io_1);
    end

endmodule"""
    )


def test_Module_ports_and_parameters_and_analog_statements():
    inputs = [cbadc.circuit_level.Wire(f"in_{i}", True, False) for i in range(2)]
    outputs = [cbadc.circuit_level.Wire(f"out_{i}", False, True) for i in range(2)]
    inouts = [cbadc.circuit_level.Wire(f"io_{i}", True, True) for i in range(3)]
    local = [cbadc.circuit_level.Wire(f"n_{i}", False, False) for i in range(2)]
    ports = [*inputs, *outputs, *inouts]
    nets = [*local, *ports]

    parameters = [
        *[
            cbadc.circuit_level.Parameter(
                f"r_{i}",
                real=True,
            )
            for i in range(2)
        ],
        *[
            cbadc.circuit_level.Parameter(f"r_{i}", real=True, initial_value=0.01)
            for i in range(2, 4)
        ],
        *[
            cbadc.circuit_level.Parameter(f"r_{i}", real=True, initial_value="10p")
            for i in range(4, 6)
        ],
        *[
            cbadc.circuit_level.Parameter(
                f"int_{i}",
                real=False,
            )
            for i in range(6, 8)
        ],
        *[
            cbadc.circuit_level.Parameter(f"int_{i}", real=False, initial_value=2)
            for i in range(8, 10)
        ],
        *[
            cbadc.circuit_level.Parameter(f"int_{i}", real=False, initial_value=3)
            for i in range(10, 12)
        ],
    ]

    analog_statements = [
        f"V({nets[0].name}) <+ {parameters[3].name} * V({nets[1].name})",
        f"V({nets[4].name}) <+ {parameters[6].name} * V({nets[7].name})",
    ]

    module = cbadc.circuit_level.Module(
        "op-amp",
        nets,
        ports,
        parameters=parameters,
        analog_statements=analog_statements,
    )
    assert (
        module.render()[0][-1]
        == """// op-amp
//
// Ports: in_0, in_1, out_0, out_1, io_0, io_1, io_2
//
// Parameters: r_0, r_1, r_2, r_3, r_4, r_5, int_6, int_7, int_8, int_9, int_10, int_11
//
module op-amp(in_0, in_1, out_0, out_1, io_0, io_1, io_2);

    input in_0;
    input in_1;

    output out_0;
    output out_1;

    inout io_0;
    inout io_1;
    inout io_2;

    parameter real r_0;
    parameter real r_1;
    parameter real r_2 = 0.01;
    parameter real r_3 = 0.01;
    parameter real r_4 = 10p;
    parameter real r_5 = 10p;

    parameter int int_6;
    parameter int int_7;
    parameter int int_8 = 2;
    parameter int int_9 = 2;
    parameter int int_10 = 3;
    parameter int int_11 = 3;

    electrical int_6;
    electrical int_7;
    electrical int_8;
    electrical int_9;
    electrical int_10;
    electrical int_11;

    analog begin
        V(n_0) <+ r_3 * V(n_1)
        V(out_0) <+ int_6 * V(io_1)
    end

endmodule"""
    )


def test_Module_ports_and_parameters_and_submodules():
    inputs = [cbadc.circuit_level.Wire(f"in_{i}", True, False) for i in range(2)]
    outputs = [cbadc.circuit_level.Wire(f"out_{i}", False, True) for i in range(2)]
    inouts = [cbadc.circuit_level.Wire(f"io_{i}", True, True) for i in range(3)]
    local = [cbadc.circuit_level.Wire(f"n_{i}", False, False) for i in range(2)]
    ports = [*inputs, *outputs, *inouts]
    nets = [*local, *ports]

    parameters = [
        *[
            cbadc.circuit_level.Parameter(f"r_{i}", real=True, initial_value="10p")
            for i in range(4, 6)
        ],
    ]
    analog_statements = [
        f"V({nets[0].name}) <+ {parameters[0].name} * V({nets[1].name})",
        f"V({nets[4].name}) <+ {parameters[1].name} * V({nets[7].name})",
    ]

    modules = [
        cbadc.circuit_level.module.SubModules(
            cbadc.circuit_level.Module(
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
    super_module = cbadc.circuit_level.Module(
        "super_module", nets, ports, parameters=parameters, submodules=modules
    )
    assert (
        "\n\n\n".join(super_module.render()[0])
        == """// submodule_2
//
// Ports: out_0, out_1, io_0
//
// Parameters: r_4, r_5
//
module submodule_2(out_0, out_1, io_0);


    output out_0;
    output out_1;

    inout io_0;

    parameter real r_4 = 10p;
    parameter real r_5 = 10p;


    analog begin
        V(n_0) <+ r_4 * V(n_1)
        V(out_0) <+ r_5 * V(io_1)
    end

endmodule


// submodule_1
//
// Ports: in_1, out_0
//
// Parameters: r_4, r_5
//
module submodule_1(in_1, out_0);

    input in_1;

    output out_0;

    parameter real r_4 = 10p;
    parameter real r_5 = 10p;


    analog begin
        V(n_0) <+ r_4 * V(n_1)
        V(out_0) <+ r_5 * V(io_1)
    end

endmodule


// submodule_0
//
// Ports: in_0
//
// Parameters: r_4, r_5
//
module submodule_0(in_0);

    input in_0;

    parameter real r_4 = 10p;
    parameter real r_5 = 10p;


    analog begin
        V(n_0) <+ r_4 * V(n_1)
        V(out_0) <+ r_5 * V(io_1)
    end

endmodule


// super_module
//
// Ports: in_0, in_1, out_0, out_1, io_0, io_1, io_2
//
// Parameters: r_4, r_5
//
module super_module(in_0, in_1, out_0, out_1, io_0, io_1, io_2);

    input in_0;
    input in_1;

    output out_0;
    output out_1;

    inout io_0;
    inout io_1;
    inout io_2;

    parameter real r_4 = 10p;
    parameter real r_5 = 10p;



    submodule_0 #(
            .r_4(r_4),
            .r_5(r_5)
    ) instance_0 (
            .in_0(in_0)
    );

    submodule_1 #(
            .r_4(r_4),
            .r_5(r_5)
    ) instance_1 (
            .in_1(in_0),
            .out_0(in_1)
    );

    submodule_2 #(
            .r_4(r_4),
            .r_5(r_5)
    ) instance_2 (
            .out_0(in_0),
            .out_1(in_1),
            .io_0(out_0)
    );

endmodule"""
    )
