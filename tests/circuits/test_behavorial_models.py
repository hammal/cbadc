from cbadc.circuit import Terminal, SubCircuitElement
from cbadc.circuit.components.observer import Observer
from cbadc.circuit.components.integrator import Integrator
from cbadc.circuit.components.summer import Summer


def test_integrator_model_ngspice():
    integrator = Integrator(
        model_name='int1',
        instance_name='2',
        input_offset=0.0,
        gain=1.0,
        out_lower_limit=-10.0,
        out_upper_limit=10.0,
        limit_range=1e-6,
        out_initial_condition=0.0,
    )

    two_terminals = [Terminal('IN'), Terminal('OUT')]

    subckt = SubCircuitElement('Xsub', 'subckt', two_terminals)
    subckt.add(integrator)

    subckt.connect(subckt['IN'], integrator[0])
    subckt.connect(subckt['OUT'], integrator[1])

    print(integrator.get_ngspice(subckt._internal_connections))
    assert integrator.get_ngspice(subckt._internal_connections) == 'A2 IN OUT int1'
    print(subckt.get_ngspice(subckt._internal_connections))
    assert subckt.get_ngspice(subckt._internal_connections) == 'Xsub IN OUT subckt'
    print([model.get_ngspice() for model in subckt._get_model_set()][0])
    assert [model.get_ngspice() for model in subckt._get_model_set()][
        0
    ] == '.model int1 int(in_offset=0.0 gain=1.0 out_lower_limit=-10.0 out_upper_limit=10.0 out_ic=0.0 limit_range=1e-06)'
    assert True


# def test_integrator_model_spectre():
#     integrator = Integrator(
#         model_name='int1',
#         instance_name='Acircuit_name',
#         input_offset=0.0,
#         gain=1.0,
#         out_lower_limit=-10.0,
#         out_upper_limit=10.0,
#         limit_range=1e-6,
#         out_initial_condition=0.0,
#     )

#     two_terminals = [Terminal('IN'), Terminal('OUT')]

#     subckt = SubCircuitElement('Xsub', 'subckt', two_terminals)
#     subckt.add(integrator)

#     subckt.connects(
#         (subckt['IN'], integrator[0]),
#         (subckt['OUT'], integrator[1])
#     )

#     print(integrator.get_spectre(subckt._internal_connections))
#     assert (
#         integrator.get_spectre(subckt._internal_connections)
#         == 'Acircuit_name IN OUT int1'
#     )
#     print(subckt.get_spectre(subckt._internal_connections))
#     assert subckt.get_spectre(subckt._internal_connections) == 'Xsub IN OUT subckt'
#     print([model.get_verilog_ams() for model in subckt._get_model_set()][0])
#     assert True


def test_summer_model_ngspice():
    summer = Summer('As1', 'sum1', 3, [0.1, 0.2, 0.3], [1, -1, 2], 0, 1, ["No comment"])

    four_terminals = [
        Terminal('IN0'),
        Terminal('IN1'),
        Terminal('IN2'),
        Terminal('OUT'),
    ]

    subckt = SubCircuitElement('Xsub', 'subckt', four_terminals)
    subckt.add(summer)

    subckt.connects(
        (subckt[0], summer[0]),
        (subckt[1], summer[1]),
        (subckt[2], summer[2]),
        (subckt[3], summer[3]),
    )

    print(summer.get_ngspice(subckt._internal_connections))
    assert (
        summer.get_ngspice(subckt._internal_connections)
        == 'As1 [%vd(IN0,VCM) %vd(IN1,VCM) %vd(IN2,VCM)] OUT sum1'
    )
    print(subckt.get_ngspice(subckt._internal_connections))
    assert (
        subckt.get_ngspice(subckt._internal_connections)
        == 'Xsub IN0 IN1 IN2 OUT subckt'
    )
    print([model.get_ngspice() for model in subckt._get_model_set()][0])
    assert [model.get_ngspice() for model in subckt._get_model_set()][
        0
    ] == '.model sum1 summer(in_offset=[0.1, 0.2, 0.3] in_gain=[1, -1, 2] out_offset=0 out_gain=1)'
    assert True


# def test_summer_model_spectre():
#     summer = Summer('sum1', 3, [0.1, 0.2, 0.3], [1, -1, 2], 's1', 0, 1, ["No comment"])

#     four_terminals = [
#         Terminal('IN0'),
#         Terminal('IN1'),
#         Terminal('IN2'),
#         Terminal('OUT'),
#     ]

#     subckt = SubCircuitElement('Xsub', 'subckt', two_terminals)
#     subckt.add(summer)

#     subckt.connect(
#         four_terminals[0], subckt.subckt_components[0]._terminals[0]
#     )
#     subckt.connect(
#         four_terminals[1], subckt.subckt_components[0]._terminals[1]
#     )
#     subckt.connect(
#         four_terminals[2], subckt.subckt_components[0]._terminals[2]
#     )
#     subckt.connect(
#         four_terminals[3], subckt.subckt_components[0]._terminals[3]
#     )

#     print(summer.get_spectre(subckt._internal_connections))
#     assert (
#         summer.get_spectre(subckt._internal_connections)
#         == 'verilog_s1 IN0 IN1 IN2 OUT sum1'
#     )
#     print(subckt.get_spectre(subckt._internal_connections))
#     assert (
#         subckt.get_spectre(subckt._internal_connections)
#         == 'Xsub IN0 IN1 IN2 OUT subckt'
#     )
#     print([model.get_verilog_ams() for model in subckt._get_model_set()][0])
#     assert True


def test_observation_model_ngspice():
    input_signals = ['IN0', 'IN1', 's0', 's1', 's2']
    terminals = [Terminal('clk')] + [Terminal(s) for s in input_signals]
    observer = Observer(
        'Aobs1',
        'obs1',
        input_signals,
    )

    subckt = SubCircuitElement('Xsub1', 'test_subckt', terminals)
    subckt.add(observer)

    subckt.connects(
        (subckt[0], observer[0]),
        (subckt[1], observer[1]),
        (subckt[2], observer[2]),
        (subckt[3], observer[3]),
        (subckt[4], observer[4]),
        (subckt[5], observer[5]),
    )

    print(observer.get_ngspice(subckt._internal_connections))
    assert (
        observer.get_ngspice(subckt._internal_connections)
        == 'Aobs1 CLK IN0 IN1 S0 S1 S2 obs1'
    )
    print(subckt.get_ngspice(subckt._internal_connections))
    assert (
        subckt.get_ngspice(subckt._internal_connections)
        == 'Xsub1 CLK IN0 IN1 S0 S1 S2 test_subckt'
    )
    print([model.get_ngspice() for model in subckt._get_model_set()][0])
    # assert [model.get_ngspice() for model in subckt._get_model_set()][
    #     0
    # ] == 'to be implemented'
    assert True


# def test_observation_model_spectre():
#     input_signals = ['IN0', 'IN1', 'S0', 'S1', 'S2']
#     terminals = [Terminal('clk')] + [Terminal(s) for s in input_signals]
#     observer = Observer('obs1', input_signals, instance_name='some_name')

#     subckt = SubCircuitElement(terminals, 'test_subckt', instance_name='sub1')
#     subckt.add(observer)

#     subckt.connect(terminals[0], subckt.subckt_components[0]._terminals[0])
#     subckt.connect(terminals[1], subckt.subckt_components[0]._terminals[1])
#     subckt.connect(terminals[2], subckt.subckt_components[0]._terminals[2])
#     subckt.connect(terminals[3], subckt.subckt_components[0]._terminals[3])
#     subckt.connect(terminals[4], subckt.subckt_components[0]._terminals[4])
#     subckt.connect(terminals[5], subckt.subckt_components[0]._terminals[5])

#     print(observer.get_spectre(subckt._internal_connections))
#     assert (
#         observer.get_spectre(subckt._internal_connections)
#         == 'verilog_some_name CLK IN0 IN1 S0 S1 S2 obs1'
#     )
#     print(subckt.get_spectre(subckt._internal_connections))
#     assert (
#         subckt.get_spectre(subckt._internal_connections)
#         == 'Xsub1 CLK IN0 IN1 S0 S1 S2 test_subckt'
#     )
#     print([model.get_verilog_ams() for model in subckt._get_model_set()][0])
#     assert True
