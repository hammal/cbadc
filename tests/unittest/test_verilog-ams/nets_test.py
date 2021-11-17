from cbadc.verilog_ams.nature_and_disciplines import Disciplines
import cbadc.verilog_ams.nets as nets
import cbadc.verilog_ams.numbers as numbers


def test_net():
    discipline = Disciplines(
        'logic',
        None,
        None,
        'discrete'
    )
    wire = nets.Wire('A_wire', discipline)
    assert(str(wire) == 'A_wire')
    assert(wire.definition() == ['wire A_wire;', 'logic A_wire;'])


def test_logic_wire():
    wire = nets.Logic('first')
    wire_with_initial_value = nets.Logic('second', numbers.Logic('0'))

    assert(str(wire) == 'first')
    assert(wire.definition() == ['wire first;', 'logic first;'])
    assert(str(wire_with_initial_value) == 'second')
    assert(wire_with_initial_value.definition() ==
           ['wire second;', 'logic second = 0;'])


def test_electrical():
    electrical = nets.Electrical('first')
    assert(str(electrical) == 'first')
    assert(electrical.definition() == ['wire first;', 'electrical first;'])


def test_ground():
    ground = nets.Ground()
    assert(str(ground) == 'gnd')
    assert(ground.definition() == ['wire gnd;',
           'electrical gnd;', 'ground gnd;'])


def test_branch():
    net_1 = nets.Electrical('vdd')
    net_2 = nets.Ground()
    branch = nets.Branch('supply', net_1, net_2)
    assert(str(branch) == "supply")
    assert(branch.definition() == ['branch (vdd, gnd) supply;'])
