# from cbadc.circuit import (
#     DeviceModel,
#     SpiceDialect,
#     CircuitElement,
#     ComponentType,
#     Terminal,
#     SubCircuitElement,
#     top_level_descriptions,
# )


# def test_device_model():
#     model = DeviceModel(
#         name='test',
#         type_name='npn',
#         pname1='pval1',
#         pname2='pval2',
#         comments=[
#             'A device model definition in ngspice',
#             'this comment spans two lines',
#         ],
#     )
#     spice_command = model.spice_instantiate(SpiceDialect.ngspice)
#     assert (
#         spice_command
#         == '''* A device model definition in ngspice
# * this comment spans two lines
# .model test npn (pname1=pval1 pname2=pval2)'''
#     )

# two_terminals = [Terminal(str(i+1)) for i in range(2)]
# number_of_resistors = 3
# resistor_terminals = [Terminal(str(2 + i)) for i in range(number_of_resistors + 1)]


# def test_circuit_element_spice_instantiate():
#     connections = {}
#     # Resistor with name and parameters
#     print([str(term) for term in two_terminals])
#     assert (
#         CircuitElement(
#             ComponentType.resistor,
#             two_terminals,
#             '10k',
#             instance_name='my_first',
#             pname1='pval1',
#             pname2='pval2',
#         ).spice_instantiate(connections, SpiceDialect.ngspice)
#         == '''Rmy_first 1 2 10k pname1=pval1 pname2=pval2'''
#     )
#     # Resistor without name
#     assert (
#         CircuitElement(
#             ComponentType.resistor,
#             two_terminals,
#             '1k',
#             instance_name='1'
#         ).spice_instantiate(connections, SpiceDialect.ngspice)
#         == '''R1 1 2 1k'''
#     )
#     # Second resistor without name
#     assert (
#         CircuitElement(
#             ComponentType.resistor,
#             two_terminals,
#             '1k',
#             instance_name='2'
#         ).spice_instantiate(connections, SpiceDialect.ngspice)
#         == '''R2 1 2 1k'''
#     )

#      # Capacitor
#     assert (
#         CircuitElement(
#             ComponentType.capacitor,
#             two_terminals,
#             '12p',
#             m = 2,
#             instance_name='3'
#         ).spice_instantiate(connections, SpiceDialect.ngspice)
#         == '''C3 1 2 12p m=2'''
#     )

# def test_top_level_descriptions():
#     # two_terminals = [Terminal() for _ in range(2)]
#     inductor_model = DeviceModel(
#         name='ind1',
#         type_name='L',
#         tc1=1e-3,
#     )
#     description = CircuitElement(
#             ComponentType.inductor,
#             two_terminals,
#             '1u',
#             'ind1',
#             dtemp = 5,
#             m = 3,
#             model=inductor_model,
#             instance_name='1'
#         )
#     description.instance_name = '1'
#     assert description.model.spice_instantiate(SpiceDialect.ngspice) == '''.model ind1 L (tc1=0.001)'''
#     assert description.spice_instantiate({}, SpiceDialect.ngspice) == '''L1 1 2 1u ind1 dtemp=5 m=3'''

# def test_subckt():
#     subckt = SubCircuitElement(
#         resistor_terminals,
#         "voltage_divider"
#     )
#     resitors = [CircuitElement(ComponentType.resistor, [Terminal(str(i * 2)), Terminal(str(i * 2 + 1))], f'{i + 1}k') for i in range(number_of_resistors)]
#     subckt.add(*resitors)
#     for i, resistor in enumerate(resitors):
#         subckt.connect(resistor_terminals[i], resistor.terminals[0])
#         if i < number_of_resistors - 1:
#             subckt.connect(resistor.terminals[1], resistor_terminals[i + 1])
#     subckt.connect(resistor_terminals[-1], resitors[-1].terminals[1])
#     subckt.instance_name = '1'
#     assert subckt.spice_instantiate({}, SpiceDialect.ngspice) == '''X1 2 3 4 5 voltage_divider'''
#     description = top_level_descriptions(subckt)
#     for definition in description.component_definitions:
#         print(definition.definition(SpiceDialect.ngspice))
#         print(subckt)

#     print(subckt[1])


# # def test_testbench():
