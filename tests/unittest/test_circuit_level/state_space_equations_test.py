from tests.fixture.chain_of_integrators import get_simulator, chain_of_integrators
import cbadc.circuit.state_space_equations


def test_AnalogSystemModule(chain_of_integrators):
    analog_system_module = cbadc.circuit.state_space_equations.AnalogSystem(
        chain_of_integrators['system']
    )
    analog_system_module.render()


def test_AnalogSystemModule_module_comment(chain_of_integrators):
    analog_system_module = cbadc.circuit.state_space_equations.AnalogSystem(
        chain_of_integrators['system']
    )
    str(analog_system_module)
