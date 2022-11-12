import cbadc
from tests.fixture.chain_of_integrators import get_simulator, chain_of_integrators


def test_AnalogFrontend(get_simulator):
    digital_control_module = cbadc.circuit.DigitalControl(get_simulator.digital_control)
    analog_system_module = cbadc.circuit.state_space_equations.AnalogSystem(
        get_simulator.analog_system
    )
    analog_frontend_module = cbadc.circuit.analog_frontend.AnalogFrontend(
        analog_system_module, digital_control_module
    )
    analog_frontend_module.to_file("chain_of_integrators.v")
