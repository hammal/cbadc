import cbadc.circuit.digital_control
from tests.fixture.chain_of_integrators import get_simulator, chain_of_integrators


def test_ComparatorModule():
    comparator = cbadc.circuit.digital_control.Comparator("my_comparator")
    comparator.to_file("comparator")


def test_DigitalControlModule(get_simulator):
    digital_control_module = cbadc.circuit.digital_control.DigitalControl(
        get_simulator.digital_control
    )
    digital_control_module.to_file("digital_control")
