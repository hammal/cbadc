"""The analog frontend"""
import cbadc


class AnalogFrontend:
    """Represents an analog frontend.

    Parameters
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        an analog system instance
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        a digital control instance

    Attributes
    ----------
    analog_system: :py:class:`cbadc.analog_system.AnalogSystem`
        the analog frontend's analog system instance
    digital_control: :py:class:`cbadc.digital_control.DigitalControl`
        the analog frontend's digital control instance
    """

    def __init__(
        self,
        analog_system: cbadc.analog_system._valid_analog_system_types,
        digital_control: cbadc.digital_control._valid_digital_control_types,
    ):
        self.analog_system = analog_system
        self.digital_control = digital_control
