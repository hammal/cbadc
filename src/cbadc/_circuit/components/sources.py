from .. import ComponentType, CircuitElement, Port, Terminal, _template_env, SPICE_VALUE
from typing import Dict, List, Union


class DCVoltageSource(CircuitElement):
    dc: SPICE_VALUE

    def __init__(
        self,
        instance_name: str,
        value: SPICE_VALUE,
    ):
        if not isinstance(value, (float, str, int)):
            raise TypeError(f'Expected float, str, or int, got {type(value)}')

        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'V':
            instance_name = 'V' + instance_name

        super().__init__(
            instance_name,
            [Terminal(), Terminal()],
            dc=value,
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('ngspice/dc_voltage_source.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'value': self._parameters_dict['dc'],
                'comments': self.comments,
            }
        )

    def get_spectre(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('spectre/dc_voltage_source.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'value': self._parameters_dict['dc'],
                'comments': self.comments,
            }
        )


class PulseVoltageSource(CircuitElement):
    val0: SPICE_VALUE
    val1: SPICE_VALUE
    pulse_width: SPICE_VALUE
    rise_time: SPICE_VALUE
    fall_time: SPICE_VALUE
    delay_time: SPICE_VALUE
    period: SPICE_VALUE

    def __init__(
        self,
        instance_name: str,
        val0: SPICE_VALUE,
        val1: SPICE_VALUE,
        period: SPICE_VALUE,
        rise_time: SPICE_VALUE,
        fall_time: SPICE_VALUE,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'V':
            instance_name = 'V' + instance_name

        super().__init__(instance_name, [Terminal(), Terminal()])
        self.val0 = val0
        self.val1 = val1

        self.delay_time = 0.5 * period - rise_time
        self.rise_time = rise_time

        self.pulse_width = period * 0.5 - fall_time
        self.fall_time = fall_time
        self.period = period

    def get_ngspice(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('ngspice/pulse_voltage_source.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'val0': self.val0,
                'val1': self.val1,
                'delay_time': 0.0,
                'pulse_width': self.pulse_width,
                'period': self.period,
                'rise_time': self.rise_time,
                'fall_time': self.fall_time,
            }
        )

    def get_spectre(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('spectre/pulse_voltage_source.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'val0': self.val0,
                'val1': self.val1,
                'period': self.pulse_width,
                'rise': self.rise_time,
                'fall': self.fall_time,
            }
        )


class SinusoidalVoltageSource(CircuitElement):
    offset: SPICE_VALUE
    amplitude: SPICE_VALUE
    frequency: SPICE_VALUE
    phase: SPICE_VALUE
    delay_time: SPICE_VALUE
    damping_factor: SPICE_VALUE

    def __init__(
        self,
        instance_name: str,
        offset: SPICE_VALUE,
        amplitude: SPICE_VALUE,
        frequency: SPICE_VALUE,
        phase: SPICE_VALUE = 0,
        delay_time: SPICE_VALUE = 0,
        damping_factor: SPICE_VALUE = 0,
        ac_gain: SPICE_VALUE = 1,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'V':
            instance_name = 'V' + instance_name

        super().__init__(
            instance_name,
            [Terminal(), Terminal()],
        )
        self.offset = offset
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.delay_time = delay_time
        self.damping_factor = damping_factor
        self.ac_gain = ac_gain

    def get_ngspice(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('ngspice/sine_voltage_source.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'offset': self.offset,
                'amplitude': self.amplitude,
                'frequency': self.frequency,
                'phase': self.phase,
                'delay': self.delay_time,
                'damping_factor': self.damping_factor,
                'ac_gain': self.ac_gain,
            }
        )

    def get_spectre(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('spectre/sine_voltage_source.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'offset': self.offset,
                'amplitude': self.amplitude,
                'frequency': self.frequency,
                'phase': self.phase,
                'delay': self.delay_time,
                'damping_factor': self.damping_factor,
                'ac_gain': self.ac_gain,
            }
        )
