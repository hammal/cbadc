from typing import Dict, List, Set
from .. import (
    CircuitElement,
    SPICE_VALUE,
    DeviceModel,
    Port,
    Terminal,
    SubCircuitElement,
    _template_env,
)
from ..models.comparator import (
    ADC_Bridge_Model,
    D_Flip_Flop_Model,
    DAC_Bridge_Model,
)
from . import ngspice_vector_terminal_vector_vector


class ADCBridgeRelative(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        in_low: SPICE_VALUE,
        in_high: SPICE_VALUE,
        rise_delay: SPICE_VALUE = 1e-9,
        fall_delay: SPICE_VALUE = 1e-9,
        comments: List[str] = [],
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'A':
            instance_name = 'A' + instance_name

        super().__init__(
            instance_name,
            [
                Terminal('IN'),
                Terminal('OUT'),
                Terminal('VCM'),
            ],
            comments=comments,
        )
        self.model = ADC_Bridge_Model(
            model_name,
            in_low=in_low,
            in_high=in_high,
            rise_delay=rise_delay,
            fall_delay=fall_delay,
            comments=comments,
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]) -> str:
        return _template_env.get_template('ngspice/xspice.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._adc_bridge_get_terminal_names(connections),
                'parameters': self._parameters_dict,
                'comments': self.comments,
                'model_instance_name': self.model.model_name,
            }
        )

    def _adc_bridge_get_terminal_names(
        self, connections: Dict[Terminal, Port]
    ) -> List[str]:
        named_nodes = self._get_terminal_names(connections)
        return [f'[%vd({named_nodes[0]},{named_nodes[2]})]', f'[{named_nodes[1]}]']


class ADCBridgeAbsolute(ADCBridgeRelative):
    def _adc_bridge_get_terminal_names(
        self, connections: Dict[Terminal, Port]
    ) -> List[str]:
        named_nodes = self._get_terminal_names(connections)
        return [f'[{named_nodes[0]}]', f'[{named_nodes[1]}]']


class DAC_Bridge(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        out_low: SPICE_VALUE,
        out_high: SPICE_VALUE,
        out_undef: SPICE_VALUE,
        input_load: SPICE_VALUE = 1e-12,
        t_rise: SPICE_VALUE = 1e-9,
        t_fall: SPICE_VALUE = 1e-9,
        comments: List[str] = [],
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'A':
            instance_name = 'A' + instance_name

        super().__init__(
            instance_name,
            [Terminal('IN'), Terminal('OUT')],
            comments=comments,
        )
        self.model = DAC_Bridge_Model(
            model_name,
            out_low=out_low,
            out_high=out_high,
            out_undef=out_undef,
            input_load=input_load,
            t_rise=t_rise,
            t_fall=t_fall,
            comments=comments,
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]) -> str:
        return _template_env.get_template('ngspice/xspice.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': ngspice_vector_terminal_vector_vector(
                    self._get_terminal_names(connections)
                ),
                'parameters': self._parameters_dict,
                'comments': self.comments,
                'model_instance_name': self.model.model_name,
            }
        )


class D_FLIP_FLOP(CircuitElement):
    def __init__(
        self,
        instance_name: str,
        model_name: str,
        clk_delay: SPICE_VALUE = 0.0,
        set_delay: SPICE_VALUE = 0.0,
        reset_delay: SPICE_VALUE = 0.0,
        ic: SPICE_VALUE = 2,
        rise_delay: SPICE_VALUE = 0.0,
        fall_delay: SPICE_VALUE = 0.0,
        comments: List[str] = [],
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f'Expected str, got {type(instance_name)}')
        elif instance_name[0] != 'A':
            instance_name = 'A' + instance_name

        super().__init__(
            instance_name,
            [
                Terminal('IN'),
                Terminal('CLK'),
                Terminal('SET'),
                Terminal('RES'),
                Terminal('F_OUT_P'),
                Terminal('F_OUT_N'),
            ],
            comments=comments,
        )
        self.model = D_Flip_Flop_Model(
            model_name,
            clk_delay=clk_delay,
            set_delay=set_delay,
            reset_delay=reset_delay,
            ic=ic,
            rise_delay=rise_delay,
            fall_delay=fall_delay,
            comments=comments,
        )

    def get_ngspice(self, connections: Dict[Terminal, Port]) -> str:
        return _template_env.get_template('ngspice/xspice.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'parameters': self._parameters_dict,
                'comments': self.comments,
                'model_instance_name': self.model.model_name,
            }
        )


class ClockedComparator(SubCircuitElement):
    def __init__(
        self,
        instance_name: str,
        sub_ckt_name: str,
        in_low: SPICE_VALUE,
        in_high: SPICE_VALUE,
        out_low: SPICE_VALUE,
        out_high: SPICE_VALUE,
        out_undef: SPICE_VALUE,
        input_load: SPICE_VALUE = 1e-12,
        clk_delay: SPICE_VALUE = 0.0,
        clk_offset: SPICE_VALUE = 0.6,
        t_tolerance: SPICE_VALUE = 0.0,
        t_rise: SPICE_VALUE = 1e-9,
        t_fall: SPICE_VALUE = 1e-9,
        rise_delay: SPICE_VALUE = 1e-15,
        fall_delay: SPICE_VALUE = 1e-15,
        set_delay: SPICE_VALUE = 1e-15,
        reset_delay: SPICE_VALUE = 1e-15,
        comments: List[str] = [],
    ):
        super().__init__(
            instance_name,
            sub_ckt_name,
            [Terminal('CLK'), Terminal('VCM'), Terminal('IN'), Terminal('OUT')],
        )

        self.add(
            ADCBridgeRelative(
                'Aclk_adc',
                'adc',
                in_low,
                in_high,
                rise_delay=rise_delay,
                fall_delay=fall_delay,
            ),
            ADCBridgeRelative(
                'Ain_adc',
                'adc',
                in_low,
                in_high,
                rise_delay=rise_delay,
                fall_delay=fall_delay,
            ),
            D_FLIP_FLOP(
                'Adffp',
                'dflip',
                clk_delay,
                set_delay=set_delay,
                reset_delay=reset_delay,
                ic=1,
                rise_delay=rise_delay,
                fall_delay=fall_delay,
            ),
            DAC_Bridge(
                'Adac',
                'dac',
                out_low,
                out_high,
                out_undef,
                input_load,
                t_rise,
                t_fall,
            ),
        )

        CLKD = Terminal('CLKD')

        self.connects(
            (self['CLK'], self.Aclk_adc['IN']),
            (self['VCM'], self.Aclk_adc['VCM']),
            (self['IN'], self.Ain_adc['IN']),
            (self['VCM'], self.Ain_adc['VCM']),
            (self['OUT'], self.Adac['OUT']),
            (CLKD, self.Adffp['CLK']),
            (CLKD, self.Aclk_adc['OUT']),
            (self.Ain_adc['OUT'], self.Adffp['IN']),
            (self.Adffp['F_OUT_P'], self.Adac['IN']),
        )

    # overwrite the get_model_set method to return the model set
    def _get_model_set(self, verilog_ams=False) -> List[DeviceModel]:
        if verilog_ams:
            return [self.get_verilog_ams()]
        else:
            return super()._get_model_set(verilog_ams)

    def get_verilog_ams(self) -> DeviceModel:
        raise NotImplementedError()
        return _template_env.get_template(
            'verilog_ams/clocked_comparator.vams.j2'
        ).render(
            {
                'module_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': self.parameters,
                'description': 'A ternary clocked comparator',
                'terminals': ['clk', 'in', 'out'],
            }
        )

    def get_spectre(self, connections: Dict[Terminal, Port]):
        return _template_env.get_template('spectre/verilog_ams.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'comments': self.comments,
                'model_instance_name': self.model.model_name,
            }
        )

    def _get_spectre_sub_circuit_definition(self) -> List[str]:
        # no subcircuit for verilog ams spectre.
        return []


class DifferentialOutputClockedComparator(SubCircuitElement):
    def __init__(
        self,
        instance_name: str,
        sub_ckt_name: str,
        in_low: SPICE_VALUE,
        in_high: SPICE_VALUE,
        out_low: SPICE_VALUE,
        out_high: SPICE_VALUE,
        out_undef: SPICE_VALUE,
        input_load: SPICE_VALUE = 1e-12,
        clk_delay: SPICE_VALUE = 0.0,
        clk_offset: SPICE_VALUE = 0.6,
        t_tolerance: SPICE_VALUE = 0.0,
        t_rise: SPICE_VALUE = 1e-15,
        t_fall: SPICE_VALUE = 1e-15,
        rise_delay: SPICE_VALUE = 1e-15,
        fall_delay: SPICE_VALUE = 1e-15,
        set_delay: SPICE_VALUE = 1e-15,
        reset_delay: SPICE_VALUE = 1e-15,
        comments: List[str] = [],
    ):
        super().__init__(
            instance_name,
            sub_ckt_name,
            [
                Terminal('CLK'),
                Terminal('IN'),
                Terminal('OUT_P'),
                Terminal('OUT_N'),
                Terminal('VCM'),
            ],
        )

        self.add(
            ADCBridgeRelative(
                'Aclk_adc',
                'adc',
                in_low,
                in_high,
                rise_delay=rise_delay,
                fall_delay=fall_delay,
            ),
            ADCBridgeRelative(
                'Ain_adc',
                'adc',
                in_low,
                in_high,
                rise_delay=rise_delay,
                fall_delay=fall_delay,
            ),
            D_FLIP_FLOP(
                'Adffp',
                'dflip',
                clk_delay,
                set_delay=set_delay,
                reset_delay=reset_delay,
                ic=1,
                rise_delay=rise_delay,
                fall_delay=fall_delay,
            ),
            DAC_Bridge(
                'Adac_p',
                'dac',
                out_low,
                out_high,
                out_undef,
                input_load,
                t_rise,
                t_fall,
            ),
            DAC_Bridge(
                'Adac_n',
                'dac',
                out_low,
                out_high,
                out_undef,
                input_load,
                t_rise,
                t_fall,
            ),
        )

        CLKD = Terminal('CLKD')
        self.connects(
            (self['CLK'], self.Aclk_adc['IN']),
            (self['VCM'], self.Aclk_adc['VCM']),
            (self['IN'], self.Ain_adc['IN']),
            (self['VCM'], self.Ain_adc['VCM']),
            (self['OUT_P'], self.Adac_p['OUT']),
            (self['OUT_N'], self.Adac_n['OUT']),
            (CLKD, self.Adffp['CLK']),
            (CLKD, self.Aclk_adc['OUT']),
            (self.Ain_adc['OUT'], self.Adffp['IN']),
            (self.Adffp['F_OUT_P'], self.Adac_p['IN']),
            (self.Adffp['F_OUT_N'], self.Adac_n['IN']),
        )

    # overwrite the get_model_set method to return the model set
    def _get_model_set(self, verilog_ams=False) -> List[DeviceModel]:
        if verilog_ams:
            return [self.get_verilog_ams()]
        else:
            return super()._get_model_set(verilog_ams)

    def get_spectre(self, connections: Dict[Terminal, Port]):
        raise NotImplementedError
        return _template_env.get_template('spectre/verilog_ams.cir.j2').render(
            {
                'instance_name': self.instance_name,
                'terminals': self._get_terminal_names(connections),
                'comments': self.comments,
                'model_instance_name': self.model.model_name,
            }
        )

    def _get_spectre_sub_circuit_definition(self) -> List[str]:
        # no subcircuit for verilog ams spectre.
        return []

    def get_verilog_ams(self) -> DeviceModel:
        raise NotImplementedError()
        return _template_env.get_template(
            'verilog_ams/clocked_comparator.vams.j2'
        ).render(
            {
                'module_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': self.parameters,
                'description': 'A ternary clocked comparator',
                'terminals': ['clk', 'in', 'out'],
            }
        )
