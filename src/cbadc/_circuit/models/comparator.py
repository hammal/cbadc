from typing import List
from .. import DeviceModel, _template_env, SPICE_VALUE


class ADC_Bridge_Model(DeviceModel):
    ng_spice_model_name = 'adc_bridge'

    def __init__(
        self,
        model_name: str,
        in_low: SPICE_VALUE,
        in_high: SPICE_VALUE,
        rise_delay: SPICE_VALUE = 1e-9,
        fall_delay: SPICE_VALUE = 1e-9,
        comments: List[str] = [],
    ):
        super().__init__(
            model_name,
            comments=comments,
            in_low=in_low,
            in_high=in_high,
            rise_delay=rise_delay,
            fall_delay=fall_delay,
        )

    def get_ngspice(self) -> str:
        return _template_env.get_template('ngspice/model.cir.j2').render(
            {
                'model_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': self.parameters,
            }
        )


class DAC_Bridge_Model(DeviceModel):
    ng_spice_model_name = 'dac_bridge'

    def __init__(
        self,
        model_name: str,
        out_low: SPICE_VALUE,
        out_high: SPICE_VALUE,
        out_undef: SPICE_VALUE,
        input_load: SPICE_VALUE = 1e-12,
        t_rise: SPICE_VALUE = 1e-9,
        t_fall: SPICE_VALUE = 1e-9,
        comments: List[str] = [],
    ):
        super().__init__(
            model_name,
            comments=comments,
            out_low=out_low,
            out_high=out_high,
            out_undef=out_undef,
            input_load=input_load,
            t_rise=t_rise,
            t_fall=t_fall,
        )

    def get_ngspice(self) -> str:
        return _template_env.get_template('ngspice/model.cir.j2').render(
            {
                'model_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': self.parameters,
            }
        )


class D_Flip_Flop_Model(DeviceModel):
    ng_spice_model_name = 'd_dff'

    def __init__(
        self,
        model_name: str,
        clk_delay: SPICE_VALUE = 13e-9,
        set_delay: SPICE_VALUE = 25e-9,
        reset_delay: SPICE_VALUE = 27e-9,
        ic: SPICE_VALUE = 0,
        rise_delay: SPICE_VALUE = 10e-9,
        fall_delay: SPICE_VALUE = 3e-9,
        comments: List[str] = [],
    ):
        super().__init__(
            model_name,
            clk_delay=clk_delay,
            set_delay=set_delay,
            reset_delay=reset_delay,
            ic=ic,
            rise_delay=rise_delay,
            fall_delay=fall_delay,
            comments=comments,
        )

    def get_ngspice(self) -> str:
        return _template_env.get_template('ngspice/model.cir.j2').render(
            {
                'model_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': self.parameters,
            }
        )


class ClockedComparatorModel(DeviceModel):
    ng_spice_model_name = 'clocked_comparator'

    def __init__(
        self,
        model_name: str,
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
        comments: List[str] = [],
    ) -> None:
        super().__init__(
            model_name,
            in_low=in_low,
            in_high=in_high,
            out_low=out_low,
            out_high=out_high,
            out_undef=out_undef,
            input_load=input_load,
            clk_delay=clk_delay,
            clk_offset=clk_offset,
            t_rise=t_rise,
            t_fall=t_fall,
            t_tolerance=t_tolerance,
            comments=comments,
        )
        # indicate that this model has a verilog-ams implementation
        self.verilog_ams = True

        self.adc_bridge = ADC_Bridge_Model(
            'in_adc_bridge',
            self.parameters['in_low'],
            self.parameters['in_high'],
            rise_delay=t_rise,
            fall_delay=t_fall,
        )
        self.clk_adc_bridge = ADC_Bridge_Model(
            'clk_adc_bridge',
            self.parameters['clk_offset'],
            self.parameters['clk_offset'],
            rise_delay=t_rise,
            fall_delay=t_fall,
        )
        self.dac_bridge = DAC_Bridge_Model(
            'dac_bridge',
            self.parameters['out_low'],
            self.parameters['out_high'],
            self.parameters['out_undef'],
            input_load=self.parameters['input_load'],
            t_rise=self.parameters['t_rise'],
            t_fall=self.parameters['t_fall'],
        )
        self.flip_flop = D_Flip_Flop_Model(
            'd_flip_flop',
            clk_delay=self.parameters['clk_delay'],
        )

    def get_ngspice(self):
        return '\n'.join(
            [
                self.adc_bridge.get_ngspice(),
                self.clk_adc_bridge.get_ngspice(),
                self.dac_bridge.get_ngspice(),
                self.flip_flop.get_ngspice(),
            ]
        )

    def get_verilog_ams(self) -> str:
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


class DifferentialClockedComparatorModel(DeviceModel):
    ng_spice_model_name = 'clocked_comparator'

    def __init__(
        self,
        model_name: str,
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
        comments: List[str] = [],
    ) -> None:
        super().__init__(
            model_name,
            in_low=in_low,
            in_high=in_high,
            out_low=out_low,
            out_high=out_high,
            out_undef=out_undef,
            input_load=input_load,
            clk_delay=clk_delay,
            clk_offset=clk_offset,
            t_rise=t_rise,
            t_fall=t_fall,
            t_tolerance=t_tolerance,
            comments=comments,
        )
        # indicate that this model has a verilog-ams implementation
        self.verilog_ams = True

        self.adc_bridge = ADC_Bridge_Model(
            'in_adc_bridge',
            self.parameters['in_low'],
            self.parameters['in_high'],
            rise_delay=t_rise,
            fall_delay=t_fall,
        )
        self.clk_adc_bridge = ADC_Bridge_Model(
            'clk_adc_bridge',
            self.parameters['clk_offset'],
            self.parameters['clk_offset'],
            rise_delay=t_rise,
            fall_delay=t_fall,
        )
        self.dac_bridge_p = DAC_Bridge_Model(
            'dac_bridge',
            self.parameters['out_low'],
            self.parameters['out_high'],
            self.parameters['out_undef'],
            input_load=self.parameters['input_load'],
            t_rise=self.parameters['t_rise'],
            t_fall=self.parameters['t_fall'],
        )

        # self.dac_bridge_n = DAC_Bridge_Model(
        #     'n_dac_bridge',
        #     self.parameters['out_low'],
        #     self.parameters['out_high'],
        #     self.parameters['out_undef'],
        #     input_load=self.parameters['input_load'],
        #     t_rise=self.parameters['t_rise'],
        #     t_fall=self.parameters['t_fall'],
        # )
        self.flip_flop = D_Flip_Flop_Model(
            'd_flip_flop',
            clk_delay=self.parameters['clk_delay'],
        )

    def get_ngspice(self):
        return '\n'.join(
            [
                self.adc_bridge.get_ngspice(),
                self.clk_adc_bridge.get_ngspice(),
                self.dac_bridge_p.get_ngspice(),
                # self.dac_bridge_n.get_ngspice(),
                self.flip_flop.get_ngspice(),
            ]
        )

    def get_verilog_ams(self) -> str:
        raise NotImplementedError
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
