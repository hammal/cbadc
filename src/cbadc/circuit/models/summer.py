from typing import List
from .. import DeviceModel, _template_env
from . import signed_weight


class SummerModel(DeviceModel):
    ng_spice_model_name = 'summer'

    def __init__(
        self,
        model_name: str,
        input_offset: List[float],
        input_gain: List[float],
        output_offset: float,
        output_gain: float,
        comments: List[str] = ['integrator'],
    ):
        if len(input_gain) != len(input_offset):
            raise ValueError('input_gain and input_offset must be the same size')

        super().__init__(
            model_name,
            comments=comments,
            input_offset=input_offset,
            input_gain=input_gain,
            output_offset=output_offset,
            output_gain=output_gain,
        )
        # indicate that this model has a verilog-ams implementation
        self.verilog_ams = True

    def get_ngspice(self):
        return _template_env.get_template('ngspice/model.cir.j2').render(
            {
                'model_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': {
                    'in_offset': self.parameters['input_offset'],
                    'in_gain': self.parameters['input_gain'],
                    'out_offset': self.parameters['output_offset'],
                    'out_gain': self.parameters['output_gain'],
                },
            }
        )

    def get_verilog_ams(self):
        return _template_env.get_template('verilog_ams/summer.vams.j2').render(
            {
                'module_instance_name': self.model_name,
                'inputs': [
                    {
                        'active': float(gain) != 0.0,
                        'magnitude': abs(float(gain)),
                        'sign': ['+', '-'][float(gain) < 0],
                        'offset': {
                            'active': float(offset) != 0.0,
                            'magnitude': abs(float(offset)),
                            'sign': ['+', '-'][float(offset) < 0],
                        },
                        'name': f'in_{index}',
                    }
                    for index, (gain, offset) in enumerate(
                        zip(
                            self.parameters['input_gain'],
                            self.parameters['input_offset'],
                        )
                    )
                ],
                'out_gain': self.parameters['output_gain'],
                'out_offset': self.parameters['output_offset'],
                'description': 'A weighted summer model',
                'terminals': [
                    f'in_{index}' for index in range(len(self.parameters['input_gain']))
                ]
                + ['out', 'vgnd'],
            }
        )
