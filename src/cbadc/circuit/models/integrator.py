from typing import List
from .. import DeviceModel, _template_env
from . import signed_weight


class IntegratorModel(DeviceModel):
    ng_spice_model_name = 'int'

    def __init__(
        self,
        model_name: str,
        in_offset: float = 0.0,
        gain: float = 1.0,
        out_lower_limit: float = -10.0,
        out_upper_limit: float = 10.0,
        limit_range: float = 1e-6,
        out_ic: float = 0.0,
        comments: List[str] = ['integrator'],
    ):
        super().__init__(
            model_name,
            comments=comments,
            in_offset=in_offset,
            gain=gain,
            out_lower_limit=out_lower_limit,
            out_upper_limit=out_upper_limit,
            out_ic=out_ic,
            limit_range=limit_range,
        )
        # indicate that this model has a verilog-ams implementation
        self.verilog_ams = True

    def get_ngspice(self):
        return _template_env.get_template('ngspice/model.cir.j2').render(
            {
                'model_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': self.parameters,
            }
        )

    def get_verilog_ams(self):
        return _template_env.get_template('verilog_ams/integrator.vams.j2').render(
            {
                'module_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': {
                    'in_offset': {
                        'active': float(self.parameters['in_offset']) != 0.0,
                        'magnitude': float(self.parameters['in_offset']),
                        'sign': ['+', '-'][float(self.parameters['in_offset']) < 0],
                    },
                    'gain': float(self.parameters['gain']),
                    'out_lower_limit': signed_weight(
                        float(self.parameters['out_lower_limit'])
                    ),
                    'out_upper_limit': signed_weight(
                        float(self.parameters['out_upper_limit'])
                    ),
                    'limit_range': signed_weight(float(self.parameters['limit_range'])),
                    'out_ic': {
                        'magnitude': abs(float(self.parameters['out_ic'])),
                        'sign': ['+', '-'][float(self.parameters['out_ic']) < 0],
                    },
                },
                'description': 'A simplistic integrator model',
                'terminals': ['in', 'out'],
            }
        )
