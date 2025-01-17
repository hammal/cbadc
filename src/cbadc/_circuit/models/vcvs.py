from typing import List
from .. import DeviceModel, _template_env


class VoltageControlledVoltageSourceModel(DeviceModel):
    ng_spice_model_name = 'vcvs'

    def __init__(
        self,
        model_name: str,
        value: float,
        comments: List[str] = ['A voltage controlled voltage source'],
    ):
        super().__init__(model_name, comments=comments, value=value)
        # indicate that this model has a verilog-ams implementation
        self.verilog_ams = True

    def get_ngspice(self):
        return ''

    def get_verilog_ams(self):
        return _template_env.get_template('verilog_ams/vcvs.vams.j2').render(
            {
                'module_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': {
                    'value': float(self.parameters['value']),
                },
                'description': self.comments,
                'terminals': ['vdd', 'vdd', 'in_p', 'in_n', 'out_p', 'out_n'],
            }
        )
