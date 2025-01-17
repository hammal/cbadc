from typing import List
from .. import DeviceModel, _template_env


class VoltageBufferModel(DeviceModel):
    ng_spice_model_name = 'voltage_buffer'

    def __init__(
        self,
        model_instance_name: str,
        comments: List[str] = ['A voltage buffer'],
    ):
        super().__init__(
            model_instance_name,
            comments=comments,
        )
        # indicate that this model has a verilog-ams implementation
        self.verilog_ams = True

    def get_ngspice(self):
        return ''

    def get_verilog_ams(self):
        return _template_env.get_template('verilog_ams/voltage_buffer.vams.j2').render(
            {
                'module_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'description': self.comments,
                'terminals': ['vdd', 'vdd', 'in_p', 'in_n', 'out_p', 'out_n'],
            }
        )
