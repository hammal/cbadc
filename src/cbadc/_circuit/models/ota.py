from typing import List
from .. import DeviceModel, _template_env


class OTAModel(DeviceModel):
    ng_spice_model_name = 'ota'

    def __init__(
        self,
        model_name: str,
        gm: float,
        comments: List[str] = ['A simplistic differential transconductance amplifier'],
    ):
        super().__init__(model_name, comments=comments, gm=gm)
        # indicate that this model has a verilog-ams implementation
        self.verilog_ams = True

    def get_ngspice(self):
        return ''

    def get_verilog_ams(self):
        return _template_env.get_template('verilog_ams/ota.vams.j2').render(
            {
                'module_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': {
                    'gm': float(self.parameters['gm']),
                },
                'description': self.comments,
                'terminals': ['vdd', 'vdd', 'in_p', 'in_n', 'out_p', 'out_n'],
            }
        )
