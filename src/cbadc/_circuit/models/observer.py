from typing import List
from .. import DeviceModel, _template_env, SPICE_VALUE


class ObserverModel(DeviceModel):
    ng_spice_model_name = 'observer'

    def __init__(
        self,
        model_name: str,
        input_signal_names: List[str],
        trigger_offset: SPICE_VALUE = 0.5,
        save_on_falling_edge: bool = True,
        comments: List[str] = [],
        filename: str = 'observations.csv',
    ):
        self.input_signal_names = input_signal_names
        super().__init__(
            model_name,
            comments=comments,
            input_signal_names=input_signal_names,
            save_on_falling_edge=save_on_falling_edge,
            filename=filename,
            trigger_offset=trigger_offset,
        )
        # indicate that this model has a verilog-ams implementation
        self.verilog_ams = True

    def get_ngspice(self):
        return ''

    def get_verilog_ams(self):
        edge_direction = -1
        if not self.parameters['save_on_falling_edge']:
            edge_direction = 1
        terminal_names = ['clk'] + self.input_signal_names
        return _template_env.get_template('verilog_ams/observer.vams.j2').render(
            {
                'module_instance_name': self.model_name,
                'inputs': terminal_names,
                'edge_direction': edge_direction,
                'description': 'An observer module',
                'terminals': terminal_names,
                'filename': self.parameters['filename'],
                'trigger_offset': self.parameters['trigger_offset'],
                'csv_header': ','.join(self.input_signal_names),
            }
        )
