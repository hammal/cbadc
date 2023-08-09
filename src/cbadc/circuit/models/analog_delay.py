from typing import List
from .. import DeviceModel, _template_env, SPICE_VALUE


class AnalogDelayModel(DeviceModel):
    ng_spice_model_name = 'delay'

    def __init__(
        self,
        model_name: str,
        delay: SPICE_VALUE,
        buffer_size: int,
        has_delay_cnt: bool = False,
        comments: List[str] = [],
    ):
        super().__init__(
            model_name,
            comments=comments,
            delay=delay,
            buffer_size=buffer_size,
            has_delay_cnt=has_delay_cnt,
        )

    def get_ngspice(self) -> str:
        return _template_env.get_template('ngspice/model.cir.j2').render(
            {
                'model_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': self.parameters,
            }
        )
