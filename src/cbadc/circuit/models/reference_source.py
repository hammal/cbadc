from typing import List
from .. import DeviceModel, _template_env


class ReferenceSourceModel(DeviceModel):
    ng_spice_model_name = 'd_source'

    def __init__(
        self,
        model_name: str,
        input_filename: str,
        comments: List[str] = [],
    ):
        super().__init__(
            model_name,
            comments=comments,
            input_file=input_filename,
        )

    def get_ngspice(self) -> str:
        return _template_env.get_template(
            'ngspice/reference_source_model.cir.j2'
        ).render(
            {
                'model_instance_name': self.model_name,
                'model_name': self.ng_spice_model_name,
                'parameters': self.parameters,
            }
        )
