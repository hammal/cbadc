from typing import Dict, List
from .. import (
    SPICE_VALUE,
    Port,
    Terminal,
    CircuitElement,
    _template_env,
)
from ..models.reference_source import ReferenceSourceModel
import numpy as np
import os


class ReferenceSource(CircuitElement):
    _input_filename: str

    def __init__(
        self,
        instance_name: str,
        model_name: str,
        number_of_sources: int,
        input_filename: str,
        number_of_samples: int = 1 << 18,
        time_step: SPICE_VALUE = 1e-9,
        ternary: bool = True,
    ):
        if not instance_name or not isinstance(instance_name, str):
            raise TypeError(f"Expected str, got {type(instance_name)}")
        elif instance_name[0] != "A":
            instance_name = "A" + instance_name

        super().__init__(
            instance_name,
            [Terminal(f"S_{i}") for i in range(number_of_sources)],
        )
        self.model = ReferenceSourceModel(
            model_name,
            input_filename,
        )
        # generate random signals
        t = np.array(
            np.arange(0, number_of_samples * time_step, time_step), dtype=str
        ).reshape((number_of_samples, 1))

        def ternary_mapper(x):
            if x == 0:
                return "Us"
            elif x == 1:
                return "0s"
            elif x == 2:
                return "1s"
            else:
                raise ValueError(f"x = {x}")

        def binary_mapper(x):
            if x == 0:
                return "0s"
            elif x == 1:
                return "1s"
            else:
                raise ValueError(f"x = {x}")

        if ternary:
            random_signals = np.array(
                list(
                    map(
                        ternary_mapper,
                        np.random.randint(0, 3, number_of_sources * t.size),
                    )
                )
            ).reshape((t.size, number_of_sources))
        else:
            random_signals = np.array(
                list(
                    map(
                        binary_mapper,
                        np.random.randint(0, 2, number_of_sources * t.size),
                    )
                )
            ).reshape((t.size, number_of_sources))
        data = np.hstack((t, random_signals))

        self._input_filename = input_filename

        if not os.path.isfile(self._input_filename):
            with open(self._input_filename, "w") as f:
                f.write(
                    "\n".join([f"{' '.join([value for value in row])}" for row in data])
                )
        else:
            print(f"{self._input_filename} already exists and has not been replaced.")

    def get_ngspice(self, connections: Dict[Terminal, Port]) -> str:
        return _template_env.get_template("ngspice/xspice.cir.j2").render(
            {
                "instance_name": self.instance_name,
                "terminals": self._reference_source_get_terminal_names(connections),
                "parameters": self._parameters_dict,
                "comments": self.comments,
                "model_instance_name": self.model.model_name,
            }
        )

    def _reference_source_get_terminal_names(
        self, connections: Dict[Terminal, Port]
    ) -> List[str]:
        named_nodes = self._get_terminal_names(connections)
        return [f'[{" ".join(named_nodes)}]']

    def get_spectre(self, connections: Dict[Terminal, Port]):
        raise NotImplementedError()

    def __del__(self):
        if os.path.isfile(self._input_filename):
            os.remove(self._input_filename)
        # super().__del__()
