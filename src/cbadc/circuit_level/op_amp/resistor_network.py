"""resistor network"""
from typing import List
from ..module import Module, Wire
import logging
import numpy as np


logger = logging.getLogger(__name__)


class ResistorNetwork(Module):
    """A resistive network

    Describes resistive connections between inputs and outputs.
    For convenience all connections are expressed using their
    corresponding admittance value.

    Parameters
    ----------
    module_name: `str`
        the name of the module.
    instance_name: `str`
        the name of the particular instance.
    G: array_like, shape=(M, N)
        admittance matrix expressing the
        conductance relations between the N
        inputs and M outputs of the network.
    """

    G: np.ndarray

    def __init__(self, module_name: str, instance_name: str, G: np.ndarray):
        inputs = [Wire(f"in_{i}", True, True, True) for i in range(G.shape[1])]
        outputs = [Wire(f"out_{i}", True, True, True) for i in range(G.shape[0])]
        ports = [*inputs, *outputs]
        nets = ports
        analog_statements = []
        # G[G == np.inf] = 0
        # G[G == -np.inf] = 0
        # self.G = np.nan_to_num(G)
        self.G = G
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                if G[i, j] != 0:
                    analog_statements.append(
                        f"I(in_{j}, out_{i}) <+ {G[i, j]} * V(in_{j},out_{i});"
                    )
        super().__init__(
            module_name, nets, ports, instance_name, analog_statements=analog_statements
        )

    def _module_comment(self) -> List[str]:
        return [
            *super()._module_comment(),
            "",
            "Functional Description:",
            "",
            "Resistor network connecting inputs and outputs according to the following matrix",
            "",
            *[
                f"[out_{i}] \u2248 [{', '.join([f'{1.0/a:.2e}' for a in self.G[i, :]])}] [in_{i}]"
                for i in range(self.G.shape[0])
            ],
            "",
            "note the resistors are specified by their resistive values in Ohms",
        ]
