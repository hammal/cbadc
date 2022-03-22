from .digital_control import DigitalControl, _ImpulseResponse, StepResponse, _valid_clock_types
from typing import List


class MultiLevelDigitalControl(DigitalControl):
    """
    Number of levels should equal M_tilde!
    """
    def __init__(self,
                clock: _valid_clock_types,
                M: int,
                number_of_levels: List[int],
                t0: float = 0.0,
                impulse_response: _ImpulseResponse = StepResponse()):
        super().__init__(clock, M, t0, impulse_response)
        self.number_of_levels = number_of_levels

