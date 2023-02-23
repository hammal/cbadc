import numpy as np


def signed_weight(x: float):
    magnitude = np.abs(x)
    sign = np.sign(x)
    if magnitude == 0.0:
        return ""
    else:
        return f"{sign} * {magnitude}"


from . import (
    comparator,
    integrator,
    observer,
    ota,
    reference_source,
    summer,
    voltage_buffer,
)
