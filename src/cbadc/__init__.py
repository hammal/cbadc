"""State-space modelling, simulation, and measurement of A/D converters.

:class:`Frontend` holds the clocked ``A, B, C, D`` model. Simulation returns
decisions, inputs, and a digital-twin readout. :mod:`circuit` provides gm-C
and active-RC realizations; :mod:`noise`, :mod:`estimator`, :mod:`snr`, and
:mod:`fom` support analysis. :mod:`delsig` is an optional synthesis bridge.
"""

from typing import TYPE_CHECKING

from . import circuit, estimator, fom, frontend, noise, snr
from .__version__ import __version__
from .circuit import ActiveRC, GmC
from .estimator import StateSpaceFilter, WienerFilter
from .frontend import (
    DAC,
    SOS,
    Constant,
    Discretization,
    Exp,
    Filter,
    Frontend,
    Pole,
    Poly,
    Samples,
    Sinusoid,
    StateSpace,
    cascade,
    parallel,
)
from .noise import (
    combine_densities,
    discrete_process_noise_cov,
    kTC_rms,
    ota_density,
    per_state_intensity,
    positive_semidefinite,
    resistor_density,
)

if TYPE_CHECKING:
    from . import delsig


def __getattr__(name: str):
    """Load the optional delta-sigma bridge only when requested.

    Parameters
    ----------
    name : str
        Package attribute name to resolve.

    Returns
    -------
    module
        Lazily imported optional ``delsig`` module.
    """
    if name == "delsig":
        from importlib import import_module

        return import_module(".delsig", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActiveRC",
    "Constant",
    "DAC",
    "Discretization",
    "Exp",
    "Filter",
    "Frontend",
    "GmC",
    "Pole",
    "Poly",
    "SOS",
    "Samples",
    "Sinusoid",
    "StateSpace",
    "StateSpaceFilter",
    "WienerFilter",
    "__version__",
    "cascade",
    "combine_densities",
    "circuit",
    "delsig",
    "discrete_process_noise_cov",
    "estimator",
    "fom",
    "frontend",
    "kTC_rms",
    "noise",
    "ota_density",
    "parallel",
    "per_state_intensity",
    "positive_semidefinite",
    "resistor_density",
    "snr",
]
