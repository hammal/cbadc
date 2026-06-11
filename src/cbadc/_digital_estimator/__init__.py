"""A selection of control-bounded digital estimators"""

from typing import Union

from ._filter_coefficients import FilterComputationBackend
from .adaptive_filter import AdaptiveFIRFilter, AdaptiveIIRFilter, batch
from .batch_estimator import BatchEstimator
from .decimation_and_demodulation import decimate, demodulate
from .fir_estimator import FIRFilter
from .iir_estimator import IIRFilter
from .nuv_estimator import NUVEstimator
from .parallel_digital_estimator import ParallelEstimator

_Estimators = Union[
    BatchEstimator,
    FIRFilter,
    IIRFilter,
    ParallelEstimator,
    NUVEstimator,
    AdaptiveFIRFilter,
]
