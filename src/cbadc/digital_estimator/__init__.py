"""A selection of control-bounded digital estimators
"""
from .batch_estimator import BatchEstimator
from .fir_estimator import (
    FIRFilter,
    AdaptiveFIRFilter,
    initial_filter,
    initial_wiener_filter,
    FixedStepSize,
    ExponentialStepSize,
    PolynomialStepSize,
    StepSize,
)
from .iir_estimator import IIRFilter
from .parallel_digital_estimator import ParallelEstimator
from .nuv_estimator import NUVEstimator
from ._filter_coefficients import FilterComputationBackend
from typing import Union
from .adaptive_filter import AdaptiveFilter

_Estimators = Union[
    BatchEstimator,
    FIRFilter,
    IIRFilter,
    ParallelEstimator,
    NUVEstimator,
    AdaptiveFIRFilter,
]
