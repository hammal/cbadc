"""digital estimator calibration.

this module contains tools for calibrating the digital estimator.
"""
from cbadc.digital_estimator import BatchEstimator
import logging

logger = logging.getLogger(__name__)


class Calibrator:
    def __init__(self, BatchEstimator):
        self.BatchEstimator = BatchEstimator

    def cost_function(self):
        pass


class InputReference:
    def __init__(
        self,
    ):
        _ = BatchEstimator
