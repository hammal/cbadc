"""digital estimator calibration.

this module contains tools for calibrating the digital estimator.
"""
from cbadc.digital_estimator import DigitalEstimator
import logging

logger = logging.getLogger(__name__)


class Calibrator:
    def __init__(self, digitalEstimator):
        self.digitalEstimator = digitalEstimator

    def cost_function(self):
        pass


class InputReference:
    def __init__(
        self,
    ):
        a = DigitalEstimator
