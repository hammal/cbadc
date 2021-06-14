"""digital estimator calibration.

this module contains tools for calibrating the digital estimator.
"""
from cbadc.digital_estimator import DigitalEstimator


class Calibrator:

    def __init__(self, digitalEstimator):
        self.digitalEstimator = digitalEstimator

    def cost_function(self):
        pass


class InputReference:

    def __init__(self, ):
        a = DigitalEstimator

