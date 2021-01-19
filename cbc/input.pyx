import numpy as np


class Input(object):
    def __init__(self, constant=0):
        self._c = constant

    @staticmethod
    def typeOfInput():
        return "default"

    def eval(self, t):
        return self._c


class Sinusodial(Input):
    def __init__(self, amplitude, frequency, phase=0, offset=0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.angluarFrequency = 2 * np.pi * self.frequency
        self.phase = phase
        self.offset = offset

    @staticmethod
    def typeOfInput():
        return "sinusodial"

    def eval(self, t):
        return (
            self.amplitude * np.sin(self.angluarFrequency * t + self.phase)
            + self.offset
        )


class GaussianNoise(Input):
    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
        self._std = np.sqrt(variance)

    @staticmethod
    def typeOfInput():
        return "gaussian_noise"

    def eval(self, t):
        return self._std * np.random.randn()


class FirstOrderHold(Input):
    def __init__(self, t, samples):
        self.samples = samples
        self.t = t

    @staticmethod
    def typeOfInput():
        return "first_order_hold"

    def eval(self, t):
        temp = np.nonzero(t > self.t)
        timeIndex = temp[0] - 1 if temp[0] > 1 else 0
        return self.samples[timeIndex]
