"""
The digital control is responsible for stabilizing the analog system.
"""
import numpy as np


class DigitalControl(object):

    def __init__(self, gammaTilde, controlPeriod, DACWaveForm=None, totalNumberOfBits=1):
        self._gammaTilde = np.array(gammaTilde)
        self.controlPeriod = controlPeriod
        self._totalNumberOfBits = totalNumberOfBits
        if not DACWaveForm:
            #  create square DAC waveform
            self._dacWaveform = lambda t: np.float(
                t >= 0 and controlPeriod - t > 0)
        else:
            self._dacWaveform = DACWaveForm
        self._T = controlPeriod
        self._holdState = np.zeros(self._as.state_space_order())
        self._nextSampleTime = 0.
        self._sampleTime = 0.
        self._controlSignal = np.zeros(self._gammaTilde.shape[0], dtype=np.int)

    def controlSignal(self, t, sample):
        [state, updateControl] = self.sampleAndHold(t, sample)
        # only compute when necessary
        if updateControl:
            self._controlSignal = self.quantizer(
                np.dot(self._gammaTilde, state))
        return (self._controlSignal, updateControl)

    def controlContribution(self):
        return lambda t: self._controlSignal * self._dacWaveform(t - self._sampleTime)

    def sampleAndHold(self, t, x):
        # When new sample
        if t >= self._nextSampleTime:
            # store the state
            self._holdState = x
            # advance to the next sampling time
            self._sampleTime = self._nextSampleTime
            self._nextSampleTime += self._T
            return (self._holdState, True)
        # return stored state
        return (self._holdState, False)

    def quantizer(self, vector, digitalCode=None, bitNumber=1, referenceValue=1, referenceThreshold=None):
        """The quantizer is impelmented as an algorithmic converter that 
        recursively calls itself until a codeword made up of totalNumberOfBits is pertained.
        The input vector might be a vector in which case the result is codeword vector
        of the same dimension. 


        :param vector: the sample vector to be converted
        :type vector: numpy array
        :param digitalCode: the initial digital code word, defaults to None
        :type digitalCode: integer numpy array, optional
        :param bitNumber: specify which bit that is converted, defaults to 1
        :type bitNumber: int, optional
        :param referenceValue: the reference value corresponding to the digital code word 1, defaults to 1
        :type referenceValue: int, optional
        :param referenceThreshold: the reference thresholds, defaults to None
        :type referenceThreshold: numpy array, optional
        :raises Exception: [description]
        :return: digital code word 
        :rtype: integer numpy array of same dim as vector
        """
        # initialize referenceThreshold
        if not referenceThreshold:
            referenceThreshold = np.zeros_like(vector)
        # ensure vector is a flat array
        if len(vector.shape) > 1:
            raise Exception(
                f'vector mush be a flat numpy array. you inserted %{vector} with shape ${vector.shape}')
        # ensure referenceThresholds and vector are of same dimension
        if vector.shape[0] != referenceThreshold.shape[0]:
            raise Exception(
                f'referenceThreshold vector dimension do not agree with vector')
        # if no digitalCode initialize as zero vector
        if not digitalCode:
            digitalCode = np.zeros_like(vector, dtype=np.int)
        # convert vector to +1,-1 bitVector
        bitVector = np.array(vector > np.zeros_like(vector)).flatten() * 2 - 1
        # add to digital codeword
        digitalCode += 1 << (self._totalNumberOfBits - bitNumber) * bitVector
        # check if more bits needs to be computed
        if bitNumber < self._totalNumberOfBits:
            # compute new vector
            newvector = 2 * vector - bitVector * referenceValue
            # recursively call self with new parameters
            return self.quantizer(newvector, digitalCode, bitNumber + 1, referenceValue)
        # return the digital codeword.
        return digitalCode
