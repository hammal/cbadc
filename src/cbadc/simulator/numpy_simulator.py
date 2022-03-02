"""A dummy simulator for turning numpy arrays into simulators
"""
import numpy as np


class NumpySimulator:
    """turns a numpy array into an simulator

    Parameters
    ----------
    filename: `str`
        the location of the numpy array .npz file to be used.
    array: array_like
        an already loaded numpy array, defaults to None. If
        array is not None the filename is ignored.
    """

    def __init__(self, filename: str, array: np.ndarray = None):
        if array is None:
            self.array = False
            self.filename = filename
        else:
            self.array = True
            self._control_signals = array
        self.reset()

    def __next__(self):
        return next(self.__iter__)

    def reset(self, t: float = 0.0):
        if self.array:
            self.__iter__ = iter(self._control_signals)
        else:
            self.__iter__ = iter(np.load(self.filename))
