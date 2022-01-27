"""Report system for states."""
import numpy as np


class StateBounds:
    def __init__(self, bounds: np.ndarray):

        self._N = bounds.shape[0]
        self._outage = []
        self._recovery = []

        if len(bounds.shape) > 1:
            # positive and negative bounds
            self._bounds_upper = bounds[:, 0]
            self._bounds_lower = bounds[:, 1]
        else:
            self._bounds_upper = np.abs(bounds[:])
            self._bounds_lower = -self._bounds_upper

    def validate(self, state):
        res = np.zeros(self._N)
        for n in range(self._N):
            if state[n] > 0:
                res[n] = state - self._bounds_upper
            else:
                res[n] = -(state - self._bounds_lower)
        return res

    def report_outage(self, event):
        self._outage.append(event)

    def outages(self):
        return self._outage[:]

    def report_recovery(self, event):
        self._recovery.append(event)

    def recoveries(self):
        return self._recovery[:]
