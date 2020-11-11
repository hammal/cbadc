
import numpy as np
from scipy.integrate import solve_ivp


class Simulator(object):

    def __init__(self, analogSystem, digitalControl, input, controlPeriod, noise):

        if analogSystem.number_of_inputs() != len(input):
            raise "The analog system does not have as many inputs as in input list"
        self._as = analogSystem,
        self._dc = digitalControl
        self._input = input

    def simulate(self, t, initialState):
        timeInterval = (t[0], t[-1])
        # make numpy array to store control signals
        # (#TimeSteps, #controls)
        controlSignals = np.zeros(
            (int((timeInterval[1] - timeInterval[0]) / self._dc.controlPeriod), self._ac.number_of_controls()), dtype=np.int)
        timeIndex = 0

        def controlContribution(t, x):
            # Check with digital control for new control signal
            [controlSignal, update] = self._dc.controlSignal(t, x)
            if update:
                controlSignals[timeIndex, :] = controlSignal
            return self._dc.controlContribution()

        def signalContribution(t): return np.array(
            [u.eval(t) for u in self._input])
        def derivative(t, x): return self._as.derivative(
            x, t, signalContribution, controlContribution(t, x))
        stateTrajectory = solve_ivp(
            derivative, timeInterval, initialState, t_eval=t, vectorized=True)

        return {
            controlSignals: controlSignals,
            stateTrajectory: stateTrajectory['y'],
            t: t,
            success: stateTrajectory['success'],
            message: stateTrajectory['message'],
            t_events: stateTrajectory['t_events'],
            x_events: stateTrajectory['y_events']
        }
