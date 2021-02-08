from tests.test_analog_system.chain_of_integrators import chain_of_integrators
import numpy as np
import os
import sys

# this is python madness
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")
if myPath:
    print("hello")


def test_derivative(chain_of_integrators):
    x = np.zeros(chain_of_integrators["N"])
    t = 1.0

    def u(tau):
        return tau * np.ones(1)

    def s(tau):
        return np.zeros(chain_of_integrators["M"])

    analogSystem = chain_of_integrators["system"]
    analogSystem.derivative(x, t, u(t), s(t))
