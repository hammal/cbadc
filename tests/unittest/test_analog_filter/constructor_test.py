from tests.fixture.chain_of_integrators import chain_of_integrators
import numpy as np

from scipy.signal import StateSpace, bode


def test_initialization(chain_of_integrators):
    A = chain_of_integrators["A"]
    B = chain_of_integrators["B"]
    C = chain_of_integrators["C"]
    D = chain_of_integrators["D"]
    StateSpace(A, B, C, D)


def test_transfer_function(chain_of_integrators):
    A = chain_of_integrators["A"]
    B = chain_of_integrators["B"]
    C = chain_of_integrators["C"]
    D = chain_of_integrators["D"]
    system = StateSpace(A, B, C, D)
    print(system)


def test_bode(chain_of_integrators):
    A = chain_of_integrators["A"]
    B = chain_of_integrators["B"]
    C = chain_of_integrators["C"]
    D = chain_of_integrators["D"]
    for i in range(chain_of_integrators["M"] + 1):
        print(f"i = {i}")
        sub_system = StateSpace(A, B[:, i : i + 1], C[-1], D[-1, i : i + 1])
        # sub_system = sub_system.to_tf()
        print(sub_system)
        w, mag, phase = bode(sub_system)
        print(w, mag, phase)
