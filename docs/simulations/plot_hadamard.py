"""
Loading a Hadamard Ramp Simulation
==================================

Builds on...
"""
import cbadc
import cbadc.examples.HadamardPCB

###############################################################################
# Create a Simulation Wrapper
# ----------------------------
#
# We load the PCB A prototype by instantiating
# the wrapper class as

simulation_wrapper = cbadc.examples.HadamardPCB('B')

###############################################################################
# Load a specific simulation
# ---------------------------
#
# In this case we load
# :py:func:`cbadc.examples.HadamardPCB.simulation_ramp_1_B`
# simulation by invoking

control_signal, ideal_control_signal, simulator, size = simulation_wrapper.simulation_ramp_1_B()

for i in range(10):
    print(next(control_signal))
