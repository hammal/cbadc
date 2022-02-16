"""
An Analog System Model
======================

We demonstrate how an analog system can be transformed into a boilerplate
verilog-ams circuit model.
"""

import cbadc
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Setting up an Analog System
# -----------------------------
#
# We start by instantiating an analag system. In this case we use
# high level utilities functions from the :py:mod:`cbadc.specification`
# module.
#

# Fix system parameters such as effective number of bits
ENOB = 16
# System order
N = 4
# Bandwidth
BW = 1e6

# Instantiate leap-frog analog system is created as
analog_system, _ = cbadc.specification.get_leap_frog(ENOB=ENOB, N=N, BW=BW)

# Visualize the analog system's transfer functions
BW_log = np.log10(BW)
frequencies = np.logspace(BW_log - 2, BW_log + 1, 500)
omegas = 2 * np.pi * frequencies
transfer_function = analog_system.transfer_function_matrix(omegas)
plt.semilogx(
    frequencies,
    20 * np.log10(np.linalg.norm(transfer_function[:, 0, :], axis=0)),
    label="leap-frog $ ||\mathbf{G}(\omega)||_2 $",
)

# Add labels and legends to figure
plt.legend()
plt.grid(which="both")
plt.title("Analog system transfer function")
plt.xlabel("$f$ [Hz]")
plt.ylabel("dB")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()
plt.show()

###############################################################################
# Instantiating a Circuit Level Model
# -----------------------------------
#
# Using the :py:mod:`cbadc.circuit_level` module we can now start turning our
# nominal analog system into a circuit level system model.
#
# We will start with the highest layer of abstraction. Namely, the state space
# equations modeled directly using the verilog-ams language
#

# verilog_analog_system = cbadc.circuit_level.AnalogSystemStateSpaceEquations(
#     analog_system
# )

# # The verilog module description can be accessed by
# print(verilog_analog_system.render())

# # Alternatively, we can create a corresponding verilog-ams file as
# verilog_analog_system.to_file(filename="verilog_analog_system.vams")

# # The created file can be :download:`downloaded here <verilog_analog_system.vams>`

# ###############################################################################
# # Op-amps
# # -------
# #
# # Next we realize the same analog system using an ideal op-amp configuration
# # with capacitive feedback.
# #

# C = 1e-12
# ideal_op_amp_analog_system = cbadc.circuit_level.AnalogSystemIdealOpAmp(
#     analog_system, C
# )

# # The verilog module description can be accessed by
# print(ideal_op_amp_analog_system.render())

# # Alternatively, we can create a corresponding verilog-ams file as
# ideal_op_amp_analog_system.to_file(filename="ideal_op_amp_analog_system.vams")
