"""
======================
Analog System
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
# equations modeled directly using the verilog-ams language. This can
# be done using the class :py:class:`cbadc.circuit_level.AnalogSystemStateSpaceEquations`
# as
#


verilog_analog_system = cbadc.circuit_level.AnalogSystemStateSpaceEquations(
    analog_system
)

# The verilog module description can be accessed by
print("\n\n\n".join(verilog_analog_system.render()[0]))

# Alternatively, we can create a corresponding verilog-ams file as
verilog_analog_system.to_file(filename="verilog_analog_system.vams")

###############################################################################
# :download:`verilog_analog_system.vams <verilog_analog_system.vams>`
#


###############################################################################
# Ideal Op-amp Implementation
# ----------------------------
#
# Next we realize the same analog system using an ideal op-amp configuration
# with capacitive feedback.
#

C = 1e-12
ideal_op_amp_analog_system = cbadc.circuit_level.AnalogSystemIdealOpAmp(
    analog_system, C
)

# The verilog module description can be accessed by
print("\n\n\n".join(ideal_op_amp_analog_system.render()[0]))

# Alternatively, we can create a corresponding verilog-ams file as
ideal_op_amp_analog_system.to_file(filename="ideal_op_amp_analog_system.vams")

###############################################################################
# :download:`ideal_op_amp_analog_system.vams <ideal_op_amp_analog_system.vams>`
#

###############################################################################
#
# Similarly, we can instantiate op-amp realizations that account for
# imperfections such as finite gain and first order pole.

C = 1e-12
A_DC = 1e2
omega_p = 2 * np.pi * BW / 8

finite_gain_op_amp_analog_system = cbadc.circuit_level.AnalogSystemFiniteGainOpAmp(
    analog_system, C, A_DC
)

first_order_pole_op_amp_analog_system = (
    cbadc.circuit_level.AnalogSystemFirstOrderPoleOpAmp(analog_system, C, A_DC, omega_p)
)

###############################################################################
#
# again the corresponding verilog files can be found below
#
# :download:`finite_gain_op_amp_analog_system.vams <finite_gain_op_amp_analog_system.vams>`,
# :download:`first_order_pole_op_amp_analog_system.vams <first_order_pole_op_amp_analog_system.vams>`
#
# and are generated using the :py:func:`cbadc.circuit_level.AnalogSystemFirstOrderPoleOpAmp.to_file`.
#

finite_gain_op_amp_analog_system.to_file(
    filename="finite_gain_op_amp_analog_system.vams"
)

first_order_pole_op_amp_analog_system.to_file(
    filename="first_order_pole_op_amp_analog_system.vams"
)

###############################################################################
# Reference Simulation
# ---------------------
#
# The primary intention of the :py:mod:`cbadc.circuit_level` module is to
# provide golden models, correctly parameterized and instantiated in the
# circuit level simulation domain. However, it might for some purposes be
# interesting to see the effects of circuit imperfections directly within
# the cbadc design tool itself.
#
# For this purpose each circuit level analog system, like the ones we
# have seen above, have an associated :py:class:`cbadc.analog_system.AnalogSystem`
# that also any discrepancies from the target analog system with which it
# was instantiated.
#
# Tom demonstarte this second use case we will next compare the transfer
# functions of the analog systems we covered previously.
#

# We don't consider the ideal_op_amp_analog_system and verilog_analog_system
# instances as these will result in identical analog systems as our target
# system.

compare_systems = {
    'target': analog_system,
    f'finite_gain, A_DC={A_DC:.0e}': finite_gain_op_amp_analog_system.analog_system,
    f'single_pole, A_DC={A_DC:.0e}, f_p={omega_p/(2 * np.pi):.0e} Hz': first_order_pole_op_amp_analog_system.analog_system,
}

for key, system in compare_systems.items():
    print(system)
    transfer_function = system.transfer_function_matrix(omegas)
    plt.semilogx(
        frequencies,
        20 * np.log10(np.linalg.norm(transfer_function[:, 0, :], axis=0)),
        label=f"{key}, " + "$ ||\mathbf{G}(\omega)||_2 $",
    )

# Add labels and legends to figure
plt.legend()
plt.grid(which="both")
plt.title("Analog system transfer function")
plt.xlabel("$f$ [Hz]")
plt.ylabel("dB")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()

# sphinx_gallery_thumbnail_number = 2
