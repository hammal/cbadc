"""
======================
Analog Frontend
======================

In this tutorial we will combine analog systems and digital controls
to which we refer to as an analog frontend. The analog frontend, in
contrast to the digital estimator, contains mixed signals (analog and digital)
and is therefore fundamentally different to model compared to the
digital estimator itself.

To create a analog fronted we require an analog system
to that end we use a similair specification as in
:doc:`./plot_a_analog_system`.
"""

import cbadc
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Analog System
# ---------------

# Fix system parameters such as effective number of bits
ENOB = 16
# System order
N = 4
# Bandwidth
BW = 1e6

# Instantiate leap-frog analog system is created as
analog_frontend_target = cbadc.synthesis.get_leap_frog(ENOB=ENOB, N=N, BW=BW)
target_analog_system = analog_frontend_target.analog_system
target_digital_control = analog_frontend_target.digital_control

verilog_analog_system = cbadc.circuit_level.AnalogSystemStateSpaceEquations(
    target_analog_system
)

C = 1e-12
ideal_op_amp_analog_system = cbadc.circuit_level.AnalogSystemIdealOpAmp(
    target_analog_system, C
)

A_DC = 1e2
omega_p = 2 * np.pi * BW

finite_gain_op_amp_analog_system = cbadc.circuit_level.AnalogSystemFiniteGainOpAmp(
    target_analog_system, C, A_DC
)

first_order_pole_op_amp_analog_system = (
    cbadc.circuit_level.AnalogSystemFirstOrderPoleOpAmp(
        target_analog_system, C, A_DC, omega_p
    )
)

###############################################################################
# Digital Control
# -----------------------------
#
# To create a verilog-ams digital control model
# we use :py:class:`cbadc.circuit_level.digital_control.DigitalControl` class
# as

verilog_digital_control = cbadc.circuit_level.DigitalControl(target_digital_control)

###############################################################################
# Analog Frontend
# ------------------
#
# Subsequently, the analog frontend can be pertained by the cbadc.circuit_level.AnalogFrontend
# class as

verilog_analog_frontend = cbadc.circuit_level.AnalogFrontend(
    verilog_analog_system, verilog_digital_control
)

ideal_op_amp_analog_frontend = cbadc.circuit_level.AnalogFrontend(
    ideal_op_amp_analog_system, verilog_digital_control
)

finite_gain_op_amp_analog_frontend = cbadc.circuit_level.AnalogFrontend(
    finite_gain_op_amp_analog_system, verilog_digital_control
)

first_order_pole_op_amp_analog_frontend = cbadc.circuit_level.AnalogFrontend(
    first_order_pole_op_amp_analog_system, verilog_digital_control
)

###############################################################################
# Verilog Modules
# ------------------
#
# These analog frontends can now be converted directly into verilog-ams
# modules as

verilog_analog_frontend.to_file(filename="verilog_analog_frontend.vams")

ideal_op_amp_analog_frontend.to_file(filename="ideal_op_amp_analog_frontend.vams")

finite_gain_op_amp_analog_frontend.to_file(
    filename="finite_gain_op_amp_analog_frontend.vams"
)

first_order_pole_op_amp_analog_frontend.to_file(
    filename="first_order_pole_op_amp_analog_frontend.vams"
)

###############################################################################
# where each generated file can be downloaded below
# :download:`verilog_analog_frontend.vams <./verilog_analog_frontend.vams>`
# :download:`ideal_op_amp_analog_frontend.vams <./ideal_op_amp_analog_frontend.vams>`
# :download:`finite_gain_op_amp_analog_frontend.vams <./finite_gain_op_amp_analog_frontend.vams>`
# :download:`first_order_pole_op_amp_analog_frontend.vams <./first_order_pole_op_amp_analog_frontend.vams>`


###############################################################################
# Transfer Functions
# ------------------
#
# As we did for the analog systems we plot the digital estimators corresponding
# noise and signal transfer functions
#
# To get the resulting digital estimator from an analog frontend we use the
# function :py:func:`cbadc.circuit_level.analog_frontend.AnalogFrontend.get_estimator`
# Which requires us to pass an digital estimator class together with an eta2, K1, and
# K2 value.

eta2 = (
    np.linalg.norm(
        target_analog_system.transfer_function_matrix(np.array([2 * np.pi * BW]))
    )
    ** 2
)
K1 = 1 << 8
K2 = K1

digital_estimator_parameters = [cbadc.digital_estimator.FIRFilter, eta2, K1, K2]

compare_systems = {
    'verilog_ideal': verilog_analog_frontend.get_estimator(
        *digital_estimator_parameters
    ),
    'finite_gain op-amp': finite_gain_op_amp_analog_frontend.get_estimator(
        *digital_estimator_parameters
    ),
    'single_pole op-amp': first_order_pole_op_amp_analog_frontend.get_estimator(
        *digital_estimator_parameters
    ),
}
BW_log = np.log10(BW)
frequencies = np.logspace(BW_log - 2, BW_log + 1, 500)
omegas = 2 * np.pi * frequencies

for key, digital_estimator in compare_systems.items():

    # Compute STF
    stf = digital_estimator.signal_transfer_function(omegas)
    stf_dB = 20 * np.log10(np.abs(stf.flatten()))

    plt.semilogx(frequencies, stf_dB, label="$|STF(\omega)|$ " + key)

for key, digital_estimator in compare_systems.items():
    # Compute NTF
    ntf = digital_estimator.noise_transfer_function(omegas)
    ntf_dB = 20 * np.log10(np.abs(ntf))

    plt.semilogx(
        frequencies,
        20 * np.log10(np.linalg.norm(ntf[0, :, :], axis=0)),
        "--",
        label="$ || NTF(\omega) ||_2 $, " + key,
    )

# Add labels and legends to figure

plt.legend()
plt.grid(which="both")
plt.title("Signal and noise transfer functions")
plt.xlabel("frequencies [Hz]")
plt.ylabel("dB")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()

###############################################################################
# Impulse Responses
# ------------------
#
# We also compare the resulting filters impulse responses

for key, digital_estimator in compare_systems.items():
    plt.plot(
        np.arange(-K1, K2),
        np.array(digital_estimator.h[0, :, 0])[:],
        label=key,
    )
plt.legend()
plt.xlabel("filter tap k")
plt.ylabel("$ h_1 [k] $")
plt.xlim((-K1, K2))
plt.grid(which="both")

plt.figure()
for key, digital_estimator in compare_systems.items():
    plt.semilogy(
        np.arange(-K1, K2),
        np.abs(np.array(digital_estimator.h[0, :, 0]))[:],
        label=key,
    )
plt.legend()
plt.xlabel("filter tap k")
plt.ylabel("$| h_1 [k] \|$")
plt.xlim((-K1, K2))
plt.grid(which="both")

# sphinx_gallery_thumbnail_number = 2
