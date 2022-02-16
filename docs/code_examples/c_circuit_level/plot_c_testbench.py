"""
======================
Testbench
======================

In this tutorial we expand on the :doc:`./plot_b_analog_frontend`
tutorial by adding a clock and input signal to the analog frontend.
Together, this makes up a full simulation environment where the
analog frontend can be evaluated through simulation. We refer to
such a setup as a testbench.
"""

import cbadc
import numpy as np
import matplotlib.pyplot as plt
import copy

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
target_analog_system, target_digital_control = cbadc.specification.get_leap_frog(
    ENOB=ENOB, N=N, BW=BW
)

verilog_analog_system = cbadc.circuit_level.AnalogSystemStateSpaceEquations(
    target_analog_system
)

C = 1e-12
ideal_op_amp_analog_system = cbadc.circuit_level.AnalogSystemIdealOpAmp(
    target_analog_system, C
)

A_DC = 2e2
omega_p = 2 * np.pi * BW / 2

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
# we use :py:class:cbadc.circuit_level.digital_control.DigitalControl class
# as

verilog_digital_control = cbadc.circuit_level.DigitalControl(
    copy.deepcopy(target_digital_control)
)

###############################################################################
# Analog Frontend
# ------------------
#
# Subsequently, the analog frontend can be pertained by the cbadc.circuit_level.AnalogFrontend
# class as

verilog_analog_frontend = cbadc.circuit_level.AnalogFrontend(
    verilog_analog_system, copy.deepcopy(verilog_digital_control)
)

ideal_op_amp_analog_frontend = cbadc.circuit_level.AnalogFrontend(
    ideal_op_amp_analog_system, copy.deepcopy(verilog_digital_control)
)

finite_gain_op_amp_analog_frontend = cbadc.circuit_level.AnalogFrontend(
    finite_gain_op_amp_analog_system, copy.deepcopy(verilog_digital_control)
)

first_order_pole_op_amp_analog_frontend = cbadc.circuit_level.AnalogFrontend(
    first_order_pole_op_amp_analog_system, copy.deepcopy(verilog_digital_control)
)

###############################################################################
# Input Signal and Simulation Clock
# ---------------------------------
#
# As in the high level simulation case, we define an input signal
# and simulation clock

amplitude = 1.0
frequency = 1.0 / target_digital_control.clock.T
while frequency > BW:
    frequency /= 2
input_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency)
simulation_clock = cbadc.analog_signal.Clock(target_digital_control.clock.T)

###############################################################################
# Testbench
# ---------
#
# Instantiating the testbench reminds the simulation setup.
#

# determine simulation endtime after 100000 digital control cycles.
t_stop = target_digital_control.clock.T * 100000

verilog_testbench = cbadc.circuit_level.TestBench(
    verilog_analog_frontend, input_signal, simulation_clock, t_stop
)

ideal_op_amp_testbench = cbadc.circuit_level.TestBench(
    ideal_op_amp_analog_frontend, input_signal, simulation_clock, t_stop
)

finite_gain_op_amp_testbench = cbadc.circuit_level.TestBench(
    finite_gain_op_amp_analog_frontend, input_signal, simulation_clock, t_stop
)

first_order_pole_op_amp_testbench = cbadc.circuit_level.TestBench(
    first_order_pole_op_amp_analog_frontend, input_signal, simulation_clock, t_stop
)


###############################################################################
# Spice Scripts and Verilog Modules
# ---------------------------------
#
# Next, the testbenches can be rendered into a spice testbench script
# together with the analog frontend verilog module.
#

verilog_testbench.to_file(filename="verilog_testbench")

ideal_op_amp_testbench.to_file(filename="ideal_op_amp_analog_testbench")

finite_gain_op_amp_testbench.to_file(filename="finite_gain_op_amp_testbench")

first_order_pole_op_amp_testbench.to_file(filename="first_order_pole_op_amp_testbench")

###############################################################################
# where each generated file can be downloaded below
#
# :download:`verilog_testbench.txt <verilog_testbench.txt>`,
# :download:`verilog_analog_frontend.vams <verilog_analog_frontend.vams>`
#
# :download:`ideal_op_amp_analog_testbench.txt <ideal_op_amp_analog_testbench.txt>`,
# :download:`ideal_op_amp_analog_frontend.vams <ideal_op_amp_analog_frontend.vams>`
#
# :download:`finite_gain_op_amp_testbench.txt <finite_gain_op_amp_testbench.txt>`,
# :download:`finite_gain_op_amp_analog_frontend.vams <finite_gain_op_amp_analog_frontend.vams>`
#
# :download:`first_order_pole_op_amp_testbench.txt <first_order_pole_op_amp_testbench.txt>`,
# :download:`first_order_pole_op_amp_analog_frontend.vams <first_order_pole_op_amp_analog_frontend.vams>`

###############################################################################
# Simulation and Verification
# ---------------------------
#


eta2 = (
    np.linalg.norm(
        target_analog_system.transfer_function_matrix(np.array([2 * np.pi * BW]))
    )
    ** 2
)
K1 = 1 << 12
K2 = K1

digital_estimator_parameters = [cbadc.digital_estimator.FIRFilter, eta2, K1, K2]
simulation_type = cbadc.simulator.SimulatorType.full_numerical

compare_systems = {
    'verilog_ideal': {
        "digital_estimator": verilog_testbench.analog_frontend.get_estimator(
            *digital_estimator_parameters
        ),
        "simulator": verilog_testbench.get_simulator(simulation_type),
    },
    'finite_gain op-amp': {
        "digital_estimator": finite_gain_op_amp_testbench.analog_frontend.get_estimator(
            *digital_estimator_parameters
        ),
        "simulator": finite_gain_op_amp_testbench.get_simulator(simulation_type),
    },
    'single_pole op-amp': {
        "digital_estimator": first_order_pole_op_amp_testbench.analog_frontend.get_estimator(
            *digital_estimator_parameters
        ),
        "simulator": first_order_pole_op_amp_testbench.get_simulator(simulation_type),
    },
}
BW_log = np.log10(BW)
frequencies = np.logspace(BW_log - 2, BW_log + 1, 500)
omegas = 2 * np.pi * frequencies

## Plot digital estimators transfer functions

for key, system in compare_systems.items():
    digital_estimator = system['digital_estimator']
    # Compute STF
    stf = digital_estimator.signal_transfer_function(omegas)
    stf_dB = 20 * np.log10(np.abs(stf.flatten()))

    plt.semilogx(frequencies, stf_dB, label="$|STF(\omega)|$ " + key)

for key, system in compare_systems.items():
    digital_estimator = system['digital_estimator']
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

size = 1 << 15
u_hat = np.zeros(size)

plt.figure()
for key, system in compare_systems.items():
    # Compute NTF
    digital_estimator = system['digital_estimator']
    simulator = system['simulator']
    digital_estimator(simulator)
    for index in range(size):
        u_hat[index] = next(digital_estimator)
    u_hat_cut = u_hat[K1 + K2 :]
    f, psd = cbadc.utilities.compute_power_spectral_density(
        u_hat_cut[:], fs=1 / target_digital_control.clock.T, nperseg=u_hat_cut.size
    )
    signal_index = cbadc.utilities.find_sinusoidal(psd, 15)
    noise_index = np.ones(psd.size, dtype=bool)
    noise_index[signal_index] = False
    noise_index[f < (BW * 1e-2)] = False
    noise_index[f > BW] = False
    fom = cbadc.utilities.snr_spectrum_computation_extended(
        psd, signal_index, noise_index, fs=1 / target_digital_control.clock.T
    )
    est_SNR = cbadc.fom.snr_to_dB(fom['snr'])
    est_ENOB = cbadc.fom.snr_to_enob(est_SNR)
    plt.semilogx(
        f,
        10 * np.log10(np.abs(psd)),
        label=key + f", est_ENOB={est_ENOB:.1f} bits, est_SNR={est_SNR:.1f} dB",
    )


plt.title(f"Power spectral density of input estimate")
plt.xlabel('Hz')
plt.ylabel('$V^2$ / Hz dB')
plt.legend()
plt.grid(which="both")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()

# sphinx_gallery_thumbnail_number = 2
