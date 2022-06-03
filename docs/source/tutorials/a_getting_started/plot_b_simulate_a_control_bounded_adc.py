"""
Simulating a CBADC
================================

This example shows how to simulate the interactions between an analog system
and a digital control while the former is excited by an analog signal.
"""
import matplotlib.pyplot as plt
import cbadc
import numpy as np

###############################################################################
# The Analog System
# -----------------
#
# .. image:: /images/chainOfIntegratorsGeneral.svg
#    :width: 500
#    :align: center
#    :alt: The chain of integrators ADC.
#
# First, we have to decide on an analog system. For this tutorial, we will
# commit to a chain-of-integrators ADC,
# see :py:class:`cbadc.analog_system.ChainOfIntegrators`, as our analog
# system.

# We fix the number of analog states.
N = 6
# Set the amplification factor.
beta = 6250.0
rho = -1e-2
kappa = -1.0
# In this example, each nodes amplification and local feedback will be set
# identically.
betaVec = beta * np.ones(N)
rhoVec = betaVec * rho
kappaVec = kappa * beta * np.eye(N)

# Instantiate a chain-of-integrators analog system.
analog_system = cbadc.analog_system.ChainOfIntegrators(betaVec, rhoVec, kappaVec)
# print the analog system such that we can very it being correctly initalized.
print(analog_system)

###############################################################################
# The Digital Control
# -------------------
#
# In addition to the analog system, our simulation will require us to specify a
# digital control. For this tutorial, we will use
# :py:class:`cbadc.digital_control.DigitalControl`.

# Set the time period which determines how often the digital control updates.
T = 1.0 / (2 * beta)
# Instantiate a corresponding clock.
clock = cbadc.analog_signal.Clock(T)
# Set the number of digital controls to be same as analog states.
M = N
# Initialize the digital control.
digital_control = cbadc.digital_control.DigitalControl(clock, M)
# print the digital control to verify proper initialization.
print(digital_control)


###############################################################################
# The Analog Signal
# -----------------
#
# The final and third component of the simulation is an analog signal.
# For this tutorial, we will choose a
# :py:class:`cbadc.analog_signal.Sinusoidal`. Again, this is one of several
# possible choices.

# Set the peak amplitude.
amplitude = 0.5
# Choose the sinusoidal frequency via an oversampling ratio (OSR).
OSR = 1 << 9
frequency = 1.0 / (T * OSR)

# We also specify a phase an offset these are hovewer optional.
phase = np.pi / 3
offset = 0.0

# Instantiate the analog signal
analog_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)
# print to ensure correct parametrization.
print(analog_signal)

###############################################################################
# Simulating
# -------------
#
# Next, we set up the simulator. Here we use the
# :py:class:`cbadc.simulator.StateSpaceSimulator` for simulating the
# involved differential equations as outlined in
# :py:class:`cbadc.analog_system.AnalogSystem`.
#

# Simulate for 2^18 control cycles.
size = 1 << 18
end_time = T * size

# Instantiate the simulator.
simulator = cbadc.simulator.get_simulator(
    analog_system,
    digital_control,
    [analog_signal],
    t_stop=end_time,
)
# Depending on your analog system the step above might take some time to
# compute as it involves precomputing solutions to initial value problems.

# Let's print the first 20 control decisions.
index = 0
for s in cbadc.utilities.show_status(simulator):
    if index > 19:
        break
    print(f"step:{index} -> s:{np.array(s)}")
    index += 1

# To verify the simulation parametrization we can
print(simulator)

###############################################################################
# Tracking the Analog State Vector
# --------------------------------
#
# Clearly the output type of the generator simulator above is the sequence of
# control signals s[k]. Sometimes we are interested in also monitoring the
# internal states of analog system during simulation.
#
# To this end we use the
# :func:`cbadc.simulator.StateSpaceSimulator.state_vector` and an
# :func:`cbadc.simulator.extended_simulation_result`.
#
# Note that the :func:`cbadc.simulator.extended_simulation_result` is
# defined like this
#
# .. code-block:: python
#
#   def extended_simulation_result(simulator):
#       for control_signal in simulator:
#           analog_state = simulator.state_vector()
#           yield {
#               'control_signal': np.array(control_signal),
#               'analog_state': np.array(analog_state)
#           }
#
# So, in essence, we are creating a new generator from the old with an extended
# output.
#
# .. note:: The convenience function extended_simulation_result is one of many
#           such convenience functions found in the
#           :py:mod:`cbadc.simulator` module.
#
# We can achieve this by appending yet another generator to the control signal
# stream as:

# Repeating the steps above we now get for the following
# ten control cycles.

ext_simulator = cbadc.simulator.extended_simulation_result(simulator)
for res in cbadc.utilities.show_status(ext_simulator):
    if index > 29:
        break
    print(f"step:{index} -> s:{res['control_signal']}, x:{res['analog_state']}")
    index += 1

###############################################################################
# .. _default_simulation:
#
# --------------------------------
# Saving to File
# --------------------------------
#
# In general, simulating the analog system and digital control interaction
# is a computationally much more intense procedure compared to the digital
# estimation step. This is one reason, and there are more, why
# you would want to store the intermediate control signal sequence to a file.
#
# For this purpose use the
# :func:`cbadc.utilities.control_signal_2_byte_stream` and
# :func:`cbadc.utilities.write_byte_stream_to_file` functions.

# Instantiate a new simulator and control.
simulator = cbadc.simulator.get_simulator(
    analog_system,
    digital_control,
    [analog_signal],
    t_stop=end_time,
)

# Construct byte stream.
byte_stream = cbadc.utilities.control_signal_2_byte_stream(simulator, M)


def print_next_10_bytes(stream):
    global index
    for byte in cbadc.utilities.show_status(stream, size):
        if index < 40:
            print(f"{index} -> {byte}")
            index += 1
        yield byte


cbadc.utilities.write_byte_stream_to_file(
    "sinusoidal_simulation.dat", print_next_10_bytes(byte_stream)
)

###############################################################################
# Evaluating the Analog State Vector in Between Control Signal Samples
# --------------------------------------------------------------------
#
# If we wish to simulate the analog state vector trajectory between
# control updates, this can be achieved using the Ts parameter of the
# :py:class:`cbadc.simulator.StateSpaceSimulator`. Technically you can scale
# :math:`T_s = T / \alpha` for any positive number :math:`\alpha`. For such a
# scaling, the simulator will generate :math:`\alpha` more control signals per
# unit of time. However, digital control is still restricted to only update
# the control signals at multiples of :math:`T`.
#

# Set sampling time two orders of magnitude smaller than the control period
Ts = T / 100.0
# Instantiate a corresponding clock
observation_clock = cbadc.analog_signal.Clock(Ts)

# Simulate for 65536 control cycles.
size = 1 << 16

# Initialize a new digital control.
new_digital_control = cbadc.digital_control.DigitalControl(clock, M)

# Instantiate a new simulator with a sampling time.
simulator = cbadc.simulator.AnalyticalSimulator(
    analog_system, new_digital_control, [analog_signal], observation_clock
)

# Create data containers to hold the resulting data.
time_vector = np.arange(size) * Ts / T
states = np.zeros((N, size))
control_signals = np.zeros((M, size), dtype=np.int8)

# Iterate through and store states and control_signals.
simulator = cbadc.simulator.extended_simulation_result(simulator)
for index in cbadc.utilities.show_status(range(size)):
    res = next(simulator)
    states[:, index] = res["analog_state"]
    control_signals[:, index] = res["control_signal"]

# Plot all analog state evolutions.
plt.figure()
plt.title("Analog state vectors")
for index in range(N):
    plt.plot(time_vector, states[index, :], label=f"$x_{index + 1}(t)$")
plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
plt.xlabel("$t/T$")
plt.xlim((0, 10))
plt.legend()

# reset figure size and plot individual results.
plt.rcParams["figure.figsize"] = [6.40, 6.40 * 2]
fig, ax = plt.subplots(N, 2)
for index in range(N):
    color = next(ax[0, 0]._get_lines.prop_cycler)["color"]
    ax[index, 0].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[index, 1].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
    ax[index, 0].plot(time_vector, states[index, :], color=color)
    ax[index, 1].plot(time_vector, control_signals[index, :], "--", color=color)
    ax[index, 0].set_ylabel(f"$x_{index + 1}(t)$")
    ax[index, 1].set_ylabel(f"$s_{index + 1}(t)$")
    ax[index, 0].set_xlim((0, 15))
    ax[index, 1].set_xlim((0, 15))
    ax[index, 0].set_ylim((-1, 1))
fig.suptitle("Analog state and control contribution evolution")
ax[-1, 0].set_xlabel("$t / T$")
ax[-1, 1].set_xlabel("$t / T$")
fig.tight_layout()

###############################################################################
# Analog State Statistics
# ------------------------------------------------------------------
#
# As in the previous section, visualizing the analog state trajectory is a
# good way of identifying problems and possible errors. Another way of making
# sure that the analog states remain bounded is to estimate their
# corresponding densities (assuming i.i.d samples).

# Compute L_2 norm of analog state vector.
L_2_norm = np.linalg.norm(states, ord=2, axis=0)
# Similarly, compute L_infty (largest absolute value) of the analog state
# vector.
L_infty_norm = np.linalg.norm(states, ord=np.inf, axis=0)

# Estimate and plot densities using matplotlib tools.
bins = 150
plt.rcParams["figure.figsize"] = [6.40, 4.80]
fig, ax = plt.subplots(2, sharex=True)
ax[0].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
ax[1].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
ax[0].hist(L_2_norm, bins=bins, density=True)
ax[1].hist(L_infty_norm, bins=bins, density=True, color="orange")
plt.suptitle("Estimated probability densities")
ax[0].set_xlabel("$\|\mathbf{x}(t)\|_2$")
ax[1].set_xlabel("$\|\mathbf{x}(t)\|_\infty$")
ax[0].set_ylabel("$p ( \| \mathbf{x}(t) \|_2 ) $")
ax[1].set_ylabel("$p ( \| \mathbf{x}(t) \|_\infty )$")
fig.tight_layout()
