"""
====================
Creating Artsy Plots
====================

This notebook shows examples of artsy plots made from data generated with the
control-bounded conversion toolbox. In this example, we use input signals (or
their estimate) and the control bitstream generated by the converter.
The plots are rendered using matplotlib.
"""

from cbadc.datasets import hadamard
from cbadc.simulator import get_simulator
from cbadc.analog_signal import Sinusoidal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from cycler import cycler


###############################################################################
# -------------
# Generate Data
# -------------
#
# First, we need to generate some data to plot it later. For these examples we
# use the Hadamard converter PCB as a model. We stimulate the system with a
# sinusoidal input and capture the generated bitstream.
#

pcb = hadamard.HadamardPCB("B")
M = pcb.digital_control.M
T = pcb.T

# Set the peak amplitude.
amplitude = 2.5  # 2.5 V is the theoretical limit for the hardware prototype
# Choose the sinusoidal frequency via an oversampling ratio (OSR).
OSR = 1 << 8
frequency = 1.0 / (T * OSR)
# Instantiate the analog signal
analog_signal = Sinusoidal(amplitude, frequency)
# print to ensure correct parametrization.
print(analog_signal)

# simulate for 1024 cycles
n_cycles = 1 << 10
end_time = T * n_cycles
# Instantiate the simulator.
simulator = get_simulator(
    pcb.analog_system, pcb.digital_control, [analog_signal], t_stop=end_time
)

##############################################################################
# Finally we extract the bitstream and store it in a numpy array.

ctrl_stream = np.zeros((n_cycles, M))
for index, s in enumerate(simulator):
    ctrl_stream[index, :] = np.array(s)

##############################################################################
# -------------------
# Barcode Style
# -------------------
# We want to display the individual control bit signals in a barcode manner.
# We achieve this by filling the area between the rectangular control signals
# and the reference level. The barcodes are then stacked vertically.

plt.rcParams["figure.figsize"] = [16, 10]  # modify default size of plot

# start and length of the displayed snippet
xstart = 128
xlim = 256 + 1  # add 1 to get a symmetrical image

x = np.arange(xstart, xstart + xlim)
ctrl_sig = np.transpose(ctrl_stream[xstart : xstart + xlim, :])
# from top to bottom: 0th to 7th bit
offset = np.reshape(np.repeat(np.arange(M)[::-1], xlim), (M, xlim))

fig1, ax1 = plt.subplots()

for i in range(M):
    ax1.fill_between(x, offset[i], ctrl_sig[i] + offset[i], step="pre")

fig1.tight_layout()
ax1.set_axis_off()
ax1.set_facecolor("w")

##############################################################################
# Note how the character of the prototype (Hadamard converter with Hadamard
# and diagonal controls) reflects in the different patterns.

##############################################################################
# -------------------
# Choose Colors
# -------------------
# We can set the individual colors of the bit signals by using a color cycler.
# Also, we add the input signal on top.

# set individual colors of control bits
custom_cycler = cycler(color=["k", "c", "m", "y"])

fig2, ax2 = plt.subplots()
ax2.set_prop_cycle(custom_cycler)
# remove the previous line to get the standard matplotlib color sequence

for i in range(M):
    ax2.fill_between(x, offset[i], ctrl_sig[i] + offset[i], step="pre", alpha=0.5)

# plot input signal        (adjust amplitude to be slightly smaller that barcode stack)
input_signal = analog_signal.evaluate(x * T) / analog_signal.amplitude * M / 2.1 + M / 2
ax2.plot(x, input_signal, "k", linewidth=3)

fig2.tight_layout()
ax2.set_axis_off()
ax2.set_facecolor("w")

##############################################################################
# ---------------------
# Color Gradients
# ---------------------
# By specifying colors in the HSL space we can control them more intuitively.
# This also allows to create neat gradients.
#
hue_upper = 0.5  # ~turquoise
hue_lower = 0.9  # ~rose
min_sat = 0.3
max_sat = 1
min_lum = 0.8
max_lum = 1

hue = [hue_upper] * 4 + [hue_lower] * 4
sat = np.concatenate(
    (np.linspace(max_sat, min_sat, 4), np.linspace(min_sat, max_sat, 4))
)
lum = np.concatenate(
    (np.linspace(min_lum, max_lum, 4), np.linspace(max_lum, min_lum, 4))
)
custom_cycler = cycler(color=[hsv_to_rgb((hue[i], sat[i], lum[i])) for i in range(8)])

fig3, ax3 = plt.subplots(figsize=(40 / 2.54, 24 / 2.54))
ax3.set_prop_cycle(custom_cycler)
# remove the previous line to get the standard matplotlib color sequence

for i in range(M):
    ax3.fill_between(x, offset[i], ctrl_sig[i] + offset[i], step="pre")

# plot input signal        (adjust amplitude to be slightly smaller that barcode stack)
input_signal = analog_signal.evaluate(x * T) / analog_signal.amplitude * M / 2.1 + M / 2
ax3.plot(x, input_signal, "k", linewidth=3, solid_capstyle="round")

fig3.tight_layout()
ax3.set_axis_off()
ax3.set_facecolor("w")

##############################################################################
# -------------
# More Examples
# -------------

hue_upper = 0.5  # ~turquoise
hue_lower = 0.9  # ~rose
min_sat = 0.3
max_sat = 1
min_lum = 0.8
max_lum = 1

hue = [hue_upper] * 4 + [hue_lower] * 4
sat = np.concatenate(
    (np.linspace(min_sat, max_sat, 4), np.linspace(max_sat, min_sat, 4))
)
lum = np.concatenate(
    (np.linspace(max_lum, min_lum, 4), np.linspace(min_lum, max_lum, 4))
)
custom_cycler = cycler(color=[hsv_to_rgb((hue[i], sat[i], lum[i])) for i in range(8)])

fig4, ax4 = plt.subplots(figsize=(40 / 2.54, 24 / 2.54))
ax4.set_prop_cycle(custom_cycler)

for i in range(M):
    ax4.fill_between(x, offset[i], ctrl_sig[i] + offset[i], step="pre")

ax4.text(
    xstart + xlim / 4,
    M / 2,
    "cbadc",
    color="w",
    fontsize=140,
    fontfamily="fantasy",
    alpha=0.85,
    ha="center",
    va="center",
)

input_signal = analog_signal.evaluate(x * T) / analog_signal.amplitude * M / 2.1 + M / 2
ax4.plot(x, input_signal, "k", linewidth=3, solid_capstyle="round")

fig4.tight_layout()
ax4.set_axis_off()
ax4.set_facecolor("w")

##############################################################################

hue_upper = 0.44  # ~green
hue_lower = 0.57  # ~blue
min_sat = 0.27
max_sat = 0.9
min_lum = 0.9
max_lum = 1

hue = np.linspace(hue_upper, hue_lower, 8)
sat = np.concatenate(
    (np.linspace(max_sat, min_sat, 4), np.linspace(min_sat, max_sat, 4))
)
lum = np.concatenate(
    (np.linspace(min_lum, max_lum, 4), np.linspace(max_lum, min_lum, 4))
)
custom_cycler = cycler(color=[hsv_to_rgb((hue[i], sat[i], lum[i])) for i in range(8)])

fig5, ax5 = plt.subplots(figsize=(40 / 2.54, 24 / 2.54))
ax5.set_prop_cycle(custom_cycler)

for i in range(M):
    ax5.fill_between(x, offset[i], ctrl_sig[i] + offset[i], step="pre")

input_signal = analog_signal.evaluate(x * T) / analog_signal.amplitude * M / 2.1 + M / 2
ax5.plot(x, input_signal, "k", linewidth=2, solid_capstyle="round")

fig5.tight_layout()
ax5.set_axis_off()
ax5.set_facecolor("w")

##############################################################################
# sphinx_gallery_thumbnail_number = 6

min_sat = 0.6
max_sat = 0.6
min_lum = 1
max_lum = 1

# rainbow with specially picked colors
hue = [0, 0.09, 0.18, 0.4, 0.5, 0.6, 0.7, 0.8]
sat = np.concatenate(
    (np.linspace(min_sat, max_sat, 4), np.linspace(max_sat, min_sat, 4))
)
lum = np.concatenate(
    (np.linspace(max_lum, min_lum, 4), np.linspace(min_lum, max_lum, 4))
)
custom_cycler = cycler(color=[hsv_to_rgb((hue[i], sat[i], lum[i])) for i in range(8)])

fig6, ax6 = plt.subplots(figsize=(40 / 2.54, 24 / 2.54))
ax6.set_prop_cycle(custom_cycler)

for i in range(M):
    ax6.fill_between(x, offset[i], ctrl_sig[i] + offset[i], step="pre")

input_signal = analog_signal.evaluate(x * T) / analog_signal.amplitude * M / 2.1 + M / 2
ax6.plot(x, input_signal, "k", linewidth=4, solid_capstyle="round")

fig6.tight_layout()
ax6.set_axis_off()
ax6.set_facecolor("w")

##############################################################################
# ------------
# Export Image
# ------------
# After playing around with the plots, uncomment  one of the lines to export
# your favourite image
#

# fig1.savefig('artsy_1.png', dpi=300)
# fig2.savefig('artsy_2.png', dpi=300)
# fig3.savefig('artsy_3.png', dpi=300)
# fig4.savefig('artsy_4.png', dpi=300)
# fig5.savefig('artsy_5.png', dpi=300)
# fig6.savefig('artsy_6.png', dpi=300)
