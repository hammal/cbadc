"""
Simulating a Continuous-Time Delta-Sigma Modulator
==================================================
"""
import cbadc
import numpy as np
import matplotlib.pyplot as plt
import json

###############################################################################
# Instantiating the Analog System and Digital Control
# ---------------------------------------------------
#
# We start by loading a delta sigma modulator constructed
# using [www.sigma-delta.de](www.sigma-delta.de) framework.
#

T = 0.5e-8
with open('CTSD_N3_OSR40_Q8_CRFB.json') as f:
    analog_frontend = cbadc.synthesis.ctsd_dict2af(json.load(f), T, 1.0)

print(analog_frontend.analog_system)
print(analog_frontend.digital_control)
