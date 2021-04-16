"""
Transfer Functions
==================

This example demonstrates how to visualize the related transfer functions of the 
analog system and digital estimator.
"""

###############################################################################
# Chain-of-Integrators ADC Example
# --------------------------------
# 
# In this example we will use the chain-of-integrators ADC analog system for
# demonstrational purposes. However, except for the analog system creation,
# the steps for a generic analog system and digital estimator.
#
# for an in depth details regarding the chain-of-integrators transfer function see
# `chain-of-integrators <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf?sequence=1&isAllowed=y#page=97/>`_
# 
# First we will import dependent modules and initialize a chain-of-integrators 
# setup. With the following analog system parameters
#
# - :math:`\beta = \beta_1 = \dots = \beta_N = 6250`
# - :math:`\rho_1 = \dots = \rho_N = - \beta / 10`
# - :math:`\kappa_1 = \dots = \kappa_N = - \beta`
# - :math:`N = 6`
#
# note that :math:`\mathbf{C}^\mathsf{T}` is automatically assumed an identity
# matrix of size :math:`N\times N`.
#
# Using the :py:class:`cbadc.analog_system.ChainOfIntegrators` class which
# derives from the main analog system class 
# :py:class:`cbadc.analog_system.AnalogSystem`.

import matplotlib.pyplot as plt
from cbadc.analog_system import ChainOfIntegrators
import numpy as np
# We fix the number of analog states.
N = 6
# Set the amplification factor.
beta = 6250.
# In this example, each nodes amplification and local feedback will be set
# identically.
betaVec = beta * np.ones(N) 
rhoVec = -betaVec / 50.
kappaVec = - beta * np.eye(N)

# Instantiate a chain-of-integrators analog system.
analog_system = ChainOfIntegrators(betaVec, rhoVec, kappaVec)
# print the system matrices.
print(analog_system)

###############################################################################
# Plotting the Analog System's Transfer Function
# ----------------------------------------------
#
# Next we plot the transfer function of the analog system
#
# :math:`\mathbf{G}(\omega) = \begin{pmatrix}G_1(\omega), \dots, G_N(\omega)\end{pmatrix}^\mathsf{T} = \mathbf{C}^\mathsf{T} \left(i \omega \mathbf{I}_N - \mathbf{A}\right)^{-1}\mathbf{B}`
#
# using the class method :func:`cbadc.analog_system.AnalogSystem.transfer_function`.
 
# Logspace frequencies
frequencies = np.logspace(-3, 0, 500)
omega = 4 * np.pi * beta * frequencies

# Compute transfer functions for each frequency in frequencies
transfer_function = analog_system.transfer_function(omega)
transfer_function_dB = 20 * np.log10(np.abs(transfer_function))

# For each output 1,...,N compute the corresponding tranfer function seen
# from the input.
for n in range(N):
    plt.semilogx(frequencies, transfer_function_dB[n, 0, :], label=f"$G_{n+1}(\omega)$")

# Add the norm ||G(omega)||_2
plt.semilogx(frequencies, 20 * np.log10(np.linalg.norm(transfer_function[:, 0, :], axis=0)), '--', label="$ ||\mathbf{G}(\omega)||_2 $")

# Add labels and legends to figure
plt.legend()
plt.grid(which='both')
plt.title("Transfer functions, $G_1(\omega), \dots, G_N(\omega)$")
plt.xlabel("$\omega / (4 \pi \\beta ) $")
plt.ylabel("dB")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()

###############################################################################
# Plotting the Estimator's Signal and Noise Transfer Function
# -----------------------------------------------------------
#
# To determine the estimate's signal and noise transfer function we must
# instantiate a digital estimator
# :py:class:`cbadc.digital_estimator.DigitalEstimator`. The bandwidth of the
# digital estimation filter is mainly goverened by the parameter :math:`\eta^2`
# as the noise transfer function (NTF) follows as
#
# :math:`\text{NTF}( \omega) = \mathbf{G}( \omega)^\mathsf{H} \left(
# \mathbf{G}( \omega)\mathbf{G}( \omega)^\mathsf{H} + \eta^2 \mathbf{I}_N
# \right)^{-1}`
#
# and similarly the signal transfer function (STF) follows as
#
# :math:`\text{STF}( \omega) = \text{NTF}( \omega) \mathbf{G}( \omega)`.
#
# We compute these two by invoking the class methods
# :func:`cbadc.digital_estimator.DigitalEstimator.noise_transfer_function` and
# :func:`cbadc.digital_estimator.DigitalEstimator.signal_transfer_function`
# respectively.
#
# the digital estimator requires us to also instantiate a digital control
# :py:class:`cbadc.digital_control.DigitalControl`.
#
# For the chain-of-integrators example the noise transfer function
# results in a row vector 
# :math:`\text{NTF}(\omega) = \begin{pmatrix} \text{NTF}_1(\omega), \dots, \text{NTF}_N(\omega)\end{pmatrix} \in \mathbb{C}^{1 \times \tilde{N}}`
# where :math:`\text{NTF}_\ell(\omega)` refers to the noise transfer function
# from the :math:`\ell`-th observation to the final estimate.
from cbadc.digital_estimator import DigitalEstimator
from cbadc.digital_control import DigitalControl

# Define dummy control and control sequence (not used when computing transfer functions)
# However necessary to instantiate the digital estimator
T = 1/(2 * beta)
digital_control = DigitalControl(T, N)
def control_sequence():
    yield np.zeros(N)

# Compute eta2 for a given bandwidth.
omega_3dB = (4 * np.pi * beta) / 100.
eta2 = np.linalg.norm(analog_system.transfer_function(np.array([omega_3dB])).flatten()) ** 2

# Instantiate estimator.
digital_estimator = DigitalEstimator(control_sequence, analog_system, digital_control, eta2, K1 = 1)

# Compute NTF
ntf = digital_estimator.noise_transfer_function(omega)
ntf_dB = 20 * np.log10(np.abs(ntf))

# Compute STF
stf = digital_estimator.signal_transfer_function(omega)
stf_dB = 20 * np.log10(np.abs(stf.flatten()))


# Plot
plt.figure()
plt.semilogx(frequencies, stf_dB, label='$STF(\omega)$')
for n in range(N):
    plt.semilogx(frequencies, ntf_dB[0, n, :], label=f"$|NTF_{n+1}(\omega)|$")
plt.semilogx(frequencies, 20 * np.log10(np.linalg.norm(ntf[0,:,:], axis=0)), '--', label="$ || NTF(\omega) ||_2 $")

# Add labels and legends to figure
plt.legend()
plt.grid(which='both')
plt.title("Signal and noise transfer functions")
plt.xlabel("$\omega / (4 \pi \\beta ) $")
plt.ylabel("dB")
plt.xlim((frequencies[0], frequencies[-1]))
plt.gcf().tight_layout()

###############################################################################
# Setting the Bandwidth of the Estimation Filter
# ----------------------------------------------
#
# Next we will investigate the effect of eta2 on the STF and NTF.

# create a vector of etas to be evaluated,
eta2_vec = np.logspace(0, 10, 11)[::2]

plt.figure()
for eta2 in eta2_vec:
    # Instantiate an estimator for each eta.
    digital_estimator = DigitalEstimator(control_sequence, analog_system, digital_control, eta2, K1 = 1)
    # Compute stf and ntf
    ntf = digital_estimator.noise_transfer_function(omega)
    ntf_dB = 20 * np.log10(np.abs(ntf))
    stf = digital_estimator.signal_transfer_function(omega)
    stf_dB = 20 * np.log10(np.abs(stf.flatten()))

    # Plot
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    plt.semilogx(frequencies, 20 * np.log10(np.linalg.norm(ntf[0,:,:], axis=0)), '--', color=color)      
    plt.semilogx(frequencies, stf_dB, label=f"$\eta^2={20 * np.log10(eta2):0.0f} dB$", color=color)

# Add labels and legends to figure
plt.legend(loc='lower left')
plt.grid(which='both')
plt.title("$|G(\omega)|$ - solid, $||\mathbf{H}(\omega)||_2$ - dashed")
plt.xlabel("$\omega / (4 \pi \\beta ) $")
plt.ylabel("dB")
plt.xlim((3e-3, 1))
plt.ylim((-240, 20))
plt.gcf().tight_layout()

# sphinx_gallery_thumbnail_number = 2