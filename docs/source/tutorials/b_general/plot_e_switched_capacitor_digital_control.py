"""
Switched-Capacitor Digital Control
==================================

In this tutorial we show how to use switched-capacitor digital control
in combination with a continuous-time system.
"""
import cbadc
import scipy
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Integration Node
# ----------------
#
# Initially we consider a integration node as shown in the figure below.
#
# .. image:: /images/RC-amp.svg
#    :width: 500
#    :align: center
#    :alt: The RC integrator with switched-capacitor digital control.
#
# Which is goverened by the differential equations
#
# :math:`\dot{v}_{x_{\ell}}(t) = \frac{1}{C_{x_{\ell}}} \left( \frac{v_{\Gamma_{\ell}}(t) - v_{s_\ell}(t)}{R_{s}} - \frac{1}{R_{\beta_\ell}}  v_{x_{\ell - 1}}(t) \right)`
#
# :math:`\dot{v}_{\Gamma_{\ell}}(t) = \frac{1}{2R_{s} C_{\Gamma_{\ell}}} \left( v_{s_\ell}(t) - v_{\Gamma_{\ell}}(t) \right)`
#
# during phase :math:`\phi_1` and
#
# :math:`\dot{v}_{x_{\ell}}(t)  =  - \frac{1}{R_{\beta_\ell} C_{x_{\ell}}} v_{x_{\ell - 1}}(t)`
#
# :math:`\dot{v}_{\Gamma_{\ell}}(t) = - \frac{1}{2 R_{s} C_{\Gamma_{\ell}}}  v_{\Gamma_{\ell}}(t)`
#
# during phase :math:`\bar{\phi}_1` where a full control phase :math:`T = T_{\phi_1} + T_{\bar{\phi}_1}`.
#
# To simulate this control-bounded converter we need to specify the whole system which for this
# tutorial will be a chain-of-integrators system as shown below.
#
# .. image:: /images/chainOfIntegratorsGeneral.svg
#    :width: 1500
#    :align: center
#    :alt: The chain-of-integrators ADC.
#
# where the digital controls are implemented as above.
# Thus the system equations can be written as

N = 4
M = N

C_x = 1e-9
C_Gamma = C_x / 2
R_s = 1e1
R_beta = 1e3

beta = 1 / (R_beta * C_x)
T = 1 / (2 * beta)

A = beta * np.eye(N, k=-1)
B = np.zeros(N)
B[0] = beta
CT = np.eye(N)

###############################################################################
# Simplifying the Differential Equation
# -------------------------------------
#
# Due to the virtual ground of the op-amp integrator we can treat the
# second differential equation (the one involving :math:`v_{\Gamma_\ell}(t)`)
# seperately. Specifically, if we assume the
# capacitor :math:`C_{\Gamma}` empties completely during :math:`\bar{\phi}_1`,
# and that :math:`v_{s_\ell} = \{\pm 1\}` for all :math:`t`, the digital
# control's effect on the analog system can be written as
#
# :math:`v_{\Gamma_\ell}(t) - v_{s_\ell}(t) = v_{s_\ell}(t) \cdot e^{-t/\tau}`
#
# where :math:`\tau=R_s C_{\Gamma_\ell}`.
#
# This can be modeled by instantiating the digital control
# :class:`cbadc.digital_control.DigitalControl`
# using the impulse response :func:`cbadc.digital_control.RCImpulseResponse`
# as

impulse_response = cbadc.digital_control.RCImpulseResponse(R_s * C_Gamma)
digital_control_sc = cbadc.digital_control.DigitalControl(
    T, M, impulse_response=impulse_response
)

Gamma = 1 / (R_s * C_x) * np.eye(M)
Gamma_tildeT = -np.eye(M)

analog_system_sc = cbadc.analog_system.AnalogSystem(
    A, B, CT, Gamma, Gamma_tildeT)

print(digital_control_sc)
print(analog_system_sc)

###############################################################################
# Visualizing the State Trajectories
# ----------------------------------
#
OSR = 32
amplitude = 1.0
analog_signal = cbadc.analog_signal.Sinusodial(amplitude, 1 / T / (OSR << 4))
Ts = T / 100.0
size = 1 << 12

simulator_sc = cbadc.simulator.extended_simulation_result(
    cbadc.simulator.StateSpaceSimulator(
        analog_system_sc, digital_control_sc, [analog_signal], Ts=Ts
    )
)


analog_system_ref = cbadc.analog_system.AnalogSystem(
    A, B, CT, np.eye(N) * beta, Gamma_tildeT
)
digital_control_ref = cbadc.digital_control.DigitalControl(T, M)
simulator_ref = cbadc.simulator.extended_simulation_result(
    cbadc.simulator.StateSpaceSimulator(
        analog_system_ref,
        digital_control_ref,
        [analog_signal],
        Ts=Ts,
    )
)

states = np.zeros((size, N))
states_ref = np.zeros_like(states)
t = Ts * np.arange(size)

# Simulations
for time_step in cbadc.utilities.show_status(range(size)):
    states[time_step, :] = next(simulator_sc)["analog_state"]
    states_ref[time_step, :] = next(simulator_ref)["analog_state"]

# Plot state trajectories
for index in range(N):
    plt.figure()
    plt.title("Analog state trajectories for " + f"$x_{index + 1}(t)$")
    plt.plot(t / T, states[:, index], label="SC")
    plt.plot(t / T, states_ref[:, index], label="ref")
    plt.grid(b=True, which="major", color="gray", alpha=0.6, lw=1.5)
    plt.xlabel("$t/T$")
    plt.legend()


###############################################################################
# Filter Coefficients
# ----------------------------------------
#
K1 = 1 << 8
K2 = K1
eta2 = (
    np.linalg.norm(
        analog_system_sc.transfer_function_matrix(
            np.array([2 * np.pi / T / OSR]))
    ).flatten()
    ** 2
)

# prepending an anti-aliasing filter
# omega_3dB = 2 * np.pi / T / 16
# wp = omega_3dB / 2.0
# ws = omega_3dB
# gpass = 1.0
# gstop = 60
# filter = cbadc.analog_system.IIRDesign(wp, ws, gpass, gstop, ftype="ellip")

# Post-filtering FIR filter
fir_filter_numtaps = K1 + K2
f_cutoff = 1.0 / OSR * 2
fir_filter = scipy.signal.firwin(fir_filter_numtaps, f_cutoff)


digital_estimator_sc = cbadc.digital_estimator.FIRFilter(
    # cbadc.analog_system.chain([filter, analog_system_sc]),
    analog_system_sc,
    digital_control_sc,
    eta2,
    K1,
    K2,
)

# Apply FIR filter
digital_estimator_sc.convolve(fir_filter)

digital_estimator_ref = cbadc.digital_estimator.FIRFilter(
    # cbadc.analog_system.chain([filter, analog_system_ref]),
    analog_system_ref,
    digital_control_ref,
    eta2,
    K1,
    K2,
)

# Apply FIR filter
digital_estimator_ref.convolve(fir_filter)

plt.figure()
plt.semilogy(
    np.arange(-K1, K2),
    np.linalg.norm(np.array(digital_estimator_sc.h[0, :, :]), axis=1)[:],
    label="Switched Capacitor Filter",
)
plt.semilogy(
    np.arange(-K1, K2),
    np.linalg.norm(np.array(digital_estimator_ref.h[0, :, :]), axis=1)[:],
    label="Ref Filter",
)

plt.legend()
plt.xlabel("filter tap k")
plt.ylabel("$|| \mathbf{h} [k]||_2$")
plt.xlim((-K1, K2))
# plt.ylim((1e-16, 1))
plt.grid(which="both")

# Logspace frequencies
frequencies = np.logspace(-3, 0, 100)
omega = 4 * np.pi * beta * frequencies

plt.figure()
plt.semilogx(
    omega / (2 * np.pi),
    20
    * np.log10(np.abs(digital_estimator_sc.signal_transfer_function(omega))).flatten(),
    label="$STF(\omega)$ SC",
)
plt.semilogx(
    omega / (2 * np.pi),
    20
    * np.log10(np.abs(digital_estimator_ref.signal_transfer_function(omega))).flatten(),
    label="$STF(\omega)$ ref",
)
plt.semilogx(
    omega / (2 * np.pi),
    20
    * np.log10(
        np.linalg.norm(
            digital_estimator_sc.noise_transfer_function(omega)[
                :, 0, :], axis=0
        )
    ),
    "--",
    label="$ || NTF(\omega) ||_2 $ SC",
)
plt.semilogx(
    omega / (2 * np.pi),
    20
    * np.log10(
        np.linalg.norm(
            digital_estimator_ref.noise_transfer_function(omega)[
                :, 0, :], axis=0
        )
    ),
    "--",
    label="$ || NTF(\omega) ||_2 $ ref",
)
# Add labels and legends to figure
plt.legend()
plt.grid(which="both")
plt.title("Signal and noise transfer functions")
plt.xlabel("$f$ [Hz]")
plt.ylabel("dB")
# plt.xlim((1e2, 5e3))
plt.gcf().tight_layout()

###############################################################################
# Mismatch Sensitivity to Switch Resistance
# -----------------------------------------
#
#

mismatch_in_percent = np.arange(3) * 10

size = 1 << 14
t = np.arange(size) * T

estimates = []

for mismatch in mismatch_in_percent:
    digital_estimator_sc = cbadc.digital_estimator.FIRFilter(
        # cbadc.analog_system.chain([filter, analog_system_sc]),
        analog_system_sc,
        cbadc.digital_control.DigitalControl(
            T,
            M,
            impulse_response=cbadc.digital_control.RCImpulseResponse(
                R_s * C_Gamma),
        ),
        eta2,
        K1,
        K2,
    )

    # Apply FIR filter
    digital_estimator_sc.convolve(fir_filter)

    digital_estimator_sc(
        cbadc.simulator.StateSpaceSimulator(
            cbadc.analog_system.AnalogSystem(
                A,
                B,
                CT,
                1 / ((1 + mismatch / 100) * R_s * C_x) * np.eye(M),
                Gamma_tildeT,
            ),
            cbadc.digital_control.DigitalControl(
                T,
                M,
                impulse_response=cbadc.digital_control.RCImpulseResponse(
                    (1 + mismatch / 100) * R_s * C_Gamma
                ),
            ),
            [analog_signal],
            pre_compute_control_interactions=False,
        )
    )
    u_hat = np.zeros(size)
    for index in cbadc.utilities.show_status(range(size)):
        u_hat[index] = next(digital_estimator_sc)
    estimates.append(u_hat)

# Plot estimates in time domain
plt.figure()
plt.title("Estimates")
for index, mismatch in enumerate(mismatch_in_percent):
    plt.plot(t / T, estimates[index], label=f"Mismatch R_s {mismatch}%")
plt.grid(b=True, which="major", color="gray", alpha=0.6, lw=1.5)
plt.xlabel("$t/T$")
plt.xlim((K1 + K2, K1 + K2 + 100))
plt.legend()

# Plot estimates PSD
plt.figure()
plt.title("Estimates PSD")
for index, mismatch in enumerate(mismatch_in_percent):
    f, psd = cbadc.utilities.compute_power_spectral_density(
        estimates[index][K1 + K2:], fs=1.0 / T
    )
    plt.semilogx(f, 10 * np.log10(psd), label=f"Mismatch R_s {mismatch}%")
plt.grid(b=True, which="major", color="gray", alpha=0.6, lw=1.5)
plt.xlabel("f [Hz]")
plt.ylabel("V^2/Hz [dB]")
plt.legend()

# ###############################################################################
# # Clock Jitter Sensitivity
# # ------------------------
# #
# jitter_std = np.power(10.0, np.arange(-6, 0)) * T
# # jitter_std = np.arange(3) * T * 0.3
# clock_jitter = [lambda: (np.random.random() - 0.5) * std for std in jitter_std]

# size = 1 << 14
# t = np.arange(size) * T

# estimates = []
# estimates_ref = []

# for jitter in clock_jitter:
#     digital_estimator_sc = cbadc.digital_estimator.FIRFilter(
#         # cbadc.analog_system.chain([filter, analog_system_sc]),
#         analog_system_sc,
#         cbadc.digital_control.DigitalControl(
#             T,
#             M,
#             impulse_response=cbadc.digital_control.RCImpulseResponse(R_s * C_Gamma),
#         ),
#         eta2,
#         K1,
#         K2,
#     )

#     # Apply FIR filter
#     # digital_estimator_sc.convolve(fir_filter)

#     digital_estimator_sc(
#         cbadc.simulator.StateSpaceSimulator(
#             analog_system_sc,
#             cbadc.digital_control.DigitalControl(
#                 T,
#                 M,
#                 impulse_response=cbadc.digital_control.RCImpulseResponse(R_s * C_Gamma),
#             ),
#             [analog_signal],
#             clock_jitter=jitter,
#         )
#     )

#     digital_estimator_ref = cbadc.digital_estimator.FIRFilter(
#         # cbadc.analog_system.chain([filter, analog_system_ref]),
#         analog_system_ref,
#         cbadc.digital_control.DigitalControl(T, M),
#         eta2,
#         K1,
#         K2,
#     )

#     # Apply FIR filter
#     # digital_estimator_ref.convolve(fir_filter)

#     digital_estimator_ref(
#         cbadc.simulator.StateSpaceSimulator(
#             analog_system_ref,
#             cbadc.digital_control.DigitalControl(T, M),
#             [analog_signal],
#             clock_jitter=jitter,
#         )
#     )

#     u_hat = np.zeros(size)
#     u_hat_ref = np.zeros_like(u_hat)
#     digital_estimator_sc.warm_up()
#     digital_estimator_ref.warm_up()
#     for index in cbadc.utilities.show_status(range(size)):
#         u_hat[index] = next(digital_estimator_sc)
#         u_hat_ref[index] = next(digital_estimator_ref)
#     estimates.append(u_hat)
#     estimates_ref.append(u_hat_ref)

# # Plot estimates in time domain
# plt.figure()
# plt.title("Estimates")
# for index, jitter in enumerate(clock_jitter):
#     plt.plot(
#         t / T,
#         estimates[index],
#         label=f"Std / T = {np.round(jitter_std[index] / T * 100, 3)}%",
#     )
# plt.grid(b=True, which="major", color="gray", alpha=0.6, lw=1.5)
# plt.xlabel("$t/T$")
# plt.xlim((K1 + K2, K1 + K2 + 1000))
# plt.legend()

# # Plot estimates in time domain
# plt.figure()
# plt.title("Ref Estimates")
# for index, jitter in enumerate(clock_jitter):
#     plt.plot(
#         t / T,
#         estimates_ref[index],
#         label=f"Ref Std / T = {np.round(jitter_std[index] / T * 100, 3)}%",
#     )
# plt.grid(b=True, which="major", color="gray", alpha=0.6, lw=1.5)
# plt.xlabel("$t/T$")
# plt.xlim((K1 + K2, K1 + K2 + 1000))
# plt.legend()

# # Plot estimates PSD
# for index, jitter in enumerate(clock_jitter):
#     plt.figure()
#     plt.title("Estimates PSD Clock Jitter")

#     f, psd = cbadc.utilities.compute_power_spectral_density(
#         estimates[index][K1 + K2 :], fs=1.0 / T
#     )
#     f_ref, psd_ref = cbadc.utilities.compute_power_spectral_density(
#         estimates_ref[index][K1 + K2 :], fs=1.0 / T
#     )
#     plt.semilogx(
#         f,
#         10 * np.log10(psd),
#         label=f"SC (Std/T) = +- {np.round(jitter_std[index] / T * 100, 3)}%",
#     )
#     plt.semilogx(
#         f_ref,
#         10 * np.log10(psd_ref),
#         "--",
#         label=f"Ref (Std/T) = +- {np.round(jitter_std[index] / T  * 100, 3)}%",
#     )
#     plt.grid(b=True, which="major", color="gray", alpha=0.6, lw=1.5)
#     plt.xlabel("f [Hz]")
#     plt.ylabel("V^2/Hz [dB]")
#     plt.legend()
