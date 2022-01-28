-------------------
Analog System Model
-------------------

The analog system (AS) purpose is to amplify the characteristics of the class of signals we are converting.
This typically means amplifying a frequency band of interest while suppressing out-of-band signals. To this
end, the analog system is implemented as a continuous-time analog filter governed by the differential equations

:math:`\dot{\mathbf{x}}(t) = \mathbf{A} \mathbf{x}(t) + \mathbf{B} \mathbf{u}(t) + \mathbf{\Gamma} \mathbf{s}(t)`

where we refer to

* :math:`\mathbf{A} \in \mathbb{R}^{N \times N}` as the system matrix,
* :math:`\mathbf{B} \in \mathbb{R}^{N \times L}` as the input matrix,
* :math:`\mathbf{\Gamma} \in \mathbb{R}^{N \times M}` as the control input matrix,
* :math:`\mathbf{x}(t)\in\mathbb{R}^{N}` as the state vector of the system,
* :math:`\mathbf{u}(t)\in\mathbb{R}^{L}` as the vector-valued, continuous-time, analog input signal,
* and :math:`\mathbf{s}(t)\in\mathbb{R}^{M}` as the vector-valued control signal.


The analog system also has two (possibly vector-valued) outputs, namely

* The control observation :math:`\tilde{\mathbf{s}}(t)=\tilde{\mathbf{\Gamma}}^\mathsf{T} \mathbf{x}(t)` and
* The signal observation :math:`\mathbf{y}(t) = \mathbf{C}^\mathsf{T} \mathbf{x}(t)`

where

* :math:`\tilde{\mathbf{\Gamma}}^\mathsf{T}\in\mathbb{R}^{\tilde{M} \times N}` is the control observation matrix
* and :math:`\mathbf{C}^\mathsf{T}\in\mathbb{R}^{\tilde{N} \times N}` is the signal observation matrix.

An overview of the system relations are given in the figure below

.. image:: images/generalSSM.svg
    :width: 600
    :align: center
    :alt: The general analog system


We model an analog system using :py:class:`cbadc.analog_system.AnalogSystem`.

.. seealso::
    :py:class:`cbadc.analog_system.ChainOfIntegrators`
    and :py:class:`cbadc.analog_system.LeapFrog` which are derived classes from
    :py:class:`cbadc.analog_system.AnalogSystem`.

Given an instantiated analog system we can also manually evaluate the time derivative
above by invoking the :py:func:`cbadc.analog_system.AnalogSystem.derivative` function.
Furthermore, the control and signal observation are obtained by
:py:func:`cbadc.analog_system.AnalogSystem.control_observation` and
:py:func:`cbadc.analog_system.AnalogSystem.signal_observation` respectively.

^^^^^^^^^^^^^^^^^^^^^^^^
Transfer Function Matrix
^^^^^^^^^^^^^^^^^^^^^^^^

The analog system can also be described by its corresponding analog transfer function matrix

:math:`\mathbf{G}(\omega) = \mathbf{C}^\mathsf{T} \left(\mathbf{A} - i \omega \mathbf{I}_N\right)^{-1} \mathbf{B}`

which can be invoked by :py:func:`cbadc.analog_system.AnalogSystem.transfer_function_matrix`.
