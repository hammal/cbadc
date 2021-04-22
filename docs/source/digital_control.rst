---------------
Digital Control
---------------

It is the task of the digital control (DC) to stabilize the :doc:`analog system (AS) <analog_system>` by 
a digital control loop. In principle this could be done in many ways; however, here we restrict ourselves
to an additive control loop that observes sampled and quantized analog states :math:`\mathbf{x}` and feeding
back a control signal :math:`\mathbf{s}(t)`. Specifically, we consider a DC as shown in the middle of 
the figure below.

.. image:: images/control-bounded-converter-additive-control.svg
    :width: 1000
    :align: center
    :alt: The general analog system

From the figure we also see both the :doc:`AS <analog_system>` and the :doc:`digital estimator (DE) <digital_estimator>`.
Note that typically :math:`\tilde{\mathbf{s}}(t) \in \mathbb{R}^\tilde{M}`, :math:`\mathbf{s}[k] \in \mathbb{R}^M`, and 
:math:`\mathbf{s}(t) \in \mathbb{R}^M` are vector valued signals. Furthermore, the DC is mainly specified by the number
of independent controls :math:`M` and the time period between control updates :math:`T`.

The DC is modeled and specified in :py:class:`cbadc.digital_control.DigitalControl`

^^^^^^^^^^^^^^^^^^^
DC Impulse Response
^^^^^^^^^^^^^^^^^^^

The relation between the control signal :math:`\mathbf{s}[k]` and the control contribution :math:`\mathbf{s}(t)` is
determined by a digital-to-analog (D/A) conversion. Typically, :math:`\mathbf{s}[k]` is a binary vector so the D/A conversion
is the least complicated imaginable. Regardless, as :math:`\mathbf{s}[k]` is a discrete-time signal and :math:`\mathbf{s}(t)`
is a continuous-time signal the mapping is modeled as a convolution with an impulse response. We can access and evaluate a DC's 
impulse response via the :py:func:`cbadc.digital_control.DigitalControl.impulse_response` function.
