******************************
Control-Bounded A/D Conversion
******************************

Analog-to-digital (A/D) conversion is an ubiquitous part of most of todays
electronic devices as it interfaces the analog world we live in with the
digital domain. Control-bounded A/D conversion is a type of A/D conversion
that indirectly converts analog signals into their digital counterpart by
stabilizing an analog system using digital control.

===========================
Conventional A/D Conversion
===========================
Traditionally, the A/D conversion process is divided into three steps,
where we convert a given analog input signal :math:`u(t)` into a
digital estimate :math:`\hat{u}[k]` by applying

1. an analog preconditioning (anti-aliasing) filter,
2. a sampler (sampling in time),
3. and a quantizer, i.e., mapping discrete-time samples into bits.

.. image:: images/conventionalConversion.svg
   :width: 400
   :alt: Conventional A/D conversion.
   :align: center

The three steps are additionally shown in the figure above, where we see the
preconditioning, sampling, and quantization steps from
left to right.

===================
A new A/D interface
===================

The control-bounded A/D conversion concept approaches this conversion process
differently, as outlined in the figure below.

.. image:: images/controlBoundedConverterOverview.svg
   :width: 600
   :alt: A general control-bounded A/D converter.
   :align: center

Specifically, the conversion process is divided into three main components.

The :doc:`analog system (AS) <analog_system>`, preconditions the input signal
by amplifying desired, while suppressing undesired, signal characteristics.
Note that the AS is a fully analog system. Additionally, the A/D
converter's overall conversion performance is directly linked to the amount of
amplification provided in this stage.

The :doc:`digital control (DC) <digital_control>` stabilizes the AS by observing
a sampled and quantized version of the internal AS states and, based on these
observations, provide a control signal :math:`\mathbf{s}[k]` which is fed back
into the AS (via a digital-to-analog (D/A) conversion step). The goal of DC is to
(physically) bound the internal AS states. The DC ability to bound the AS states
will directly affect the overall conversion performance. In contrast to the AS,
the DC is a fully digital system, operating in synchronization with a global clock,
with the exception of the control signal contribution :math:`\mathbf{s}(t)` which
is a continuous-time analog version of the control signal :math:`\mathbf{s}[k]`.

Finally, the :doc:`digital estimator (DE) <digital_estimator>` provides samples of a
continuous-time estimate :math:`\hat{u}(t)` given the control signal
:math:`\mathbf{s}[k]` and the knowledge of the general AS and DC parametrization.
In many ways, the DE is the heart of the control-bounded A/D conversion scheme as it
is able to produce estimates for essentially arbitrary AS and DC combinations. The
internals of the DE might seem overwhelmingly complicated at first glance. After all,
this is the result of many years of theoretical work (see references). However, the good
news are:

- For uniform samples, the DE reduces to a linear filter and can therefore be implemented
  with a complexity comparable to a :math:`\Delta\Sigma` decimation filter.
- :py:mod:`cbadc.digital_estimator` implements all the necessary computations and can,
  therefore, for a given AS and DC, provide you with the resulting filter coefficient
  of a DE.

In summary, the control-bounded A/D converter principle approaches A/D
conversion unconventionally as, instead of breaking down the conversion
into sampling and quantization steps, we focus on stabilizing an analog system
using a digital control. In this view, conversion performance takes on a new shape as
increasing the AS amplification in combination with a DC that enforces tighter control
implies an increased A/D conversion performance. This results in
a whole new analog design space with a considerable more unconstrained A/D interface,
which in turn provides design opportunities for the analog designer.

-------------------------------------------
Relation to :math:`\Delta\Sigma` Modulators
-------------------------------------------

But wait, the figure above looks like a continuous-time :math:`\Delta\Sigma` modulator?
Is it just the same?

Not quite. It is true that, in the scalar input, state vector, and control signal case,
there is no difference between the AS and DC of the control-bounded architecture presented
above and a first order continuous-time :math:`\Delta\Sigma` modulator. However, the DE
filter is derived in a conceptually different way compared to a decimation filer.
Furthermore, for a general AS and DC the closest :math:`\Delta\Sigma` concept is the
MASH :math:`\Delta\Sigma` modulator concept. However, the MASH concept requires a
digital cancellation logic which fundamentally constrains how the AS and DC can be
interconnected. In comparison, the control-bounded DE pose no such restrictions but
instead enables a vast AS and DC design space.

Interestingly, :doc:`any MASH converter can be written in the form of a control-bounded A/D
converter<MASH_delta_sigma>` and thereby benefit from using the simple design procedure of the
DE as opposed to the conventional way of a digital cancellation logic followed by a
decimation filter.

======================
References
======================

This was a brief introduction to the control-bounded A/D conversion concept.
For a more in-depth description, consider the following references.


* `H. Malmberg, Control-bounded converters, Ph.D. dissertation, Dept. Inf. Technol. Elect. Eng., ETH Zurich, 2020.  <https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf>`_
* `H.-A. Loeliger, H. Malmberg, and G. Wilckens, Control-bounded analog-to-digital conversion: transfer function analysis, proof of concept, and digital filter implementation, arXiv:2001.05929 <https://arxiv.org/abs/2001.05929>`_
* `H.-A. Loeliger and G. Wilckens, Control-based analog-to-digital conversion without sampling and quantization, 2015 Information Theory & Applications Workshop (ITA), UCSD, La Jolla, CA, USA, Feb. 1-6, 2015 <https://ieeexplore.ieee.org/document/7308975>`_
* `H.-A. Loeliger, L. Bolliger, G. Wilckens, and J. Biveroni, Analog-to-digital conversion using unstable filters, 2011 Information Theory & Applications Workshop (ITA), UCSD, La Jolla, CA, USA, Feb. 6-11, 2011 <https://ieeexplore.ieee.org/abstract/document/5743620>`_

======================
What's next
======================

Next follows a series of tutorials demonstrating common use cases of the cbadc package.
In particular, consider the :ref:`getting_started`.

.. toctree::
   :hidden:

   analog_system
   digital_control
   digital_estimator
   MASH_delta_sigma
