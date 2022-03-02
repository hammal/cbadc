:orphan:



.. _sphx_glr_tutorials:

*********
Tutorials
*********

Here we demonstrate common use cases for the cbadc package.


.. raw:: html

    <div class="sphx-glr-clear"></div>



.. _sphx_glr_tutorials_a_getting_started:

.. _getting_started:

-------------------------
Getting Started
-------------------------

Next follows three basic examples demonstrating how to

1. simulate analog system and digital control interactions,
2. estimate samples :math:`\hat{\mathbf{u}}(t)` from control signals :math:`\mathbf{s}[k]`,
3. and evaluate and visualize transfer functions from analog systems and digital estimators.



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to simulate the interactions between an analog system and a digital cont...">

.. only:: html

 .. figure:: /tutorials/a_getting_started/images/thumb/sphx_glr_plot_b_simulate_a_control_bounded_adc_thumb.png
     :alt: Simulating a Control-Bounded ADC

     :ref:`sphx_glr_tutorials_a_getting_started_plot_b_simulate_a_control_bounded_adc.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/a_getting_started/plot_b_simulate_a_control_bounded_adc

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Converting a stream of control signals into a estimate samples.">

.. only:: html

 .. figure:: /tutorials/a_getting_started/images/thumb/sphx_glr_plot_c_digital_estimator_thumb.png
     :alt: Digital Estimation

     :ref:`sphx_glr_tutorials_a_getting_started_plot_c_digital_estimator.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/a_getting_started/plot_c_digital_estimator

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to visualize the related transfer functions of the analog system ...">

.. only:: html

 .. figure:: /tutorials/a_getting_started/images/thumb/sphx_glr_plot_d_transfer_function_thumb.png
     :alt: Transfer Functions

     :ref:`sphx_glr_tutorials_a_getting_started_plot_d_transfer_function.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/a_getting_started/plot_d_transfer_function
.. raw:: html

    <div class="sphx-glr-clear"></div>



.. _sphx_glr_tutorials_b_high_level_simulation:

----------------------
High Level Simulations
----------------------



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial we investigate different estimator implementation techniques and compare their...">

.. only:: html

 .. figure:: /tutorials/b_high_level_simulation/images/thumb/sphx_glr_plot_a_compare_estimator_thumb.png
     :alt: Compare Estimators

     :ref:`sphx_glr_tutorials_b_high_level_simulation_plot_a_compare_estimator.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/b_high_level_simulation/plot_a_compare_estimator

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We demonstrate how to set up the FIR filter implementation.">

.. only:: html

 .. figure:: /tutorials/b_high_level_simulation/images/thumb/sphx_glr_plot_b_FIR_Filtering_thumb.png
     :alt: Digital Estimator as a FIR Filter

     :ref:`sphx_glr_tutorials_b_high_level_simulation_plot_b_FIR_Filtering.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/b_high_level_simulation/plot_b_FIR_Filtering

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial we demonstrate how to configure the digital estimator for downsampling.">

.. only:: html

 .. figure:: /tutorials/b_high_level_simulation/images/thumb/sphx_glr_plot_c_downsample_thumb.png
     :alt: Downsampling

     :ref:`sphx_glr_tutorials_b_high_level_simulation_plot_c_downsample.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/b_high_level_simulation/plot_c_downsample

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial we demonstrate we can evaluate the FIR filter using fixed-point precision arit...">

.. only:: html

 .. figure:: /tutorials/b_high_level_simulation/images/thumb/sphx_glr_plot_d_fixed_point_aritmetics_thumb.png
     :alt: Fixed-Point Arithmetics

     :ref:`sphx_glr_tutorials_b_high_level_simulation_plot_d_fixed_point_aritmetics.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/b_high_level_simulation/plot_d_fixed_point_aritmetics

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial we show how to use switched-capacitor digital control in combination with a co...">

.. only:: html

 .. figure:: /tutorials/b_high_level_simulation/images/thumb/sphx_glr_plot_e_switched_capacitor_digital_control_thumb.png
     :alt: Switched-Capacitor Digital Control

     :ref:`sphx_glr_tutorials_b_high_level_simulation_plot_e_switched_capacitor_digital_control.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/b_high_level_simulation/plot_e_switched_capacitor_digital_control

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial we investigate the :py:func:`cbadc.specification.get_chain_of_integrator` and ...">

.. only:: html

 .. figure:: /tutorials/b_high_level_simulation/images/thumb/sphx_glr_plot_f_designing_a_leap_frog_to_specifications_thumb.png
     :alt: Designing for a Target Specification

     :ref:`sphx_glr_tutorials_b_high_level_simulation_plot_f_designing_a_leap_frog_to_specifications.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/b_high_level_simulation/plot_f_designing_a_leap_frog_to_specifications

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Calibrating Digital Estimator for Mismatch">

.. only:: html

 .. figure:: /tutorials/b_high_level_simulation/images/thumb/sphx_glr_plot_g_calibrating_mismatch_thumb.png
     :alt: Calibrating Digital Estimator for Mismatch

     :ref:`sphx_glr_tutorials_b_high_level_simulation_plot_g_calibrating_mismatch.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/b_high_level_simulation/plot_g_calibrating_mismatch

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Calibrating an Unknown Op-Amp Pole from Data">

.. only:: html

 .. figure:: /tutorials/b_high_level_simulation/images/thumb/sphx_glr_plot_h_calibrating_from_numpy_simulator_thumb.png
     :alt: Calibrating an Unknown Op-Amp Pole from Data

     :ref:`sphx_glr_tutorials_b_high_level_simulation_plot_h_calibrating_from_numpy_simulator.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/b_high_level_simulation/plot_h_calibrating_from_numpy_simulator
.. raw:: html

    <div class="sphx-glr-clear"></div>



.. _sphx_glr_tutorials_c_circuit_level:

-------------------------
Circuit Level Simulations
-------------------------



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We demonstrate how an analog system can be transformed into a boilerplate verilog-ams circuit m...">

.. only:: html

 .. figure:: /tutorials/c_circuit_level/images/thumb/sphx_glr_plot_a_analog_system_thumb.png
     :alt: Analog System

     :ref:`sphx_glr_tutorials_c_circuit_level_plot_a_analog_system.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/c_circuit_level/plot_a_analog_system

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial we will combine analog systems and digital controls to which we refer to as an...">

.. only:: html

 .. figure:: /tutorials/c_circuit_level/images/thumb/sphx_glr_plot_b_analog_frontend_thumb.png
     :alt: Analog Frontend

     :ref:`sphx_glr_tutorials_c_circuit_level_plot_b_analog_frontend.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/c_circuit_level/plot_b_analog_frontend

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial we expand on the ./plot_b_analog_frontend tutorial by adding a clock and input...">

.. only:: html

 .. figure:: /tutorials/c_circuit_level/images/thumb/sphx_glr_plot_c_testbench_thumb.png
     :alt: Testbench

     :ref:`sphx_glr_tutorials_c_circuit_level_plot_c_testbench.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/c_circuit_level/plot_c_testbench
.. raw:: html

    <div class="sphx-glr-clear"></div>



.. _sphx_glr_tutorials_d_further:

----------------
Further Examples
----------------

These are fun and otherwise useful examples showcasing the cbadc package.



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This notebook demonstrate the main functionality of the :py:class:`cbadc.fom.MurmannSurvey` con...">

.. only:: html

 .. figure:: /tutorials/d_further/images/thumb/sphx_glr_plot_a_Murmann_ADC_survey_thumb.png
     :alt: The Murmann ADC Survey

     :ref:`sphx_glr_tutorials_d_further_plot_a_Murmann_ADC_survey.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/d_further/plot_a_Murmann_ADC_survey

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This notebook shows examples of artsy plots made from data generated with the control-bounded c...">

.. only:: html

 .. figure:: /tutorials/d_further/images/thumb/sphx_glr_plot_z_artsy_sine_thumb.png
     :alt: Creating Artsy Plots

     :ref:`sphx_glr_tutorials_d_further_plot_z_artsy_sine.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/d_further/plot_z_artsy_sine
.. raw:: html

    <div class="sphx-glr-clear"></div>



.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-gallery


  .. container:: sphx-glr-download sphx-glr-download-python

    :download:`Download all examples in Python source code: tutorials_python.zip </tutorials/tutorials_python.zip>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

    :download:`Download all examples in Jupyter notebooks: tutorials_jupyter.zip </tutorials/tutorials_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
