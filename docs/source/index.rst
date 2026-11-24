#####################################################################
The Control-Bounded A/D Conversion (cbadc) Toolbox API Documentation
#####################################################################

cbadc models control-bounded A/D converters with one clocked
:class:`~cbadc.frontend.Frontend`. Version 0.4.0 keeps the main path small:
define a state-space model, simulate its decisions and digital twin, then
measure the result. Physical gm-C and active-RC realizations, Wiener
reconstruction, and the optional delta-sigma bridge extend that path.

=========
Tutorials
=========

.. toctree::
    :maxdepth: 1

    tutorials/index
    ngspice-validation

======================
The cbadc Root Modules
======================

.. autosummary::
    :toctree: autosummary
    :recursive:
    :template: custom-module-template.rst


    cbadc.frontend
    cbadc.circuit
    cbadc.noise
    cbadc.estimator
    cbadc.snr
    cbadc.fom
    cbadc.delsig

======
Github
======

The project is hosted at `github.com/hammal/cbadc <https://github.com/hammal/cbadc>`_
where code contributions are welcome.

==================
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
