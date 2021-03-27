Welcome to Control-Bounded A/D Conversion Toolbox's Documentation
=================================================================

This is the official documentation for the `cbadc
<https://pypi.org/project/cbadc/>`_ python package which is a toolbox intended
to ease the design of control-bounded analog-to-digital (A/D) converters. 

This python module enables you to

* Reconstruct samples from control signals using the :doc:`digitalEstimator`.
* Simulate :doc:`analogSystem` and :doc:`digitalControl` interactions for a given :doc:`analogSignal`.

The control-bounded A/D conversion concept is a new A/D conversion paradigm
reminiscent of the delta-sigma modulator conversion architecture. An in depth
introduction to control-bounded A/D conversion can be found here 
`control-bounded converters
<https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/469192/control-bounded_converters_a_dissertation_by_hampus_malmberg.pdf?sequence=1&isAllowed=y#page=28/>`_.

The general structure of this documentation is given in :ref:`_contents`. Regardless, we want to make
the reader aware of the two main sections:

* :doc:`control-bounded_converters` which gives a detailed documentation of the included classes and structure of the package
   as well as describing the control-bounded conversion concepts.  
* :doc:`tutorials/tutorials` which provides plenty of code examples demonstrating the main functionality of the package.
* something with transfer functions and analytical...

Installation
-------------

You simply install the package by typing::

   pip install cbadc

or alternatively::

   conda install cbadc

into your console.

Github
-------

The projects source files can be found at `github.com/hammal/cbadc <https://github.com/hammal/cbadc>`_
and contributions are always welcome.

.. _contents:

Contents:
-----------
.. toctree::
   :maxdepth: 2
   
   control-bounded_converters
   auto_examples/index
   api/api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
