Welcome to Control-Bounded A/D Conversion Toolbox's Documentation
=================================================================

This is the official documentation for the `cbadc
<https://pypi.org/project/cbadc/>`_ python package which is a **toolbox** intended
to aid and inspire the **creation** of **control-bounded analog-to-digital (A/D) converters**. 

The cbadc toolbox enables you to:

* **Generate** transfer functions of analog systems and/or digital estimator parametrizations.
* **Estimate** samples :math:`\hat{\mathbf{u}}(t)` from control signals :math:`\mathbf{s}[k]`.
* **Simulate** analog system and digital control interactions.



Contents
========

This documentation is structured in three parts. 

.. toctree::
   :maxdepth: 1
   
   control-bounded_converters
   auto_examples/index
   api/api

where :doc:`control-bounded_converters` gives an overview of the control-bounded
A/D conversion's main concepts and terminology, :doc:`auto_examples/index` provides 
code examples that demonstrate common use cases, and :doc:`api/api` contains the 
full package documentation.

Installation
============

Install  `cbadc <https://pypi.org/project/cbadc/>`_ by typing::

   pip install cbadc

.. or alternatively::

..    conda install cbadc

into your console.

Getting Started
===============
If you are familiar with the basics of control-bounded A/D conversion a good place
to start is the :ref:`getting_started`.

Alternatively, for a brief crash course on control-bounded A/D conversion first 
check out :doc:`control-bounded_converters`.

Github
======

The project is hosted at `github.com/hammal/cbadc <https://github.com/hammal/cbadc>`_
where code contributions are welcome.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
