#####################################################################
Welcome to the Control-Bounded A/D Conversion Toolbox's Documentation
#####################################################################

This is the official documentation for the `control-bounded analog-to-digital conversion (cbadc)
<https://pypi.org/project/cbadc/>`_ python package, which is a **toolbox** intended
to aid and inspire the **creation** of **control-bounded analog-to-digital (A/D) converters**. 

The cbadc toolbox enables you to:

* **Generate** transfer functions of analog systems and digital estimator parametrizations.
* **Estimate** samples :math:`\hat{\mathbf{u}}(t)` from control signals :math:`\mathbf{s}[k]`.
* **Simulate** analog system and digital control interactions.

Contents
========

This documentation is structured in four parts. 

.. toctree::
   :maxdepth: 1
   
   control-bounded_converters
   tutorials/index
   api/api
   datasets/index
   
where :doc:`control-bounded_converters` gives an overview of the control-bounded
A/D conversion's main concepts and terminology, :doc:`tutorials/index` provide
tutorials demonstrating common use cases, :doc:`datasets/index` provide
interfaces to simulation results and hardware prototypes, and the :doc:`api/api` chapter 
contains the package documentation.

Installation
============

Install  `cbadc <https://pypi.org/project/cbadc/>`_ by typing::

   pip install cbadc

.. or alternatively::

..    conda install cbadc

into your console.

.. note:: Currently cbadc is only supported for Python3.8 and later.

Getting Started
===============
If you are familiar with the basics of control-bounded A/D conversion a good place
to start is the :ref:`getting_started` tutorials.

Alternatively, for a brief crash course on control-bounded A/D conversion first 
check out the :doc:`control-bounded_converters` chapter.

Github
======

The project is hosted at `github.com/hammal/cbadc <https://github.com/hammal/cbadc>`_
where code contributions are welcome.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
