NDSimulator
===========

Documentation: https://ndsimulator.rtfd.io

A molecular dynamics tool for toy model systems.

Features
--------

-  model a particle on a N-dimensional energy landscape
-  Molecular dynamics, metadynamics, umbrella sampling are implemented.
-  Allow users to define collective variables
-  Langevin (1st and 2nd order), velocity rescale and NVE ensemble are
   available for MD

Installation
------------

1. from GitHub

.. highlight:: bash
.. code:: bash

   git clone git@github.com:mir-group/NDSimulator.git
   pip install -e ./

2. from pip

::

   pip install ndsimulator

Prerequisits
~~~~~~~~~~~~

-  Python3.8
-  matplotlib
-  NumPy
-  pyfile-utils

Testing
~~~~~~~

.. code:: bash

   pytest tests

Commandline interface
---------------------

The inputs specifying the system, method, parameters, and visualization
options can be received via an inputs file

.. code:: bash

   ndsimulator examples/2d-md.yaml

.. figure:: https://github.com/mir-group/NDSimulator/raw/main/reference/md_gamma_0.1/oneplot.png
