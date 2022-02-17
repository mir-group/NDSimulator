"""
Potentials
~~~~~~~~~~

This module defines all the build-in potentials, including double well, gaussian, mueller-brown, and three-hole.

write a new python function class from the base class `Potential`

.. highlight:: python
.. code-block:: python

   from ndsimulator.potentials import Potential
   class NewPotential(Potential):
       ndim = 3 # set to None if it works for arbiturary dimension
       require_colvar = True # if it is derived from some underlying transformation
   
       def compute(self, x):
           ...
           return e, f

1. Make sure this function can be imported in your current python environment. 

   a. Put this code under your running environment with a filename `my_potential.py`

   b. prepend the directory that contains the `my_potential.py`  to your `PYTHONPATH`

2. specify it in the yaml config.

   .. highlight:: yaml
   .. code-block:: yaml
   
      potential: my_potential.NewPotential
"""

from .double_well import *
from .flat import *
from .misc import *
from .mueller import *
from .potential import *
from .three_hole import *
from ._build import potential_from_config
