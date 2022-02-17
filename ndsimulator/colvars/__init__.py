"""
Collective Variables
~~~~~~~~~~~~~~~~~~~~
This module defines all the build-in collective variables

The way to add new collective variables is to write a classs
from the base class `Colvar`

The class has to have two functions: `compute(self, x)`
and `jacobian(self, x)`.  The former reads in n-dimensional vector
and return a two-dimensional one. The later reads in n-dimensional
vector and return 5x2 (colxrow) matrix.


.. highlight:: python
.. code-block:: python

   from ndsimulator.colvars import Colvar
   class NewColvar(Colvar):
      colvardim = 1 # the output dimension
      jacobian = np.zeros(3) # if the jacobian is a constant
   
      def compute(self, x):
          ...
          return c

      def jacobian(self, x):
          ...
          return j

1. Make sure this function can be imported in your current python environment. 

   a. Put this code under your running environment with a filename `my_colvar.py`

   b. prepend the directory that contains the `my_colvar.py`  to your `PYTHONPATH`

2. specify it in the yaml config.

   .. highlight:: yaml
   .. code-block:: yaml
   
      colvar: my_colvar.NewColvar
      true_colvar: my_colvar.NewColvar
"""

# from ndsimulator.control import Control

from .two import *
from .five import *
from .path import *
from .pytorch import *
from .misc import *
from .colvar import Colvar
from ._build import colvar_from_config
