Tutorials
=========

Basic run
~~~~~~~~~

The metadynamics run basic is similar to plain molecular dynamics run, except for additional arguments to define the biases.

1. Run a demo example.

   .. highlight:: bash
   .. code-block:: bash
   
      ndsimulator examples/2d-mtd.yaml
   
   The yaml file can be found on the github repo or as shown below. 
   We will explain the arguments one by one in the following section.
   
   .. literalinclude:: ../../examples/2d-mtd.yaml
      :language: yaml

2. With the arguments ``root: ./reference`` and  ``run_name: mtd``, the call will generate a folder in `reference/mtd`

   .. highlight:: bash
   .. code-block:: bash
   
     ls reference/mtd/
     $ colvar.dat  fix0.dat  force.dat  log  oneplot.png  pos.dat  thermo.dat  velocity.dat
   
   the additional ``fix0.dat`` file store the indices, locations (x, y), and height of the deposited gaussian biases 
   in the order of ``index bias_x bias_y sigma_x sigma_y w```

3. The simulation result is automatically visualized in the ``oneplot.png``

   .. figure:: ../../reference/mtd/oneplot.png

4. the arguments and equations see :ref:`Gaussian <_mtd_autoclass>`
   

Tunning the parameters
~~~~~~~~~~~~~~~~~~~~~~

1. ``deposition``

Post Processing
~~~~~~~~~~~~~~~
