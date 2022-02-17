Tutorials
=========

Basic run
~~~~~~~~~

The umbrella sampling run basic is similar to plain molecular dynamics run, except for additional arguments to define the biases.

1. Run a demo example.

   .. highlight:: bash
   .. code-block:: bash
   
      ndsimulator examples/2d-umb.yaml
   
   The yaml file can be found on the github repo or as shown below. 
   We will explain the arguments one by one in the following section.
   
   .. literalinclude:: ../../examples/2d-umb.yaml
      :language: yaml

2. With the arguments ``root: ./reference`` and  ``run_name: umb``, the call will generate a folder in `reference/umb`

   .. highlight:: bash
   .. code-block:: bash
   
     ls reference/umb/
     $ colvar.dat  fix0.dat  force.dat  log  oneplot.png  pos.dat  thermo.dat  velocity.dat
   
   unlike metadynamics, the ``fix0.dat`` do not store any information because the harmonic penalty function is a constant.
   

3. The simulation result is automatically visualized in the ``oneplot.png``

   .. figure:: ../../reference/umb/oneplot.png

4. the arguments and equations see :ref:`Harmonic <_umb_autoclass>`
   

Tunning the parameters
~~~~~~~~~~~~~~~~~~~~~~

1. ``k``

Post Processing
~~~~~~~~~~~~~~~
