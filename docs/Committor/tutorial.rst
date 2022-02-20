Tutorials
=========

Basic run
~~~~~~~~~

1. Run a demo example.

   .. highlight:: bash
   .. code-block:: bash
   
      ndsimulator examples/2d-md.yaml
   
   The yaml file can be found on the github repo or as shown below. 
   We will explain the arguments one by one in the following section.
   
   .. literalinclude:: ../../examples/2d-committor.yaml
      :language: yaml

2. The result files are similar to an ordinary MD simulation, except that the run will terminate once
   one of the committing criteria is satisfied. The end of the log file will state which basin it has commit to

   .. highlight:: bash
   
   !! Commit to basin 0

   If the committed basin id is -1, it means the simulation has not found any basins before the max number of steps is reached.

3. The simulation result is automatically visualized in the ``oneplot.png``

   .. figure:: ../../reference/committor_gamma_0.1/oneplot.png
   

Tunning Parameters
~~~~~~~~~~~~~~~~~~

1. ```criteria```

   The committing criteria is not set arbiturary. One should do an unbiased MD simulation at each basin,
   and use the distribution to determine the committing criteria.
   
   Since the position distribution is a function of temperature, the committing criteria shall also be a function of the temperature.

   If the criteria is too strict, the simulation will waste a lot of time in the basin.

   If the criteria is too loose, the ridge area will not be sampled sufficiently.

2. ```steps``` and ```gamma```

   The ensemble is critical to the committor analysis. The committor function is a function of both position and velocities.
   With the same initial velocities, only stochastic methods can yield different results. Therefore, ```Langevin Dyanmics``` is highligh recommended.

   But one should note that under-damped and over-damped dynamics can yield quite different committing behavior.