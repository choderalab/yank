.. _yaml_cookbook_basics_npt:

Basic NPT Simulation Recipe
***************************

This recipe shows a stock NPT setup in the :ref:`yaml-options-head`. We will run the simulations at
1 atmosphere and 300 Kelvin.

NPT simulations by their nature are run with periodic boundary conditions, so they
support explicit solvent, which we will also set. For this recipe, we choose the Particle Mesh Ewald (PME) nonbonded
method which works well with periodic systems and explicit solvents.

Here in the NPT, we also turn verbosity on to see if everything gets setup correctly. For your run, you may choose
to turn it off.

.. literalinclude:: raw_cooking_basics/rawnpt.yaml
   :language: yaml