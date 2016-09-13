.. _yaml-head:

YAML Files for Simulation
*************************

YANK now supports setting up entire simulations with easy to read and write `YAML files <http://www.yaml.org/about.html>`. No longer do you need to remember long command line arguments with many ``--`` flags, you can just write a ``yank.yaml`` file and call ``yank script --yaml=yank.yaml`` to carry out all your preparation and simulations. Below are the valid YAML options. We also have example YANK simulations which use YAML files to run.

.. todo:: This section is a work in progress and not all options are shown yet

.. literalinclude:: ../examples/yank-yaml-cookbook/all-options.yaml
   :language: yaml

|

Combinatorial Options in YAML
-----------------------------
YANK's YAML inputs also support running experiments combinatorially, instead of individually running them one at a time. YANK will automatically set up each combination of options you specify with the special ``!Combinatorial [itemA, itemB, ...]`` structure and run them back to back for you. Below is the cookbook for the combinatorial-experiments:

.. literalinclude:: ../examples/yank-yaml-cookbook/combinatorial-experiment.yaml
   :language: yaml
