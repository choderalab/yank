.. _yaml-head:

YAML Files for Simulation
*************************

YANK now supports setting up entire simulations with easy to read and write `YAML files <http://www.yaml.org/about.html>`. No longer do you need to remember long command line arguments with many ``--`` flags, you can just write a ``yank.yaml`` file and call ``yank script --yaml=yank.yaml`` to carry out all your preparation and simulations. Below are the valid YAML options. We also have example YANK simulations which use YAML files to run.

.. todo:: This section is a work in progress and not all options are shown yet

* :doc:`options <yamlpages/options.rst>`

  * :ref:`**General Options:** yaml_options_options`
  * :ref:`verbose yaml_options_verbose`
  * :ref:`resume_setup yaml_options_resume_setup`
  * :ref:`resume_simulation yaml_options_resume_simulation`
  * output_dir
  * setup_dir
  * experiments_dir
  * **System and Simulation Prep:**
  * randomize_ligand
  * randomize_ligand_sigma_multiplier
  * randomize_ligand_close_cutoff
  * temperature
  * pressure
  * hydrogen_mass
  * constraints
  * restraint_type
  * **Simulation Parameters**
  * online_analysis
  * online_analysis_min_iterations
  * show_energies
  * show_mixing_statistics
  * minimize
  * minimize_max_iterations
  * number_of_equilibration_iterations
  * equilibration_timestep
  * number_of_iterations
  * nsteps_per_iteration
  * timestep
  * replica_mixing_scheme
  * collision_rate
  * constraint_tolerance
  * mc_displacemnt_sigma
  * **Alchemy Parameters**
  * annihilate_electrostatics
  * annihilate_sterics
  * softcore_alpha
  * softcore_beta
  * softcore_a
  * softcore_b
  * softcore_c
  * softcore_d
  * softcore_e
  * softcore_f


.. literalinclude:: ../examples/yank-yaml-cookbook/all-options.yaml
   :language: yaml

|

Combinatorial Options in YAML
-----------------------------
YANK's YAML inputs also support running experiments combinatorially, instead of individually running them one at a time. YANK will automatically set up each combination of options you specify with the special ``!Combinatorial [itemA, itemB, ...]`` structure and run them back to back for you. Below is the cookbook for the combinatorial-experiments:

.. literalinclude:: ../examples/yank-yaml-cookbook/combinatorial-experiment.yaml
   :language: yaml
