.. _yaml_cookbook_head:

Cookbook for YAML Options
*************************

Having all the options laid out in front of you is good for advanced users, but sometimes practical examples are much more helpful.
Here you can find the YANK YAML Cookbook: A handy guide to making your
A series of examples that may help you understanding how to put together the options and settings found on the
:ref:`main YAML pages <yaml_head>`.

There are a few things to keep in mind with these recipes:

* Arbitrary named keys (such as the YANK name for the ``molecules``) will be given example concrete values
* Paths are all examples and will not reflect your path structure, replaces as needed unless explicitly stated the
  entry is to be taken literally.

----

.. toctree::
   :maxdepth: 2

   cookingbasics/index
   cookingbinding
   cookinghydration




* :doc:`options <options>`


    * :ref:`setup_dir <yaml_options_setup_dir>`
    * :ref:`experiments_dir <yaml_options_experiments_dir>`
    * :ref:`platform <yaml_options_platform>`
    * :ref:`precision <yaml_options_precision>`
    * :ref:`max_n_contexts <yaml_options_max_n_contexts>`
    * :ref:`switch_experiment_interval <yaml_options_switch_experiment_interval>`
    * :ref:`processes_per_experiment <yaml_options_processes_per_experiment>`


    * :ref:`randomize_ligand <yaml_options_randomize_ligand>`
    * :ref:`randomize_ligand_sigma_multiplier <yaml_options_randomize_ligand_sigma_multiplier>`
    * :ref:`randomize_ligand_close_cutoff <yaml_options_ligand_close_cutoff>`
    * :ref:`hydrogen_mass <yaml_options_hydrogen_mass>`
    * :ref:`constraints <yaml_options_constraints>`


    * :ref:`switch_phase_interval <yaml_options_switch_phase_interval>`
    * :ref:`minimize <yaml_options_minimize>`
    * :ref:`minimize_max_iterations <yaml_options_minimize_max_iterations>`
    * :ref:`minimize_tolerance <yaml_options_minimize_tolerance>`
    * :ref:`number_of_equilibration_iterations <yaml_options_number_of_equilibration_iterations>`
    * :ref:`equilibration_timestep <yaml_options_equilibration_timestep>`
    * :ref:`default_number_of_iterations <yaml_options_default_number_of_iterations>`
    * :ref:`default_nsteps_per_iteration <yaml_options_default_nsteps_per_iteration>`
    * :ref:`default_timestep <yaml_options_default_timestep>`
    * :ref:`checkpoint_interval <yaml_options_checkpoint_interval>`
    * :ref:`store_solute_trajectory <yaml_options_store_solute_trajectory>`
    * :ref:`constraint_tolerance <yaml_options_constraint_tolerance>`
    * :ref:`yaml_options_anisotropic_dispersion_cutoff`


    * :ref:`annihilate_electrostatics <yaml_options_annihilate_electrostatics>`
    * :ref:`annihilate_sterics <yaml_options_annihilate_sterics>`
    * :ref:`softcore_alpha <yaml_options_alchemical_sterics>`
    * :ref:`softcore_beta <yaml_options_alchemical_electrostatics>`
    * :ref:`softcore_a <yaml_options_alchemical_sterics>`
    * :ref:`softcore_b <yaml_options_alchemical_sterics>`
    * :ref:`softcore_c <yaml_options_alchemical_sterics>`
    * :ref:`softcore_d <yaml_options_alchemical_electrostatics>`
    * :ref:`softcore_e <yaml_options_alchemical_electrostatics>`
    * :ref:`softcore_f <yaml_options_alchemical_electrostatics>`
    * :ref:`alchemical_pme_treatment <yaml_options_alchemical_pme_treatment>`
    * :ref:`disable_alchemical_dispersion_correction <yaml_options_disable_alchemical_dispersion_correction>`
    * :ref:`split_alchemical_forces <yaml_options_split_alchemical_forces>`