####
YAML Files for Simulation
####

YANK now supports setting up entire simulations with easy to read and write `YAML files <http://www.yaml.org/about.html>`. No longer do you need to remember long command line arguments with many ``--`` flags, you can just write a ``yank.yaml`` file and call ``yank script --yaml=yank.yaml`` to carry out all your preparation and simulations. Below are the valid YAML options. We also have example YANK simulations which use YAML files to run.

.. todo:: This section is a work in progress and not all options are shown yet

All Pages
---------

.. toctree::
   :maxdepth: 1

   options <options>
   molecules <molecules>
   solvents <solvents>
   systems <systems>
   protocols <protocols>
   experiments <experiments>

Detailed Options List
---------------------
* :doc:`options <options>`

  * :ref:`General Options: <yaml_options_options>`

    * :ref:`verbose <yaml_options_verbose>`
    * :ref:`resume_setup <yaml_options_resume_setup>`
    * :ref:`resume_simulation <yaml_options_resume_simulation>`
    * :ref:`output_dir <yaml_options_output_dir>`
    * :ref:`setup_dir <yaml_options_setup_dir>`
    * :ref:`experiments_dir <yaml_options_experiments_dir>`

  * :ref:`System and Simulation Prep <yaml_options_sys_and_sim_prep>`

    * :ref:`randomize_ligand <yaml_options_randomize_ligand>`
    * :ref:`randomize_ligand_sigma_multiplier <yaml_options_randomize_ligand_sigma_multiplier>`
    * :ref:`randomize_ligand_close_cutoff <yaml_options_ligand_close_cutoff>`
    * :ref:`temperature <yaml_options_temperature>`
    * :ref:`pressure <yaml_options_pressure>`
    * :ref:`hydrogen_mass <yaml_options_hydrogen_mass>`
    * :ref:`constraints <yaml_options_constraints>`
    * :ref:`restraint_type <yaml_options_restraint_type>`

  * :ref:`Simulation Parameters: <yaml_options_simulation_parameters>`

    * :ref:`online_analysis <yaml_options_online_analysis>`
    * :ref:`online_analysis_min_iterations <yaml_options_online_analysis_min_iterations>`
    * :ref:`show_energies <yaml_options_show_energies>`
    * :ref:`show_mixing_statistics <yaml_options_show_mixing_statistics>`
    * :ref:`minimize <yaml_options_minimize>`
    * :ref:`minimize_max_iterations <yaml_options_minimize_max_iterations>`
    * :ref:`minimize_tolerance <yaml_options_minimize_tolerance>`
    * :ref:`number_of_equilibration_iterations <yaml_options_number_of_equilibration_iterations>`
    * :ref:`equilibration_timestep <yaml_options_equilibration_timestep>`
    * :ref:`number_of_iterations <yaml_options_number_of_iterations>`
    * :ref:`nsteps_per_iteration <yaml_options_nsteps_per_iteration>`
    * :ref:`timestep <yaml_options_timestep>`
    * :ref:`replica_mixing_scheme <yaml_options_replica_mixing_scheme>`
    * :ref:`collision_rate <yaml_options_collision_rate>`
    * :ref:`constraint_tolerance <yaml_options_constraint_tolerance>`
    * :ref:`mc_displacemnt_sigma <yaml_options_mc_displacemnt_sigma>`

  * :ref:`Alchemy Parameters <yaml_options_alchemy_parameters>`

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

* :doc:`molecules <molecules>`

  * :ref:`Specifying Molecule Names <yaml_molecules_specifiy_names>`

    * :ref:`filepath <yaml_molecules_filepath>` 
    * :ref:`smiles <yaml_molecules_smiles>`
    * :ref:`name <yaml_molecules_name>`
    * :ref:`strip_protons <yaml_molecules_strip_protons>`
    * :ref:`select <yaml_molecules_select>`
  
  * :ref:`Assigning Partial Charges <yaml_molecules_assign_charges>`

    * :ref:`antechamber <yaml_molecules_antechamber>`
    * :ref:`openeye <yaml_molecules_openeye>`

  * :ref:`Assigning Extra Information <yaml_molecules_extras>`
 
    * :ref:`leap <yaml_molecules_leap>`
    * :ref:`epik <yaml_molecules_epik>`

* :doc:`solvents <solvents>`

  * Items!

* :doc:`systems <systems>`

  * Items!

* :doc:`protocols <protocols>`

  * Items!

* :doc:`experiements <experiments>`


.. literalinclude:: ../../examples/yank-yaml-cookbook/all-options.yaml
   :language: yaml

|

Combinatorial Options in YAML
-----------------------------
YANK's YAML inputs also support running experiments combinatorially, instead of individually running them one at a time. YANK will automatically set up each combination of options you specify with the special ``!Combinatorial [itemA, itemB, ...]`` structure and run them back to back for you. Below is the cookbook for the combinatorial-experiments:

.. literalinclude:: ../../examples/yank-yaml-cookbook/combinatorial-experiment.yaml
   :language: yaml
