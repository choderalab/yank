.. _yaml-options-head:

Options for YAML files
**********************

These are all the simulation, alchemy, and file I/O options controlled by the ``options`` header in the YAML files for
YANK. We have subdivided the categories below, but all settings on this page go under the ``options`` header in the YAML file:

* :ref:`General Options: <yaml_options_options>`
* :ref:`System and Simulation Prep <yaml_options_sys_and_sim_prep>`
* :ref:`Simulation Parameters: <yaml_options_simulation_parameters>`
* :ref:`Alchemy Parameters <yaml_options_alchemy_parameters>`

----

.. _yaml_options_options:

General Options:
================

.. _yaml_options_verbose:

verbose
-------
.. code-block:: yaml

  options:
    verbose: no

Turn on/off verbose output.

Valid Options: [no]/yes


.. _yaml_options_resume_setup:

resume_setup
------------
.. code-block:: yaml

   options:
     resume_setup: no

Choose to resume a setup procedure. YANK will raise an error when it detects that it will overwrite an existing file in
the directory specified by :ref:`setup_dir <yaml_options_setup_dir>`.

Valid Options: [no]/yes


.. _yaml_options_resume_simulation:

resume_simulation
-----------------
.. code-block:: yaml

   options:
     resume_simulation: no

Choose to resume simulations. YANK will raise an error when it detects that it will overwrite an existing file in the
directory specified by :ref:`experiments_dir <yaml_options_experiments_dir>`.

Valid Options: [no]/yes

.. _yaml_options_output_dir:

output_dir
----------
.. code-block:: yaml

   options:
     output_dir: output

The main output folder of YANK simulations. A folder will be created if none exists. Path is relative to the YAML script path

Valid Options (output): <Path String>

.. _yaml_options_setup_dir:

setup_dir
---------
.. code-block:: yaml

   options:
     setup_dir: setup

The folder where all generate simulation setup files are stored. A folder will be created if none exists.
Path is relative to the :ref:`output_dir <yaml_options_output_dir>` folder.

Valid Options (setup): <Path String>


.. _yaml_options_experiments_dir:

experiments_dir
---------------
.. code-block:: yaml

   options:
     experiments_dir: experiments

The folder where all generate simulation setup files are stored. A folder will be created if none exists. Path is
relative to to the :ref:`output_dir <yaml_options_output_dir>` folder.

Valid Options (experiments): <Path String>


.. _yaml_options_platform:

platform
--------
.. code-block:: yaml

   options:
     platform: fastest

The OpenMM platform used to run the calculations. The default value (``fastest``) automatically selects the fastest
available platform. Some platforms (especially ``CUDA`` and ``OpenCL``) may not be available on all systems.

Valid options: [fastest]/CUDA/OpenCL/CPU/Reference

.. _yaml_options_precision:

precision
---------
.. code-block:: yaml

   options:
     precision: auto

Floating point precision to use during the simulation. It can be set for OpenCL and CUDA platforms only. The default
value (``auto``) is equivalent to ``mixed`` when the device support this precision, and ``single`` otherwise.

Valid options: [auto]/double/mixed/single

|

.. _yaml_options_sys_and_sim_prep:

System and Simulation Prepartion:
=================================

.. _yaml_options_randomize_ligand:

randomize_ligand
----------------
.. code-block:: yaml

   options:
     randomize_ligand: no

Randomize the position of the ligand before starting the simulation.
Only works in Implicit Solvent. The ligand will be randomly rotated and displaced by
a vector with magnitude proportional  to
:ref:`randomize_ligand_sigma_multiplier <yaml_options_randomize_ligand_sigma_multiplier>`
with the constraint of being at a distance greater than
:ref:`randomize_ligand_close_cutoff <yaml_options_ligand_close_cutoff>` from the receptor.

Valid options: [no]/yes


.. _yaml_options_randomize_ligand_sigma_multiplier:

randomize_ligand_sigma_multiplier
---------------------------------
.. code-block:: yaml

   options:
     randomize_ligand_sigma_multiplier: 2.0

See :ref:`randomize_ligand <yaml_options_randomize_ligand>`.

Valid options (2.0): <float>


.. _yaml_options_ligand_close_cutoff:

randomize_ligand_close_cutoff
-----------------------------
.. code-block:: yaml

   options:
     randomize_ligand_close_cutoff: 1.5 * angstrom

See :ref:`randomize_ligand <yaml_options_randomize_ligand>`.

Valid options (1.5 * angstrom): <Quantity Length> [1]_


.. _yaml_options_temperature:

temperature
-----------
.. code-block:: yaml

   options:
     temperature: 298 * kelvin

Temperature of the system.

Valid options (298 * kelvin): <Quantity Temperature> [1]_


.. _yaml_options_pressure:

pressure
--------
.. code-block:: yaml

   options:
     pressure: 1.0 * atmosphere

Pressure of the system. If set to ``null``, the simulation samples as an NVT ensemble.

Valid options (1 * atmosphere): null / <Quantity Pressure> [1]_


.. _yaml_options_hydrogen_mass:

hydrogen_mass
-------------
.. code-block:: yaml

   options:
     hydrogen_mass: 1.0 * amu

Hydrogen mass for HMR simulations.

Valid options (1*amu): <Quantity Mass> [1]_


.. _yaml_options_constraints:

constraints
-----------
.. code-block:: yaml

   options:
     constraints: HBonds

Constrain bond lengths and angles. See OpenMM ``createSystem()`` documentation for more details.

Valid options: [Hbonds]/AllBonds/HAngles

|

.. _yaml_options_simulation_parameters:


Simulation Parameters
=====================

.. _yaml_options_online_analysis:

online_analysis
---------------
.. code-block:: yaml

   options:
     online_analysis: no

Analysis will occur at each iteration of the simulations if set. **WARNING:** This can be a slow process!

Valid options: [no]/yes


.. _yaml_options_online_analysis_min_iterations:

online_analysis_min_iterations
------------------------------
.. code-block:: yaml

   options:
     online_analysis_min_iterations: 20

The minimum number of iterations that must pass before :ref:`online analysis <yaml_options_online_analysis>` begins.

Valid options (20): <Integer>


.. _yaml_options_show_energies:

show_energies
-------------
.. code-block:: yaml

   options:
     show_energies: yes

If ``yes``, will print out the energies at each iteration.

Valid options: [yes]/no


.. _yaml_options_show_mixing_statistics:

show_mixing_statistics
----------------------
.. code-block:: yaml

   options:
     show_mixing_statistics: yes

If ``yes``, will print the Hamiltonian Replica Exchange swapping statistics at each iteration. This process adds a small
amount of overhead to each iteration.

Valid options: [yes]/no


.. _yaml_options_minimize:

minimize
--------
.. code-block:: yaml

   options:
     minimize: yes

Minimize the input configuration before starting simulation. Highly recommended if a pre-minimized structure is provided,
or if explicit solvent generation is left to YANK.

Valid Options: [yes]/no


.. _yaml_options_minimize_max_iterations:

minimize_max_iterations
-----------------------
.. code-block:: yaml

   options:
     minimize_max_iterations: 0

Set the maximum number of iterations the
:ref:`energy minimization process <yaml_options_minimize>` attempts to converge to :ref:`given tolerance energy <yaml_options_minimize_tolerance>`. 0 steps indicate unlimited.

Valid Options (0): <Integer>


.. _yaml_options_minimize_tolerance:

minimize_tolerance
------------------
.. code-block:: yaml

   options:
     minimize_tolerance: 1.0 * kilojoules_per_mole / nanometers

Set the tolerance of the :ref:`energy minimization process <yaml_options_minimize>`. System is considered minimized when
the energy does not change by the given tolerance in subsequent iterations.

Valid Options (1.0 * kilojoules_per_mole / nanometers): <Quantity (Molar Energy)/(Length)> [1]_


.. _yaml_options_number_of_equilibration_iterations:

number_of_equilibration_iterations
----------------------------------
.. code-block:: yaml

   options:
     number_of_equilibration_iterations: 1

Number of iterations used for equilibration before production run. Iterations written to file are post-equilibration.

Valid Options (1): <Integer>


.. _yaml_options_equilibration_timestep:

equilibration_timestep
----------------------
.. code-block:: yaml

   options:
     equilibration_timestep: 1.0 * femtosecond

Timestep of the *equilibration* timestep (not production).

Valid Options (1.0 * femtosecond): <Quantity Time> [1]_


.. _yaml_options_number_of_iterations:

number_of_iterations
--------------------
.. code-block:: yaml

   options:
     number_of_iterations: 1

Number of iterations for production simulation. Note: If :ref:`resume_simulation <yaml_options_resume_simulation>` is
set, this option can be used to extend previous simulations past their original number of iterations.

Valid Options (1): <Integer>


.. _yaml_options_nsteps_per_iteration:

nsteps_per_iteration
--------------------
.. code-block:: yaml

   options:
     nsteps_per_iteration: 500

Number of timesteps between each iteration. We highly recommend using a number greater than 1 to improve decorrelation
between iterations. Hamiltonian Replica Exchange swaps are attempted after each iteration.

Valid Options (500): <Integer>


.. _yaml_options_timestep:

timestep
--------
.. code-block:: yaml

   options:
     timestep: 2.0 * femtosecond

Timestep of Langevin Dynamics production runs.

Valid Options (2.0 * femtosecond): <Quantity Time> [1]_


.. _yaml_options_replica_mixing_scheme:

replica_mixing_scheme
---------------------
.. code-block:: yaml

   options:
     replica_mixing_scheme: swap-all

Specifies how the Hamiltonian Replica Exchange attempts swaps between replicas.
``swap-all`` will attempt to exchange every state with every other state. ``swap-neighbors``  will attempt only
exchanges between adjacent states.

Valid Options: [swap-all]/swap-neighbors


.. _yaml_options_collision_rate:

collision_rate
--------------
.. code-block:: yaml

   options:
     collision_rate: 5.0 / picosecond

The collision rate used for Langevin dynamics. Default quantity of 5.0 / picosecond works well for explicit solvent.
Implicit solvent will require a different collision rate, e.g. 91 / picosecond works well for TIP3P water.

Collision rates (or friction coefficients) appear in the Langevin dynamics equation as either inverse time, or one over
some time constant, :math:`1/\tau`.  When comparing collision rates, double check if the collision rate is in units of
inverse time, or just time. For example: a collision rate of 5.0/ps -> :math:`\tau = 0.2 \, ps`.

Valid Options (5.0 / picosecond): <Quantity Inverse Time> [1]_


.. _yaml_options_constraint_tolerance:

constraint_tolerance
--------------------
.. code-block:: yaml

   options:
     constraint_tolerance: 1.0e-6

Relative tolerance on the :ref:`constraints <yaml_options_constraints>` of the system.

Valid Options (1.0e-6): <Scientific Notation Float>


.. _yaml_options_mc_displacement_sigma:

mc_displacement_sigma
---------------------
.. code-block:: yaml

   options:
     mc_displacement_sigma: 10.0 * angstroms

YANK will augment Langevin dynamics with MC moves rotating and displacing the ligand. This parameter controls the size of the displacement

Valid Options (10 * angstroms): <Quantity Length> [1]_

|


.. _yaml_options_alchemy_parameters:

Alchemy Parameters
==================

.. _yaml_options_annihilate_electrostatics:

annihilate_electrostatics
-------------------------
.. code-block:: yaml

   options:
     annihilate_electrostatics: yes

Annihilate electrostatics rather than decouple them. This means that ligand-ligand (alchemical-alchemical) nonbonded
electrostatics will be turned off as well as ligand-nonligand nonbonded electrostatics.

Valid Options: [yes]/no


.. _yaml_options_annihilate_sterics:

annihilate_sterics
------------------
.. code-block:: yaml

   options:
     annihilate_sterics: no

Annihilate sterics (Lennad-Jones or Halgren potential) rather than decouple them. This means that ligand-ligand
(alchemical-alchemical) nonbonded sterics will be turned off as well as ligand-nonligand nonbonded sterics.
**WARNING:** Do *not* set this option if ``annihilate_electrostatics`` is "no".

Valid Options: [no]/yes


.. _yaml_options_alchemical_sterics:

Steric Alchemical Options
-------------------------
.. code-block:: yaml

   options:
     softcore_alpha: 0.5
     softcore_a: 1
     softcore_b: 1
     softcore_c: 6

The options that control the soft core energy function for decoupling/annihilating steric interactions. Setting
``softcore_alpha = 0`` with ``softcore_a = 1`` gives linear scaling of the Lennard-Jones energy function.

Valid Options for ``softcore_alpha`` (0.5): <Float>

Valid Options for ``softcore_[a,b,c]`` (1,1,6): <Integer prefered, Float accepted>


.. _yaml_options_alchemical_electrostatics:

Electrostatic Alchemical Options
--------------------------------
.. code-block:: yaml

   options:
     softcore_beta: 1.0
     softcore_d: 1
     softcore_e: 1
     softcore_f: 2

The options that control the soft core energy functnon for decoupling/annihilating electrostatic interactions.
Setting ``softcore_beta = 0`` with ``softcore_d = 1`` gives linear scaling of Coulomb's law.

Valid Options for ``softcore_beta`` (1.0): <Float>

Valid Options for ``softcore_[d,e,f]`` (1,1,2): <Integer prefered, Float accepted>


.. [1] Quantity strings are of the format: ``<float> * <unit>`` where ``<unit>`` is any valid unit specified in the "Valid Options" for an option. e.g. "<Quantity Length>" indicates any measure of length may be used for <unit> such as nanometer or angstrom.
   Compound units are also parsed such as ``kilogram / meter**3`` for density.
   Only full unit names as they appear in the simtk.unit package (part of OpenMM) are allowed; so "nm" and "A" will be rejected.
