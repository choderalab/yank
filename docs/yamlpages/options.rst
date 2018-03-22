.. _yaml-options-head:

Options for YAML files
**********************

These are all the simulation, alchemy, and file I/O options controlled by the ``options`` header in the YAML files for
YANK. We have subdivided the categories below, but all settings on this page go under the ``options`` header in the YAML file:

* :ref:`General Options <yaml_options_options>`
* :ref:`System and Simulation Prep <yaml_options_sys_and_sim_prep>`
* :ref:`Simulation Parameters <yaml_options_simulation_parameters>`
* :ref:`Alchemy Parameters <yaml_options_alchemy_parameters>`

Besides the options listed in :ref:`General Options <yaml_options_options>` that can be specified exclusively in the
``options`` section of the YAML script, everything else can go either in ``options`` as a general setting, or in
``experiments.options``. In the latter case, an option can be expanded combinatorially with the ``!Combinatorial`` tag.

----

.. _yaml_options_options:

General Options:
================



.. _yaml_options_verbose:

.. rst-class:: html-toggle

``verbose``
-----------

.. code-block:: yaml

  options:
    verbose: no

Turn on/off verbose output.

Valid Options: [no]/yes




.. _yaml_options_resume_setup:

.. rst-class:: html-toggle

``resume_setup``
----------------

.. code-block:: yaml

   options:
     resume_setup: no

Choose to resume a setup procedure. YANK will raise an error when it detects that it will overwrite an existing file in
the directory specified by :ref:`setup_dir <yaml_options_setup_dir>`.

Valid Options: [no]/yes




.. _yaml_options_resume_simulation:

.. rst-class:: html-toggle

``resume_simulation``
---------------------
.. code-block:: yaml

   options:
     resume_simulation: no

Choose to resume simulations. YANK will raise an error when it detects that it will overwrite an existing file in the
directory specified by :ref:`experiments_dir <yaml_options_experiments_dir>`.

Valid Options: [no]/yes





.. _yaml_options_output_dir:

.. rst-class:: html-toggle

``output_dir``
--------------
.. code-block:: yaml

   options:
     output_dir: output

The main output folder of YANK simulations. A folder will be created if none exists. Path is relative to the YAML script path

Valid Options (output): <Path String>




.. _yaml_options_setup_dir:

.. rst-class:: html-toggle

``setup_dir``
-------------
.. code-block:: yaml

   options:
     setup_dir: setup

The folder where all generate simulation setup files are stored. A folder will be created if none exists.
Path is relative to the :ref:`output_dir <yaml_options_output_dir>` folder.

Valid Options (setup): <Path String>




.. _yaml_options_experiments_dir:

.. rst-class:: html-toggle

``experiments_dir``
-------------------
.. code-block:: yaml

   options:
     experiments_dir: experiments

The folder where all generate simulation setup files are stored. A folder will be created if none exists. Path is
relative to to the :ref:`output_dir <yaml_options_output_dir>` folder.

Valid Options (experiments): <Path String>




.. _yaml_options_platform:

.. rst-class:: html-toggle

``platform``
------------
.. code-block:: yaml

   options:
     platform: fastest

The OpenMM platform used to run the calculations. The default value (``fastest``) automatically selects the fastest
available platform. Some platforms (especially ``CUDA`` and ``OpenCL``) may not be available on all systems.

Valid options: [fastest]/CUDA/OpenCL/CPU/Reference



.. _yaml_options_precision:

.. rst-class:: html-toggle

``precision``
-------------
.. code-block:: yaml

   options:
     precision: auto

Floating point precision to use during the simulation. It can be set for OpenCL and CUDA platforms only. The default
value (``auto``) is equivalent to ``mixed`` when the device support this precision, and ``single`` otherwise.

Valid options: [auto]/double/mixed/single



.. _yaml_options_max_n_contexts:

.. rst-class:: html-toggle

``max_n_contexts``
------------------
.. code-block:: yaml

   options:
     max_n_contexts: 3

The maximum number of GPU contexts that can be in memory during the simulation. In general, YANK does not need more
than 3 contexts.

Valid options (3): <Integer>



.. _yaml_options_switch_experiment_interval:

.. rst-class:: html-toggle

``switch_experiment_interval``
------------------------------
.. code-block:: yaml

   options:
     switch_experiments_interval: 0

When running multiple experiments using the ``!Combinatorial`` tag, this allows to switch between experiments every
``switch_experiments_interval`` iterations, and gather data about multiple molecules/conditions before
completing the specified ``number_of_iterations``. If 0, YANK will complete the combinatorial calculations
sequentially.

Valid options (0): <Integer>

.. _yaml_options_processes_per_experiment:

.. rst-class:: html-toggle

``processes_per_experiment``
----------------------------
.. code-block:: yaml

   options:
     processes_per_experiment: auto

When running YANK on multiple processes with MPI, you can run several experiments in parallel by using this option to
allocate a given number of processes to each experiment. This option is ignored if YANK is not run with MPI. If ``null``,
the experiments are performed one after the other on all the available MPI processes. When ``auto`` is selected, YANK
tries to run as many experiment as possible in parallel on independent MPI processes. Currently, only
``processes_per_experiment = 1`` is supported for the SAMS sampler.

Valid options (auto): auto / null / <Integer>



.. _yaml_options_sys_and_sim_prep:


System and Simulation Preparation:
==================================

.. _yaml_options_randomize_ligand:

.. rst-class:: html-toggle

``randomize_ligand``
--------------------
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

.. rst-class:: html-toggle

``randomize_ligand_sigma_multiplier``
-------------------------------------
.. code-block:: yaml

   options:
     randomize_ligand_sigma_multiplier: 2.0

See :ref:`randomize_ligand <yaml_options_randomize_ligand>`.

Valid options (2.0): <float>




.. _yaml_options_ligand_close_cutoff:

.. rst-class:: html-toggle

``randomize_ligand_close_cutoff``
---------------------------------
.. code-block:: yaml

   options:
     randomize_ligand_close_cutoff: 1.5 * angstrom

See :ref:`randomize_ligand <yaml_options_randomize_ligand>`.

Valid options (1.5 * angstrom): <Quantity Length> [1]_




.. _yaml_options_temperature:

.. rst-class:: html-toggle

``temperature``
---------------
.. code-block:: yaml

   options:
     temperature: 298 * kelvin

Temperature of the system.

Valid options (298 * kelvin): <Quantity Temperature> [1]_




.. _yaml_options_pressure:

.. rst-class:: html-toggle

``pressure``
------------
.. code-block:: yaml

   options:
     pressure: 1.0 * atmosphere

Pressure of the system. If set to ``null``, the simulation samples as an NVT ensemble.

Valid options (1 * atmosphere): null / <Quantity Pressure> [1]_



.. _yaml_options_hydrogen_mass:

.. rst-class:: html-toggle

``hydrogen_mass``
-----------------
.. code-block:: yaml

   options:
     hydrogen_mass: 1.0 * amu

Hydrogen mass for HMR simulations.

Valid options (1*amu): <Quantity Mass> [1]_




.. _yaml_options_constraints:

.. rst-class:: html-toggle

``constraints``
---------------
.. code-block:: yaml

   options:
     constraints: HBonds

Constrain bond lengths and angles. See OpenMM ``createSystem()`` documentation for more details.

Valid options: [Hbonds]/AllBonds/HAngles



.. _yaml_options_anisotropic_dispersion_cutoff:

.. rst-class:: html-toggle

``anisotropic_dispersion_cutoff``
---------------------------------
.. code-block:: yaml

   options:
     anisotropic_dispersion_cutoff: auto

Tell YANK to compute anisotropic dispersion corrections for long-range interactions. YANK accounts for these effects
by creating two additional thermodynamic states at either end of the :ref:`thermodynamic cycle <yank_cycle>` with
larger long-range cutoffs to remove errors introduced from treating long-range interactions as a homogeneous, equal
density medium. We estimate the free energy relative to these expanded cutoff states. No simulation is actually carried
out at these states but energies from simulations are evaluated at them.

This option only applies if you have specified a
:ref:`system with periodic boundary conditions <yaml_solvents_nonbonded_method>`.

We put this option in the general options category instead of the :doc:`solvents <solvents>` section since these
additional states are unique to YANK's setup.

The size of the expanded cutoff distance can be set in a few ways through this option. If
``auto`` the cutoff will be set to ``0.99*min_box_size/2`` if no barostat is in use or ``0.8*min_box_size/2`` if
one is in use (to account for box size fluctuations), with ``min_box_size`` denoting the norm of the smallest OpenMM
box vector defining the initial triclinic cell volume.

Valid options: [auto]/``null``/<Quantity Length> [1]_

|



.. _yaml_options_simulation_parameters:


Simulation Parameters
=====================


.. _yaml_options_switch_phase_interval:

.. rst-class:: html-toggle

``switch_phase_interval``
-------------------------
.. code-block:: yaml

   options:
     switch_phase_interval: 0

This allows to switch the simulation between the two phases of the calculation every ``switch_phase_interval`` iterations.
If 0, YANK will exhaust the ``number_of_iterations`` iterations of the first phase before switching to the second one.

Valid options (0): <Integer>




.. _yaml_options_minimize:

.. rst-class:: html-toggle

``minimize``
------------
.. code-block:: yaml

   options:
     minimize: yes

Minimize the input configuration before starting simulation. Highly recommended if a pre-minimized structure is provided,
or if explicit solvent generation is left to YANK.

Valid Options: [yes]/no




.. _yaml_options_minimize_max_iterations:

.. rst-class:: html-toggle

``minimize_max_iterations``
---------------------------
.. code-block:: yaml

   options:
     minimize_max_iterations: 0

Set the maximum number of iterations the
:ref:`energy minimization process <yaml_options_minimize>` attempts to converge to :ref:`given tolerance energy <yaml_options_minimize_tolerance>`. 0 steps indicate unlimited.

Valid Options (0): <Integer>




.. _yaml_options_minimize_tolerance:

.. rst-class:: html-toggle

``minimize_tolerance``
----------------------
.. code-block:: yaml

   options:
     minimize_tolerance: 1.0 * kilojoules_per_mole / nanometers

Set the tolerance of the :ref:`energy minimization process <yaml_options_minimize>`. System is considered minimized when
the energy does not change by the given tolerance in subsequent iterations.

Valid Options (1.0 * kilojoules_per_mole / nanometers): <Quantity (Molar Energy)/(Length)> [1]_




.. _yaml_options_number_of_equilibration_iterations:

.. rst-class:: html-toggle

``number_of_equilibration_iterations``
--------------------------------------
.. code-block:: yaml

   options:
     number_of_equilibration_iterations: 1

Number of iterations used for equilibration before production run. Iterations written to file are post-equilibration.

Valid Options (1): <Integer>




.. _yaml_options_equilibration_timestep:

.. rst-class:: html-toggle

``equilibration_timestep``
--------------------------
.. code-block:: yaml

   options:
     equilibration_timestep: 1.0 * femtosecond

Timestep of the *equilibration* timestep (not production).

Valid Options (1.0 * femtosecond): <Quantity Time> [1]_




.. _yaml_options_default_number_of_iterations:

.. rst-class:: html-toggle

``default_number_of_iterations``
--------------------------------
.. code-block:: yaml

   options:
     default_number_of_iterations: 1

Default number of iterations for the :ref:`samplers that do not explicitly specify <yaml_samplers_example>`
the option ``number_of_iterations``.
Note: If :ref:`resume_simulation <yaml_options_resume_simulation>` is set, this option can be used to extend previous
simulations past their original number of iterations.

Specifying ``0`` will run through the setup, create all the simulation files, store all options, and minimize the
initial configurations (if specified), but will not run any production simulations.

Set this to ``.inf`` (note the prepended dot character) to run an unlimited number of iterations. The simulation will
not stop unless some other criteria is stops it. We **strongly** recommend specifying either
:ref:`online free energy analysis <yaml_samplers_online_analysis_parameters>` and/or
:ref:`a phase switching interval <yaml_options_switch_phase_interval>` to ensure there is at least some stop criteria,
and all phases yield some samples.

Valid Options (1): <Integer> or ``.inf``


..
   .. _yaml_options_extend_simulation:

   extend_simulation
   --------------------
   .. code-block:: yaml

       options:
         extend_simulation: False

   Special modification of :ref:`yaml_options_number_of_iterations` which allows **extending** a simulation by
   :ref:`yaml_options_number_of_iterations` instead of running for a maximum. If set to ``True``,
   the simulation will run the additional specified number of iterations, even if a simulation already has
   run for a length of time. For fresh simulations, the resulting simulation is identical to not setting this flag.

   This is helpful for running consecutive batches of simulations for time lengths that are unknown.

   *Recommended*: Also set :ref:`resume_setup <yaml_options_resume_setup>` and
   :ref:`resume_simulation <yaml_options_resume_simulation>` to allow resuming simulations.

   *Example*: You have a simulation that ran for 500 iterations, you wish to add an additional 1000 iterations. You would
   set ``number_of_iterations: 1000`` and ``extend_simulation: True`` in your YAML file and rerun. The simulation would
   then resume at iteration 500, then continue to iteration 1500. The same behavior would be achieved if you set
   ``number_of_iterations: 1500``, but the ``extend_simulation`` has the advantage that it can be run multiple times to
   keep extending the simulation without modifying the YAML file.

   **WARNING**: Extending simulations affects ALL simulations for :doc:`Combinatorial <combinatorial>`. You cannot extend
   a subset of simulations from a combinatorial setup; all simulations will be extended if this option is set.

   **OPTIONAL** and **MODIFIES** :ref:`yaml_options_number_of_iterations`

   Valid Options: True/[False]




.. _yaml_options_default_nsteps_per_iteration:

.. rst-class:: html-toggle

``default_nsteps_per_iteration``
--------------------------------
.. code-block:: yaml

   options:
     default_nsteps_per_iteration: 500

Number of timesteps between each iteration with the default MCMC move. We highly recommend using a number greater than 1
to improve decorrelation between iterations. Hamiltonian Replica Exchange swaps are attempted after each iteration. This
option is ignored if a custom MCMC move is used for the experiment.

Valid Options (500): <Integer>




.. _yaml_options_default_timestep:

.. rst-class:: html-toggle

``default_timestep``
--------------------
.. code-block:: yaml

   options:
     default_timestep: 2.0 * femtosecond

The timestep of the Langevin Dynamics with the default MCMC move. This option is ignored when a custom MCMC move is used.

Valid Options (2.0 * femtosecond): <Quantity Time> [1]_




.. _yaml_options_checkpoint_interval:

.. rst-class:: html-toggle

``checkpoint_interval``
-----------------------
.. code-block:: yaml

   options:
     checkpoint_interval: 10

Specify how frequently checkpoint information should be saved to file relative to iterations. YANK simulations can be
resumed only from checkpoints, so if something crashes, up to ``checkpoint_interval`` worth of iterations will be lost
and YANK will resume from the most recent checkpoint.

This option helps control write-to-disk time and file sizes. The fewer times a checkpoint is written, the less of both
you will get. If you want to write a checkpoint every iteration, set this to ``1``.

Checkpoint information includes things like full coordinates and box vectors, as well as more static information such
as metadata, simulation options, and serialized thermodynamic states.

Valid Options (10): <Integer ``>= 1``>




.. _yaml_options_store_solute_trajectory:

.. rst-class:: html-toggle

``store_solute_trajectory``
---------------------------
.. code-block:: yaml

   options:
     store_solute_trajectory: yes

Specify if you want an additional trajectory of just the solute atoms stored every iteration, regardless of the
``checkpoint_interval``.

If specified, this will write the data to the analysis file in addition to the normal information stored in the
checkpoint file. As such, you should be careful when considering space and the ``checkpoint_interval`` setting. For
instance, an implicit solvent simulation with ``checkpoint_interval: 1`` will result in a redundant copy of the
complete trajectory.

Valid Options: [yes]/no



.. _yaml_options_constraint_tolerance:

.. rst-class:: html-toggle

``constraint_tolerance``
------------------------
.. code-block:: yaml

   options:
     constraint_tolerance: 1.0e-6

Relative tolerance on the :ref:`constraints <yaml_options_constraints>` of the system.

Valid Options (1.0e-6): <Scientific Notation Float>


|


.. _yaml_options_alchemy_parameters:

Alchemy Parameters
==================

.. _yaml_options_annihilate_electrostatics:

.. rst-class:: html-toggle

``annihilate_electrostatics``
-----------------------------
.. code-block:: yaml

   options:
     annihilate_electrostatics: yes

Annihilate electrostatics rather than decouple them. This means that ligand-ligand (alchemical-alchemical) nonbonded
electrostatics will be turned off as well as ligand-nonligand nonbonded electrostatics.

Valid Options: [yes]/no




.. _yaml_options_annihilate_sterics:

.. rst-class:: html-toggle

``annihilate_sterics``
----------------------
.. code-block:: yaml

   options:
     annihilate_sterics: no

Annihilate sterics (Lennad-Jones or Halgren potential) rather than decouple them. This means that ligand-ligand
(alchemical-alchemical) nonbonded sterics will be turned off as well as ligand-nonligand nonbonded sterics.
**WARNING:** Do *not* set this option if ``annihilate_electrostatics`` is "no".

Valid Options: [no]/yes




.. _yaml_options_alchemical_sterics:

.. rst-class:: html-toggle

``Steric Alchemical Options``
-----------------------------
.. code-block:: yaml

   options:
     softcore_alpha: 0.5
     softcore_a: 1
     softcore_b: 1
     softcore_c: 6

The options that control the soft core energy function for decoupling/annihilating steric interactions. Setting
``softcore_alpha = 0`` with ``softcore_a = 1`` gives linear scaling of the Lennard-Jones energy function.

Valid Options for ``softcore_alpha`` (0.5): <Float>

Valid Options for ``softcore_[a,b,c]`` (1,1,6): <Integer preferred, Float accepted>




.. _yaml_options_alchemical_electrostatics:

.. rst-class:: html-toggle

``Electrostatic Alchemical Options``
------------------------------------
.. code-block:: yaml

   options:
     softcore_beta: 0.0
     softcore_d: 1
     softcore_e: 1
     softcore_f: 2

The options that control the soft core energy functnon for decoupling/annihilating electrostatic interactions.
Setting ``softcore_beta = 0`` with ``softcore_d = 1`` gives linear scaling of Coulomb's law.

Valid Options for ``softcore_beta`` (0.0): <Float>

Valid Options for ``softcore_[d,e,f]`` (1,1,2): <Integer preferred, Float accepted>




.. _yaml_options_alchemical_pme_treatment:

.. rst-class:: html-toggle

``alchemical_pme_treatment``
----------------------------
.. code-block:: yaml

   options:
     alchemical_pme_treatment: direct-space

When using PME, by default YANK runs the simulation modeling exclusively the direct space of PME. The reciprocal space
is taken into account by reweighting the end states (the same reweighting performed for the anisotropic long-range
dispersion correction). This makes it very fast to compute the energy matrix at each iteration. However, charged ligands
may have a poor overlap between the direct-space-only and the full PME space. In this case, convergence rates can be
very long, and it is recommended to use exact treatment of PME electrostatics.

Valid Options: [direct-space]/exact/coulomb




.. _yaml_options_disable_alchemical_dispersion_correction:

.. rst-class:: html-toggle

``disable_alchemical_dispersion_correction``
--------------------------------------------
.. code-block:: yaml

   options:
     disable_alchemical_dispersion_correction: yes

By default, the contribution of the alchemical atoms to the analytical long-range dispersion correction is not included
to speed up the computation of the energy matrix. This contribution is included in the end states anisotropic long-range
dispersion correction.

Valid Options: [yes]/no




.. _yaml_options_split_alchemical_forces:

.. rst-class:: html-toggle

``split_alchemical_forces``
---------------------------
.. code-block:: yaml

   options:
     split_alchemical_forces: yes

By default, the alchemical forces are split into their own OpenMM force groups to speed up the computation of the energy
matrix. If your input system is particularly loaded with forces, and they occupy many force group, you may incur into
errors during the creation of the alchemical system as OpenMM supports a maximum of 32 force groups. In this case, it is
recommended to merge some of your forces into a single group. If this is not possible, set this to ``no`` to proceed
without this optimization.

Valid Options: [yes]/no

.. [1] Quantity strings are of the format: ``<float> * <unit>`` where ``<unit>`` is any valid unit specified in the "Valid Options" for an option. e.g. "<Quantity Length>" indicates any measure of length may be used for <unit> such as nanometer or angstrom.
   Compound units are also parsed such as ``kilogram / meter**3`` for density.
   Only full unit names as they appear in the simtk.unit package (part of OpenMM) are allowed; so "nm" and "A" will be rejected.

|
