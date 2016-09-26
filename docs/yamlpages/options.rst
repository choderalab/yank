.. _yaml-options-head:

Options for YAML files
**********************

These are all the simulation, alchemy, and file I/O options controlled by the `options` header in the YAML files for YANK.

.. _yaml-options-options:

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

Choose to resume a setup procedure. YANK will raise an error when it detects that it will overwrite an existing file in the directory specified by :ref:`setup_dir yaml_options_setup_dir`.

Valid Options: [no]/yes


.. _yaml_options_resume_simulation:

resume_simulation
-----------------
.. code-block:: yaml

   options:
     resume_simulation: no

Choose to resume simulations. YANK will raise an error when it detects that it will overwrite an existing file in the directory specified by :ref:`experiments_dir yaml_options_experiments_dir`.

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

The folder where all generate simulation setup files are stored. A folder will be created if none exists. Path is relative to to the :ref:`output_dir <yaml_options_output_dir>` folder.

Valid Options (setup): <Path String>


.. _yaml_options_experiments_dir:

experiments_dir
---------------
.. code-block:: yaml

   options:
     experiments_dir: experiments

The folder where all generate simulation setup files are stored. A folder will be created if none exists. Path is relative to to the :ref:`output_dir <yaml_options_output_dir>` folder.

Valid Options (experiments): <Path String>

|

.. _yaml_options_sys_and_sim_prep:

System and Simulation Prepartion: 
=================================

.. _yaml_options_randomize_ligand:

randomize_liand
---------------
.. code-block:: yaml

   options:
     randomize_liand: no

Randomize the position of the ligand before starting the simulation. Only works in Implicit Solvent. The ligand will be randomly rotated and displaced by a vector with magnitude proportioal to :ref:`randomize_ligand_sigma_multiplier <yaml_options_randomize_ligand_sigma_multiplier>` with the constraint of being at a distance greater than :ref:`randomize_ligand_close_cutoff <yaml_options_ligand_close_cutoff>` from the receptor.

Valid options: [no]/yes


.. _yaml_options_randomize_ligand_sigma_multiplier:

randomize_ligand_sigma_multiplier
---------------------------------
.. code-block:: yaml

   options:
     randomize_ligand_sigma_multiplier: 2.0

See :ref:`randomize_ligand <yaml_options_randomize_ligand>`. 

Valid options: <float>


.. _yaml_options_ligand_close_cutoff:

randomize_ligand_close_cutoff
-----------------------------
.. code-block:: yaml
   
   options:
     randomize_ligand_close_cutoff: 1.5 * angstrom

See :ref:`randomize_ligand <yaml_options_randomize_ligand>`.

Valid options: <Quantity Length> [1]_


.. _yaml_options_temperature:

temperature
-----------
.. code-block:: yaml

   options:
     temperature: 298 * kelvin

Temperature of the system.

Valid options: <Quantity Temperature> [1]_


.. _yaml_options_pressure:

presuure
--------
.. code-block:: yaml

   options:
     pressure: 1.0 * atmosphere

Pressure of the system. If set to `null`, the simulation samples as an NVT ensemble.

Valid options: null / <Quantity Pressure> [1]_


.. _yaml_options_hydrogen_mass:

hydrogen_mass
-------------
.. code-block:: yaml

   options:
     hydrogen_mass: 1.0 * amu

Hydrogen mass for HMR simulations.

Valid options: <Quantity Mass> [1]_


.. _yaml_options_constraints:

constraints
-----------
.. code-block:: yaml
   
   options:
     constraints: HBonds

Constrain bond lengths and angles. See OpenMM `createSystem()` documentation for more details.

Valid options: [Hbonds]/AllBonds/HAngles


.. _yaml_options_restraint_type:

restraint_type
--------------
.. code-block:: yaml

   options:
     restraint_type: flat-bottom

Apply a restraint to the ligand to keep it close to the receptor. This only works in Implicit Solvent. `null` option means no restraint.

Valid options: [flat-bottom]/harmonic/null

|

.. _yaml_options_simulation_parameters:


Simulation Parameters
---------------------


.. [1] Quantiy strings are of the format: `<float> * <unit>` where `<unit>` is any valid unit specified in the "Valid Options" for an option. e.g. "<Quantity Length>" indicates any measure of length may be used for <unit> such as nanometer or angstrom. 
   Compound units are also parsed such as "kilogram / meter**2" for density. 
   Only full unit names as they appear in the simtk.unit package (part of OpenMM) are allowed; so "nm" and "A" will be rejected.
