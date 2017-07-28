.. _yaml_solvents_head:

Solvents Header for YAML Files
******************************

The ``solvents`` header lets users specify what solvents are available to the systems they will create. 
"Solvents" in this case is a very general term and includes the ability to specify vacuum (i.e. no solvent). 
Just like in :doc:`molecules <molecules>`, the solvents can have arbitrary, user defined names. 
In the examples, these user defined names are marked as ``{UserDefinedSolvent}``.

You can define as many ``{UserDefinedSolvent}`` as you like. These solvents will be used in other YAML headers.

Most of the solvent options are tied directly to OpenMM options of the same name in the primary function
``simtk.openmm.app.amberprmtopfile.AmberPrmtopFile.createSystem()``, however, there are other options tied to LEaP preparation instructions.
Similarly, the ``solvents`` section is where you specify Periodic Boundary Conditions (PBC) and long range electrostatic treatments.
Each of the arguments in this category is optional and the default option will be assumed if not specified.


----

.. _yaml_solvents_openmm_system_options:

OpenMM System Creation Options
==============================



.. _yaml_solvents_nonbonded_method:

.. rst-class:: html-toggle

``nonbonded_method``
--------------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       nonbonded_method: NoCutoff

Specify the nonbonded scheme that the solvent is put in. This argument defines the type of solvent (none, implicit, explicit). 
Although technically optional, this is the most important setting in this header.
Because each option has very different behavior, we list each of them here and subsequent options will note which mode they apply to.

* ``NoCutoff``: **Default**. Specifies a non-periodic system with no cutoff. 
  This is the de facto choice for vacuum and implicit solvent.
* ``CutoffNonPeriodic``: Specify a non-periodic system which uses a cutoff scheme to reduce the computational overhead of computing long range interactions. 
  This scheme is helpful for vacuum and implicit solvent systems with large number of molecules where computing the pairwise interaction at long range between all atoms is inefficient.
* ``CutoffPeriodic``: Specifies a periodic system with a cutoff scheme and Reaction Field electrostatics for long range interactions.
  There is some cutoff based error associated with using this option and is not currently recommended.
* ``Ewald``: Specifies a periodic system with a cutoff scheme and Ewald decomposed electrostatics. This is typically not used over PME
* ``PME``: Specifies a periodic system with a cutoff scheme and Particle Mesh Ewald (PME) decomposed electrostatics. 
  Currently support by YANK, however, there is a small error introduced by YANK due to the inability efficiently treat long range alchemical PME electrostatics during simulation.
  We partially correct for this error by computing it at run-time at the cost of a bit of computational overhead.
  **This is YANK's currently preferred mode for explicit solvent.**



.. _yaml_solvents_nonbonded_cutoff:

.. rst-class:: html-toggle

``nonbonded_cutoff``
--------------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       nonbonded_cutoff: 1 * nanometer

Specify the cutoff radius for the ``nonbonded_method``'s which rely on it.
What happens beyond this cutoff depends both on the ``nonbonded_method`` and the ``switch_distance``.

Nonbonded Methods: ``CutoffPeriodic``, ``Ewald``, ``PME``

Valid Options (1 * nanometer): <Quantity Length> [1]_



.. _yaml_solvents_switch_distance:

.. rst-class:: html-toggle

``switch_distance``
-------------------
.. code-block:: yaml

   solvents:
      {UserDefinedSolvent}:
         switch_distance: 0.9 * nanometer

The distance at which the potential energy switching function is turned on for Lennard-Jones interactions. 
If the ``switch_distance`` is 0 (or not specified), no switching function will be used. 
Values greater than ``nonbonded_cutoff`` or less than 0 raise errors.

Nonbonded Methods: ``CutoffPeriodic``, ``Ewald``, ``PME``

Valid Options (0 * nanometer) <Quantity Length> [1]_




.. _yaml_solvents_solvent_model:

.. rst-class:: html-toggle

``solvent_model``
-----------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       solvent_model: tip4pew

Specify the water model used to solvate the box.

Nonbonded Methods: ``CuttoffNonPeriodic``, ``CuttoffPeriodic``, ``Ewald``, ``PME``

Valid Options: [tip4pew] / tip3p / tip3pfb / tip5p / spce




.. _yaml_solvents_rigid_water:

.. rst-class:: html-toggle

``rigid_water``
---------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       rigid_water: True

If True, the water molecules will be fully rigid, regardless of the settings in :ref:`yaml_options_constraints`.

Nonbonded Methods: All

Valid Options: [True] / False 




.. _yaml_solvents_implicit_solvent:

.. rst-class:: html-toggle

``implicit_solvent``
--------------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       implicit_solvent: OBC2

Specify an implicit solvent model. Please check the OpenMM documentation on each option to see the differences in the models.

When not specified, no implicit solvent is set.

Nonbonded Methods: ``NoCutoff``, ``CutoffNonPeriodic``

Valid Options: HCT / OBC1 / OBC2 / GBn / GBn2



.. _yaml_options_implicit_solvent_salt_conc:

.. rst-class:: html-toggle

``implicit_solvent_salt_concentration``
---------------------------------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       implicit_solvent_salt_concentration: 1.0 * moles / liter

Specify the salt concentration of the implicit model. Requires an ``implicit_solvent``.

You may also specify a Debye length ``temperature`` parameter which accepts <Quantity Temperature> [1]_ as an argument, default ``300 * kelvin``.
*Note*: This is NOT the temperature for the system as a whole.

Nonbonded Methods: ``NoCutoff``, ``CutoffNonPeriodic``

Valid Options (0.0 * moles / liter): <Quantity Moles / Volume> OR <Quantity Temperature> [1]_



.. _yaml_options_solute_dielectric:

.. rst-class:: html-toggle

``solute_dielectric``
----------------------
.. code-block:: yaml
   
   solvents:
     {UserDefinedSolvent}:
       solute_dielectric: 1.5

Specify the dielectric of the solute molecules.

Nonbonded Methods: ``NoCutoff``, ``CutoffNonPeriodic``

Valid Options (1.0): <Float>



.. _yaml_options_solvent_dielectric:

.. rst-class:: html-toggle

``solvent_dielectric``
----------------------
.. code-block:: yaml
   
   solvents:
     {UserDefinedSolvent}:
       solvent_dielectric: 78.5

Specify the dielectric of the implcit solvent models

Nonbonded Methods: ``NoCutoff``, ``CutoffNonPeriodic``

Valid Options (78.5): <Float>



.. _yaml_options_ewald_error_tol:

.. rst-class:: html-toggle

``ewald_error_tolerance``
-------------------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       ewald_error_tolerance: 0.0005

The relative error tolerance to use for Ewald summations. 
There are very few times this will need to be explicitly set.

Nonbonded Methods: ``Ewald``, ``PME``

Valid Options (0.0005): <Float>

|

.. _yaml_solvents_LEaP_options:

LEaP Solvation Options
======================



.. _yaml_solvents_clearance:

.. rst-class:: html-toggle

``clearance``
-------------
.. code-block:: yaml
   
   solvents:
     {UserDefinedSolvent}:
       clearance: 10 * angstrom

The edge of the solvation box will be at ``clearance`` distance away from any atom of the receptor and ligand.
This method is a way to solvate without explicitly defining solvent atoms.
We highly recommend  having a 
:ref:`number of equilibration iterations <yaml_options_number_of_equilibration_iterations>` 
when this option is invoked.

This option is mandatory only for systems that need to go through the automatic preparation pipeline, and it is ignored
for systems :ref:`defined by Amber, GROMACS, or OpenMM files <yaml_systems_user_defined>`.

Nonbonded Methods: ``CuttoffNonPeriodic``, ``CuttoffPeriodic``, ``Ewald``, ``PME``

Valid Options: <Quantity Length> [1]_




.. _yaml_solvents_positive_ion:

.. rst-class:: html-toggle

``positive_ion``
----------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       positive_ion: Na+

Specifies the positive counter ions that will be added as needed.

No positive counter ions will be added if this option is not specified. Note that the name must match a known atom type
in LEaP based on the parameter files you specified to load.

Nonbonded Methods: ``CuttoffPeriodic``, ``Ewald``, ``PME``

Valid Options: <Ion Symbol and charge>




.. _yaml_solvents_negative_ion:

.. rst-class:: html-toggle

``negative_ion``
----------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       negative_ion: Cl-

Specifies the negative counter ions that will be added as needed.

No negative counter ions will be added if this option is not specified. Note that the name must match a known atom type
in LEaP based on the parameter files you specified to load.

Nonbonded Methods: ``CutoffPeriodic``, ``Ewald``, ``PME``

Valid Options: <Ion Symbol and charge>




.. _yaml_solvents_ionic_strength:

.. rst-class:: html-toggle

``ionic_strength``
------------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       ionic_strength: 0.0*molar

The ionic strength of the ions.

Both ``positive_ion`` and ``negative_ion`` must be specified with this, and only monovalent ions are supported. Note
that this does not include the ions that are used to neutralize the periodic box.

Nonbonded Methods: ``CutoffPeriodic``, ``Ewald``, ``PME``

Valid Options (0 * molar) <Quantity Length> [1]_




.. [1] Quantity strings are of the format: ``<float> * <unit>`` where ``<unit>`` is any valid unit specified in the "Valid Options" for an option.
   e.g. "<Quantity Length>" indicates any measure of length may be used for <unit> such as nanometer or angstrom.
   Compound units are also parsed such as ``kilogram / meter**3`` for density.
   Only full unit names as they appear in the simtk.unit package (part of OpenMM) are allowed; so "nm" and "A" will be rejected.
