.. _yaml_solvents_head:

Solvents Header for YAML Files
******************************

The ``solvents`` header lets users specify what solvents are available to the systems they will create. 
"Solvents" in this case is a very general term and includes the ability to specify vacuum (i.e. no solvent). 
Just like in :doc:`molecules <molecules>`, the solvents can have arbitrary, user defined names. 
In the examples, these user defined names are marked as ``{UserDefinedSolvent}``.

You can define as many ``{UserDefinedSolvent}`` as you like. These solvents will be used in other YAML headers.

Most of the solvent options are tied directly to OpenMM options of the same name in the primary function ``simtk.openmm.app.amberprmtopfile.AmberPrmtopFile.createSystem()``, however, there are other options tied to LEaP preparation instructions.
Similarly, the ``solvents`` section is where you specify Periodic Boundary Conditions (PBC) and long range electrostatic treatments.
Each of the arguments in this category is optional and the default option will be assumed if not specified.


----

.. _yaml_solvents_openmm_system_options:

OpenMM System Creation Options
==============================

.. _yaml_solvents_nonbonded_method:

nonbonded_method
----------------
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
* ``CutoffPeriodic``: Specifies a periodic system with a cutoff scheme and Reaction Field electrostatics for long range interactions. This is YANK's currently prefered mode for explicit solvent.
* ``Ewald``: Specifies a periodic system with a cutoff scheme and Ewald decomposed electrostatics. This is typically not used over PME
* ``PME``: Specifies a periodic system with a cutoff scheme and Particle Mesh Ewald (PME) decomposed electrostatics.  Currently support by YANK, however, there will be some error introduced by YANK due to the inability efficiently treat long range alchemical PME electrostatics. A solution is being worked on.


.. _yaml_solvents_nonbonded_cutoff:

nonbonded_cutoff
----------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       nonbonded_cutoff: 1 * nanometer

Specify the cutoff radius for the ``nonbonded_method``\s which rely on it. 
What happens beyond this cutoff depends both on the ``nonbonded_method`` and the ``switchDistance``.

Nonbonded Methods: ``CutoffPeriodic``, ``Ewald``, ``PME``

Valid Options (1 * nanometer): <Quantity Length> [1]_


.. _yaml_solvents_constraints:

constraints
-----------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       constraints: Hbonds

.. todo:: Figure out which ``constraints`` option overwrites the other.

Specify constraints over the entire system. This option is redundant with the general options for :ref:`constraints <yaml_options_constraints>` and should almost never be set here unless you want to specify two solvents with different constraints.

Nonbonded Methods: All

Valid Options: [None] / Hbonds / AllBonds / HAngles


.. _yaml_solvents_rigid_water:

rigid_water
-----------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvent}:
       rigid_water: True

.. todo:: Check if this is True/False or yrs/no

If True, the water molecules will be fully rigid, regardless of the settings in ``constraints``.

Nonbonded Methods: All

Valid Options: [True] / False 


.. _yaml_solvents_implicit_solvent:

implicit_solvent
----------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvents}:
       implicit_solvent: OBC2

Specify an implicit solvent model. Please check the OpenMM documentation on each option to see the differences in the models. 

Nonbonded Methods: ``NoCutoff``, ``CutoffNonPeriodic``

Valid Options: [None] / HCT / OBC1 / OBC2 / GBn / GBn2


.. _yaml_options_implicit_solvent_salt_conc:

implicit_solvent_salt_concentration
-----------------------------------
.. code-block:: yaml

   solvents:
     {UserDefinedSolvents}:
       implicit_solvent_salt_concentratio: 1.0 * moles / liter

Specify the salt concentration of the implicit model. Requires that ``implicit_solvent != None``.

Nonbonded Methods: ``NoCutoff``, ``CutoffNonPeriodic``

Valid Options (0.0 * moles / liter): <Quantity Moles / Volume> [1]_

.. [1] Quantiy strings are of the format: ``<float> * <unit>`` where ``<unit>`` is any valid unit specified in the "Valid Options" for an option.
   e.g. "<Quantity Length>" indicates any measure of length may be used for <unit> such as nanometer or angstrom. 
   Compound units are also parsed such as ``kilogram / meter**3`` for density. 
   Only full unit names as they appear in the simtk.unit package (part of OpenMM) are allowed; so "nm" and "A" will be rejected.
