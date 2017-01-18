.. _yaml_systems_head:

Systems Header for YAML Files
*****************************

The ``systems`` header is where we combine :doc:`molecules <molecules>` and 
:doc:`solvents <solvents>` together with molecular force fields and general
options to create the environment we want to simulation.
There are several ways we can construct a system based on what type of free
energy calculation we want to run and on if you want YANK to handle combining receptors and ligands.

Just like in :doc:`molecules <molecules>`, the solvents can have arbitrary, user defined names.
In the examples, these user defined names are marked as ``{UserDefinedSystem}``.

The ``{UserDefinedSystem}`` references both objects created in the 
:doc:`molecules <molecules>` and :doc:`solvents <solvents>` headers, 
which were also given arbitrary names. 
They will be referenced as ``{UserDefinedMolecule}`` and ``{UserDefinedSolvent}``.

We sort the options by type of free energy calculation due to how the 
arguments change between each type, each requiring their own mandatory arguments.

----

.. _yaml_systems_receptor_ligand:

Ligand/Receptor Free Energies Setup by YANK
===========================================
.. code-block:: yaml

   systems:
     {UserDefinedSystem}:
       receptor: {UserDefinedMolecule}
       ligand: {UserDefinedMolecule}
       solvent: {UserDefinedSolvent}
       pack: no
       leap:
           parameters: [leaprc.ff14SB, leaprc.gaff]

This is the mandatory structure for leting YANK handle the creation and 
setup of a Ligand/Receptor system to do binding free energies.

* ``receptor``: Tells YANK which ``{UserDefinedMolecule}`` to load in as the receptor.
* ``ligand``: Tells YANK which ``{UserDefinedMolecule}`` to load in as the ligand. This molecule will be alchemically modified to compute the free energy difference.
* ``solvent``: Tells YANK  which ``{UserDefinedSolvent}`` to surround the ``receptor`` and ``ligand`` in. This must be set even if you want to use a vacuum, which is a type of ``solvent`` you can define.
* ``pack``: When set to "yes", if the ligand is far away from the receptor or if there are clashing atoms 
  (defined as closer than 1.5 angstroms),  
  Yank will randomly translate and rotate the ligand until  this is solved. Do not set this to "yes" if you
  want your ligand to start from the positions specified in the original mol2/sdf file.
* ``leap`` and ``parameters``: Both options must be specified as shown and tell LEaP which parameter files to use.
  Each file can be one native to LEaP, or a custom file relative to the YAML script location. 
  If multiple files are specified, they must be enclosed around square brackets, ``[ ]``.


.. _yaml_systems_hydration:

Hydration Free Energies Setup By YANK
=====================================
.. code-block:: yaml

   systems:
     {UserDefinedSystem}:
       solute: {UserDefinedMolecule}
       solvent1: {UserDefinedSolvent}
       solvent2: {UerDefinedSolvent}
       leap:
         parameters: [leaprc.ff14SB, leaprc.gaff]

YANK can also do relative hydration free energy simulations. More generally, it can do relative *solvation* free energies between any solvents.
For example, if you specify a reaction-field solvent of water and a vacuum solvent, then you would get the hydration free energy (remember, "vacuum" is a ``{UerDefinedSolvent}`` that must be specified to use).

* ``solute``: Tells yank which ``{UserDefinedMolecule}`` to use for solvation free energy.
* ``solvent1``: The first ``{UserDefinedSolvent}`` to use for the solvation free energy.
* ``solvent2``: The second ``{UserDefinedSolvent}`` to use for the solvation free energy.
* ``leap`` and ``parameters``:  Both options must be specified as shown and tell LEaP which parameter files to use.
  Each file can be one native to LEaP, or a custom file relative to the YAML script location.
  If multiple files are specified, they must be enclosed around square brackets, ``[ ]``.


.. _yaml_systems_user_defined:

Arbitrary Phase Free Energies Setup by User
===========================================
.. code-block:: yaml

   systems:
     {UserDefinedSystem}:
       phase1_path: [complex.prmtop, complex.inpcrd]
       phase2_path: [solvent.top, solvent.gro]
       ligand_dsl: resname MOL
       solvent_dsl: resname WAT  # optional
       solvent: {UserDefinedSolvent}
       gromacs_include_dir: include/

YANK will allow users to specify arbitrary free energy calculations with systems they have prepared themselves.
Both Amber and GROMACS input file types are accepted. It is also possible to specify a pair of ``[*.pdb, *.xml]``
files for each phase, where the XML contains a serialized OpenMM system.
MDTraj is required to use this options since picking the ligand out of the files is done with an MDTraj DSL.

* ``phase1_path``: The set of files which fully describe the first phase of the free energy simulation you want to run.
* ``phase2_path``: The set of files which fully describe the second phase of the free energy simulation you want to run.
* ``ligand_dsl``: An MDTraj DSL string which identifies the ligand in the files provided by ``phase1_path`` and ``phase2_path``.
* ``solvent_dsl``: An optional MDTraj DSL string which identifies the solvent atoms in the files provided by ``phase1_path``
  and ``phase2_path``. If not specified, a list of common solvent residue names will be used to automatically detect
  solvent atoms.
* ``solvent``: A ``{UserDefinedSolvent}`` to put the two phases in. Only one solvent is allowed for this calculation.
  This option must be omitted if using XML/PDB files, since the solvent options are inherently specified in the XML
  definition of the system. Finally, if the two phases require two different solvents, it is possible to substitute the
  ``solvent`` option with two ``solvent1`` and ``solvent2``, which are associated to phase 1 and phase 2 respectively.
* ``gromacs_include_dir``: *Optional*, Tells YANK where the GROMACS include directory is to pull files and parameters from.
  This is particularly helpful if your topology file does not contain all parameters.
  Path is relative to the YAML script.
