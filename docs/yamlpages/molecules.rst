.. _yaml_molecules_head:

Molecules Header for YAML Files
*******************************

Everything under the ``molecules`` defines what molecules are in your systems.
You can specify your own molecule names.
Because of this user defined names in the syntax examples are marked as ``{UserDefinedMolecule}``.

You can define as many ``{UserDefinedMolecule}`` as you like.
These molecules will be used in other YAML headed sections.

Unlike the primary :doc:`options <options>` for YAML files,
many of these settings are optional and do nothing if not specified.
The mandatory/optional of each setting (and what conditionals),
as well as the default behavior of each setting is explicitly stated in the setting's description.

All of the molecules will be built, even if they are not used in a later system, so ensure your molecules do not
have errors or are commented out.

----

.. _yaml_molecules_specify_names:

Specifying Molecule Names
=========================

.. _yaml_molecules_filepath:

.. rst-class:: html-toggle

``filepath``
------------
.. code-block:: yaml

    molecules:
      {UserDefinedMolecule}:
        filepath: benzene.pdb

Filepath of the molecule. Path is relative to the directory the YAML script is in. Depending on what type of molecule
you pass in (ligand vs. protein) will determine what mandatory arguments are required. For small molecules, you need a
way to specify the charges, proteins however can rely on built in force field parameters.

**MANDATORY** but exclusive with :ref:`smiles <yaml_molecules_smiles>` and :ref:`name <yaml_molecules_name>`

Valid Filetypes: PDB, mol2, sdf, cvs

**Note:** If CVS is specified and there is only one moleucle, the first column must be a SMILES string.
If multiple molecules are to be used (for the :doc:`!Combinatorial <combinatorial>` ability),
then each row is its own molecule where the second column is the SMILES string.



.. _yaml_molecules_smiles:

.. rst-class:: html-toggle

``smiles``
----------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       smiles: c1ccccc1

YANK can process SMILES stings to build the molecule as well. Usually only recommended for small ligands.
Requires that the OpenEye Toolkits are installed.

**MANDATORY** but exclusive with :ref:`filepath <yaml_molecules_filepath>` and :ref:`name <yaml_molecules_name>`



.. _yaml_molecules_name:

.. rst-class:: html-toggle

``name``
--------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       name: benzene

YANK can process raw molecule name if the OpenEye Toolkits are installed

**MANDATORY** but exclusive with :ref:`filepath <yaml_molecules_filepath>` and :ref:`smiles <yaml_molecules_smiles>`




.. _yaml_molecules_strip_protons:

.. rst-class:: html-toggle

``strip_protons``
-----------------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       strip_protons: no

Specifies if LEaP will re-add all hydrogen atoms.
This is helpful if the PDB contains atom names for hydrogens that AMBER does not recognize.
Primarily for proteins, not small molecules.

**OPTIONAL** and defaults to ``no``

Valid Options: [no]/yes




.. _yaml_molecules_pdbfixer:

.. rst-class:: html-toggle

``pdbfixer``
------------

.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       pdbfixer:
         replace_nonstandard_residues: no
         remove_heterogens: none
         add_missing_atoms: none
         apply_mutations:
           mutations: T315I
           chain_id: A

Specifies whether PDBFixer should be used to modify the molecule.
Can only be used on proteins, on files with ``.pdb`` file extensions.

Replacing nonstandard residues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       pdbfixer:
         replace_nonstandard_residues: [no] | yes

If ``yes`` is specified, nonstandard amino acid residues will be replaced with
one of the 20 standard amino acids according to the scheme used by
`PDBFixer <http://htmlpreview.github.io/?https://raw.github.com/pandegroup/pdbfixer/master/Manual.html>`_.

**OPTIONAL** with default value of ``no``

Valid Options: [no]/yes

Removing heterogens
^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       pdbfixer:
         remove_heterogens: [none] | water | all

This directs PDBFixer to remove some heterogen residues from the PDB file:

* ``all`` : all heterogens (residues that are not one of the standard amino acids) will be removed
* ``water`` : only water residues will be removed
* ``none`` : no residues will be removed

**OPTIONAL** with default value of ``none``

Valid Options: [none]/water/all

Adding missing residues and atoms atoms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add missing atoms (including entire residues, loops, and termini).

**WARNING:** PDBFixer uses a very simple approach to adding missing residues that will only
produce sensible geometries in the simplest of cases. Use this option with caution.

.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       pdbfixer:
         add_missing_residues: [no] | yes
         add_missing_atoms: none | [heavy] | hydrogens | all
         ph: 7.4

``add_missing_residues`` specifies whether missing residues should be added

* ``no`` : only missing atoms in existing residues will be added (DEFAULT)
* ``yes`` : missing residues specified in SEQRES will be added

``add_missing_atoms`` specifies which detected missing atoms should be added:

* ``none`` : no missing atoms will be added
* ``heavy`` : only heavy atoms will be added (DEFAULT)
* ``hydrogens`` : add only hydrogens (not recommended)
* ``all`` : all missing atoms (including hydrogens) will be added (not recommended)

``ph`` specifies the pH to be used for adding hydrogens (default: 7.4)

**OPTIONAL**

Mutations
^^^^^^^^^

Make the directed mutations to amino acid residues.

.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       pdbfixer:
         apply_mutations:
           mutations: T315I
           chain_id: A

Mutations are specified using the format ``<original-one-letter-code><resid><new-one-letter-code>``,
with the character ``/`` being an optional separator if multiple mutations are desired.

The initial PDB file numbering is used for residue identifier ``resid``.

Examples for specifying ``mutations:``:

* ``T315I`` : a single mutation that changes Thr at resid 315 to Ile
* ``L858R/T790M`` : a double mutation

If ``chain_id`` is not specified, it defaults to ``null`` (no chain designator).

PDBFixer is applied after ``strip_protons`` if both are requested.

**OPTIONAL**



.. _yaml_molecules_select:

.. rst-class:: html-toggle

``select``
----------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       filepath: clinical-kinase-inhibitors.csv
       antechamber:
           charge_method: bcc
       select: !Combinatorial [0, 3]

The "select" keyword works the same way if you specify a
pdb, mol2, sdf, or cvs file containing multiple structures.
``select`` has 3 modes:

1. ``select: all`` includes all the molecules in the given file.
2. ``select: <Integer>`` picks the molecule in the file with index ``<Integer>``
3. ``select: !Combinatorial: <List of Ints>`` pick specific indices in the file. See :doc:`Combinatorial <combinatorial>` options for more information.

Indexing starts at 0.

**OPTIONAL** with default value of ``all``

Valid Options: [all]/<Integer>/<Combinatorial List of ints>

|

.. _yaml_molecules_assign_charges:

Assigning Missing Parameters
============================



.. _yaml_molecules_antechamber:

.. rst-class:: html-toggle

``antechamber``
---------------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       filepath: benzene.mol2
       antechamber:
         charge_method: bcc

Pass the molecule through AmberTools ANTECHAMBER to assign missing parameters such as torsions and angle terms.

``charge_method`` is a required sub-directive which allows assigning missing charges to the molecule. It is either given
a known charge method to ANTECHAMBER method or ``null`` to skip assigning charges. The later is helpful when you
already have the charges, but are missing other parameters.

**OPTIONAL**

**PARTIALLY EXCLUSIVE** If you have acess to the OpenEye toolkits and want to use them to **assign partial charges**
to the atoms through the :ref:`openeye <yaml_molecules_openeye>` command, then you should set ``charge_method`` to ``null``.
ANTECHAMBER can still get the other missing parameters such as torsions and angles.

**OPTIONALLY SUPERSEDED** by :ref:`leap <yaml_molecules_leap>` or the :ref:`leap argument in systems <yaml_systems_head>`.
If the parameter files you feed into either ``leap`` argument have the charges and molecular parameters already
included (such as standard protein residues in many force fields), then there is no need to invoke this command. If the
force fields you give to the ``leap`` commands are missing parameters though, you should call this.



.. _yaml_molecules_openeye:

.. rst-class:: html-toggle

``openeye``
-----------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       filepath: benzene.mol2
       openeye:
         quacpac: am1-bcc

Use the OpenEye Toolkits if installed to determine molecular charge.
Only the current options as shown are permitted and must be specified as shown.

**OPTIONAL**

**PARTIALLY EXCLUSIVE** If you want to use :ref:`antechamber <yaml_molecules_antechamber>` to assign partial charges,
do not use this command. However, if you want to use :ref:`antechamber <yaml_molecules_antechamber>` to only get other
missing parameters such as torsions and angles, use this command but set ``charge_method`` to ``null`` in
:ref:`antechamber <yaml_molecules_antechamber>`

**OPTIONALLY SUPERSEDED** by :ref:`leap <yaml_molecules_leap>` or the :ref:`leap argument in systems <yaml_systems_head>`.
If the parameter files you feed into either ``leap`` argument have the charges and molecular parameters already
included (such as standard protein residues in many force fields), then there is no need to invoke this command. If the
force fields you give to the ``leap`` commands are missing partial charges though, you should call this.

|

.. _yaml_molecules_extras:

Assigning Extra Information
===========================



.. _yaml_molecules_leap:

.. rst-class:: html-toggle

``leap``
--------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       leap:
         parameters: [mymol.frcmod, mymol.off]

Load molecule-specific force field parameters into the molecule.
These can be created from any source so long as leap can parse them.
It is possible to assign partial charges with the files read in this way,
which would supersede the options of
:ref:`antechamber <yaml_molecules_antechamber>`
and :ref:`openeye <yaml_molecules_openeye>`.

This command has only one mandatory subargument ``parameters``,
which can accept both single files as a string,
or can accept a comma separated list of files enclosed by [ ].
Filepaths are relative to either the AmberTools default paths or to the folder the YAML script is in.

*Note*: Proteins do not necessarily   need this command if the force fields given to the :ref:`leap argument in systems <yaml_systems_head>` will fully describe them.

**OPTIONAL**



.. _yaml_molecules_epik:

.. rst-class:: html-toggle

``epik``
--------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
        epik:
          select: 0
          ph: 7.6
          ph_tolerance: 3.0
          tautomerize: no

Run Schrodinger's tool Epik with to select the most likely protonation state for the molecule in solution. Parameters
in this call are direct reflections of the function to invoke ``epik`` from OpenMolTools. Each of the parameters in this
list (with the exception of ``select``) are optional.

We note that the option ``ph_tolerance`` set to a value here of
``3.0``, the pH range which will be searched will be ``pH +- 3.0``, which is a 7 log unit range, which may take a
some time to enumerate, although will likely be less than the simulation overall. Should you feel this time is
too long, you might consider reducing the ``ph_tolerance``.

**OPTIONAL**


.. _yaml_molecules_regions:

.. rst-class:: html-toggle

``regions``
-----------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
        regions:
           {UserDefinedRegion}: region_string
           ...

Define molecular regions in the molecule which can be used in upcoming features such as defining restraint regions in
more general ways, or specific atom subsets you want to track through the :class:`yank.yank.Topography` object which is
stored as part of the simulation's metadata, accessible through :class:`yank.repex.Reporter`.

Any number of user defined regions can be specified for every molecule, so long as their name is unique between all
molecules which ultimately wind up in a :doc:`system <systems>`. E.g. If you have 2 ligands you want to bind to a
receptor in a combinatorial setup, both ligands can have a region named "my_region" since they will never be in the
same system together. However, the receptor cannot have a region named "my_region" as well, as that will
be ambiguous as to which region, ligand or receptor, to define.

The regions apply only to the molecule the ``regions`` section is under, so even if the atom index changes in the
:class:`yank.yank.Topography`, the atomic indices defined in the ``region`` entry will be converted.

The region definition supports multiple selection formats:

* DSL String: An MDTraj DSL string which identifies will identify a region.
* **Future Ability** SMARTS String: Molecular selection format similar to regular expression for strings, but for
  molecules instead. This feature is not in yet, but is planned. The regions framework is the pre-cursor to this
  feature. See
  `Daylight's website for more information on SMARTS <http://www.daylight.com/dayhtml/doc/theory/theory.smarts.html>`_.
* List of Ints: Select atoms by integers, this applies only to the final system, so numbers will probably not align
  with the atom numbers from the input files.
* Single Int: Same as the list of ints, but with a single entry, subject to same rules

**OPTIONAL**
