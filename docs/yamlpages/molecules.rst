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


----

.. _yaml_molecules_specifiy_names:

Specifying Molecule Names
=========================

.. _yaml_molecules_filepath:

filepath
--------
.. code-block:: yaml

    molecules:
      {UserDefinedMolecule}:
        filepath: benzene.pdb

Filepath of the molecule. Path is relative to the directory the YAML script is in. Depending on what type of molecule you pass in (ligand vs. protein) will determine what mandatory arguments are required. For small molecules, you need a way to specify the charges, proteins however can rely on built in force field parameters. 

**MANDATORY** but exclusive with :ref:`smiles <yaml_molecules_smiles>` and :ref:`name <yaml_molecules_name>`

Valid Filetypes: PDB, mol2, sdf, cvs

**Note:** If CVS is specified and there is only one moleucle, the first column must be a SMILES string. 
If multiple molecules are to be used (for the :doc:`!Combinatorial <combinatorial>` ability), 
then each row is its own molecule where the second column is the SMILES string.

.. _yaml_molecules_smiles:

smiles
------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       smiles: c1ccccc1

YANK can process SMILES stings to build the molecule as well. Usually only recommended for small ligands. 
Requires that the OpenEye Toolkits are installed.

**MANDATORY** but exclusive with :ref:`filepath <yaml_molecules_filepath>` and :ref:`name <yaml_molecules_name>`


.. _yaml_molecules_name:

name
----
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       name: benzene

YANK can process raw molecule name if the OpenEye Toolkits are installed

**MANDATORY** but exclusive with :ref:`filepath <yaml_molecules_filepath>` and :ref:`smiles <yaml_molecules_smiles>`


.. _yaml_molecules_strip_protons:

strip_protons
-------------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       strip_protons: no

Specifies if LEaP will re-add all hydrogen atoms. 
This is helpful if the PDB contains atom names for hydrogens that AMBER does not recognize. 
Primarily for proteins, not small molecules.

**OPTIONAL** and defaults to ``no``

Valid Options: [no]/yes


.. _yaml_molecules_select:

select
------
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

antechamber
-----------
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

**OPTIONALLY SUPERSEDED** by :ref:`leap <yaml_molecules_leap>` or the :ref:`leap argument in systems <yaml_systems_leap>`.
If the parameter files you feed into either ``leap`` argument have the charges and molecular parameters already
included (such as standard protein residues in many force fields), then there is no need to invoke this command. If the
force fields you give to the ``leap`` commands are missing parameters though, you should call this.


.. _yaml_molecules_openeye:

openeye
-------
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

**OPTIONALLY SUPERSEDED** by :ref:`leap <yaml_molecules_leap>` or the :ref:`leap argument in systems <yaml_systems_leap>`.
If the parameter files you feed into either ``leap`` argument have the charges and molecular parameters already
included (such as standard protein residues in many force fields), then there is no need to invoke this command. If the
force fields you give to the ``leap`` commands are missing partial charges though, you should call this.

|

.. _yaml_molecules_extras:

Assigning Extra Information
===========================

.. _yaml_molecules_leap:

leap
----
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

*Note*: Protiens do not necessarily   need this command if the force fields given to the :ref:`leap argument in systems <yaml_systems_leap>` will fully describe them.

**OPTIONAL**


.. _yaml_molecules_epik:

epik
----
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
        epik:
          select: 0
          ph: 7.6
          ph_tolerance: 0.7
          tautomerize: no

Run Schrodinger's tool Epik with to select the most likely protonation state for the molecule in solution. Parameters in this call are direct reflections of the function to invoke epik from OpenMolTools.

**OPTIONAL**
