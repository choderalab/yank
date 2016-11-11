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
       parameter: antechamber
       select: [0, 3]
       
The "select" keyword works the same way if you specify a 
pdb, mol2, sdf, or cvs file containing multiple structures. 
You can alternatively specify ``select: all`` which includes 
all the molecules in the given file. 
Indexing starts at 0.

**OPTIONAL** with default value of ``all``

Valid Options: [all]/<Integer>/<List of ints>

|

.. _yaml_molecules_assign_charges:

Assigning Parital Charges
=========================

.. _yaml_molecules_antechamber:

antechamber
-----------
.. code-block:: yaml

   molecules:
     {UserDefinedMolecule}:
       filepath: benzene.mol2
       antechamber:
         charge_method: bcc

Pass the molecule through AmberTools ANTECHAMBER to assign changes and parameters. Fine grain control is handled through the ``charge_method`` argument which accepts either a known ANTECHAMBER method or ``null`` for none.

Primarially used to assign charges to small molecules

**MANDATORY** but exclusive with :ref:`openeye <yaml_molecules_openeye>` but...

**OPTIONALLY SUPERSEDED** by :ref:`leap <yaml_molecules_leap>` if pre-processed partial charge data is avilalble for small molecules OR if the partial charge data is included as part of the protein force feild used to buld the :ref:`leap argument in systems <yaml_systems_leap>`.


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
Specifying this method is preferred over :ref:`antechamber <yaml_molecules_antechamber>` if available.

**MANDATORY** but exclusive with :ref:`antechamber <yaml_molecules_antechamber>` *for assigning charges*. `antechamber: null` *must still be set* to get the missing atomic parameters from GAFF. However...

**OPTIONALLY SUPERSEDED** by :ref:`leap <yaml_molecules_leap>` 
if pre-processed partial charge data is available for small molecules OR 
if the partial charge data is included as part of the protein 
force feild used to build the :ref:`leap argument in systems <yaml_systems_leap>`.

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
