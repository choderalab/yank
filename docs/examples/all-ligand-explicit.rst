.. _all-ligand-explicit:

Absolute Binding of Binders and Non-binders to T4 lysozyme L99A in Explicit Solvent with Combinatorial Options
==============================================================================================================

This example performs absolute binding free energy calculations for a series of small molecules that contain known
binders and nonbinders to the T4 Lysozyme L99A protein.

We take advantage of three advanced features of YANK in this example (please see the
:doc:`basic tutorial for the YAML sections and binding free energy calculation <p-xylene-explicit>`
for learning how to set up basic YANK simulations):

1. YANK's ability to build out small molecules from SMILES strings (through OpenEye Toolkit)

   * Please see our :ref:`OpenEye Install Instructions <optional_tools>` for installing these tools
   * Alternately see `OpenEye's Instructions <http://docs.eyesopen.com/toolkits/python/quickstart-python/install.html>`_ for additional help

2. YANK's ability to run :ref:`multiple ligands <yaml_molecules_specifiy_names>` through the same commands
3. YANK's ability to run :ref:`Combinatorial <yaml_combinatorial_head>` options

   * See :ref:`the section below <yank_example_combo>` or the :ref:`full documentation page <yaml_combinatorial_head>` for more information on ``!Combinatorial``


This example resides in ``{PYTHON SOURCE DIR}/share/binding/t4-lysozyme``.

Original source ligands collected by the `Shoichet Lab <http://shoichetlab.compbio.ucsf.edu/take-away.php>`_.

Disclaimer on this Example
--------------------------

This example runs many simulations due to the large library of binders and non-binders. You may find it helpful to
change the following lines in the YAML file

.. code-block:: yaml

    binders:
        select: all
    non-binders:
        select: !Combinatorial [...]

to

.. code-block:: yaml

    binders:
        select: <Integer>
    non-binders:
        select: <Integer>

where ``<Integer>`` is a single integer representing a line in the corresponding file.

Examining YAML file
-------------------

Here we look at the ``all-ligands-explicit.yaml`` file in this example, highlighting the differences between this file and similar
files in other examples.

Options Header
^^^^^^^^^^^^^^

There are no new features introduced in this header.

Molecules Header
^^^^^^^^^^^^^^^^

.. code-block:: yaml

    molecules:
      t4-lysozyme:
        filepath: input/receptor.pdbfixer.pdb
        leap:
          parameters: oldff/leaprc.ff14SB
      binders:
        filepath: input/L99A-binders.csv
        antechamber:
          charge_method: bcc
        select: all
      non-binders:
        filepath: input/L99A-non-binders.csv
        antechamber:
          charge_method: bcc
        select: !Combinatorial [4,5,6,7,10,11,12,13,14,15,18,19,28,29,30,31,32,33,34,35,36,37,41,42,43,48,49,50,56,58,59]

The molecules header has quite a number of differences from the other examples. First, there is one receptor, ``t4-lysozyme``,
and two "ligands", which are actually (semi)comma separated value files, ``.csv`` with multiple ligands per file.
Further, there is the ``select`` command with different arguments. Let's break down each part of these ligands one at time

First we look at the CSV file itself. The two files under the ``binders`` and ``non-binders`` headers are formatted the
same way. Each line is a molecule where the second column (semicolon separated) is the SMILES string of that molecule.
The remaining columns do not mater for YANK, so long as the 2nd column is the SMILES string. These files also take
commas as the delimiter. We could have easily made these one file, but kept them split up for the example, and so you
can differentiate between the binder and non-binder set.

When YANK reads a SMILES string, it passes that string off to the OpenEye Toolkit to generate a ligand with all atom
types and coordinates that will be used in YANK. Because the structure it generates is in no way optimized, it is
highly recommended you set ``minimize`` in the primary ``options`` header.

The ``select`` argument tells YANK which line(s) (and therefore molecules) in the multi-line CSV file to read. It defaults
to ``all`` which tells YANK to make a simulation for each molecule in the file, and then run them sequentially. It does
NOT run a single simulation with every molecule present at the same time. Since the default is ``all``, we did not have
to set the option in the ``binders`` molecule, but we explicitly set it so you can see how it works in this example.

The ``select`` option could also accept an integer to choose a single molecule from your CSV file, where the index
starts at 0. e.g. ``select: 0`` chooses the first molecule in the list, after any leading commented lines.

Let us now look at one of YANK's most powerful features the ``!Combinatorial`` options.

.. _yank_example_combo:

!Combinatorial
++++++++++++++

``!Combinatorial`` tells YANK to set up a unique simulation for every entry in the list following the ``!Combinatorial`` command.
YANK will construct a unique simulation for every combination of every set of parameters across all ``!Combinatorial``
lists in the YAML file.

For example, suppose we had

.. code-block:: yaml

    options:
      temperature: !Combinatorial [200*kelvin, 300*kelvin]
    systems:
      leap:
        parameters: !Combinatorial [leaprc.gaff, leaprc.gaff2]

then 4 simulations would be run iterating over every combination across the options. EVERY option can be given the
``!Combinatorial`` flag except for the options in the ``protocols`` and ``solvents`` headers. Take care
of how many of these flags you set as it will increase the number of simulations that have to be run combinatorially.
However YANK will automatically figure out what options should be combined. For instance, if you set a ``!Combinatorial``
option in two separate molecules, they will not necessarily run every combination between the two molecules, UNLESS there
is a system that uses both molecules. It will run a simulation for every option in a given molecule's ``!Combinatorial``
option, but will not cross them unless there is system which combines both.

In this example, the ``!Combinatorial <List of Ints>`` called in the ``non-binders`` molecule selects the indices of
molecules which have an oxygen in them. There is no reason for this list other than we can for this example.
The ``select: all`` is a shortcut in this option for ``select: !Combinatorial [0, 1, 2, 3, 4, 5, ... N]`` where ``N``
is number of molecules in the file.


Solvents Header
^^^^^^^^^^^^^^^

Nothing is changed in this header.


Systems Header
^^^^^^^^^^^^^^

.. code-block:: yaml

    systems:
      t4-ligand:
        receptor: t4-lysozyme
        ligand: !Combinatorial [binders, non-binders]
        solvent: pme
        leap:
          parameters: [oldff/leaprc.ff14SB, leaprc.gaff2, frcmod.ionsjc_tip3p]

Here we choose to use the ``!Combinatorial`` syntax again to specify that we want to try both sets of molecules as our
``ligand``. The output we would expect from this is a unique simulation with every binder and non-binder in both
files. This is a common type of use for ``!Combinatorial`` since you can specify multiple molecules to run in a single
YAML file and compare results when done.

Other Headers
^^^^^^^^^^^^^

The ``experiments`` and ``protocols`` headers are not changed in this example.


Running the Simulation
----------------------

Running the simulation is the same as the other examples where you can either run the ``run-explicit.sh`` script, or
by running ``yank script --yaml=explicit.yaml``. For running on multiple nodes, use ``run-torque-explicit.sh`` and
adapt it to your parallel platform.

The output of this run will be different from simulations where ``!Combinatorial`` is not invoked. First, YANK figures
out all the combinations this run will generate. Next it pre-constructs all the molecules and system files before it
runs any of them. Finally, each simulation is run one after another.

Analyzing the Simulation
------------------------

YANK automatically generates the instructions that ``yank analyze`` will use to compute the free energy difference
for every combination of options. Right now YANK will only tell you the free energy for each individual simulation.
It will be up to you to trap this information and split it into each simulation.

Future versions of YANK will generate more helpful output for ``!Combinatorial`` simulations.

Other Files in this Example
---------------------------

We also provide inputs for running implicit simulation of the same problem.

* ``all-ligands-implicit.yaml`` - YAML file for running implcit solvent
* ``run-all-ligands-implicit.sh`` - Shell script for serial running of the implicit all-ligands example
* ``run-torque-all-ligands-implicit.sh`` - Shell script for parallel/cluster running of the implicit all-ligands example