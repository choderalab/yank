.. _freesolv-imp-exp:

Hydration Free Energy of part of the FreeSolv Database in Implicit and Explicit Solvent
=======================================================================================

This example looks at a the hydration free energy of a subset of the
`FreeSolv <http://link.springer.com/article/10.1007%2Fs10822-014-9747-x>`_
`database <https://github.com/MobleyLab/FreeSolv>`_ in both implicit and explicit solvent.

This example showcases how easy it is to compute the free energy of a large dataset from a single input file, and how
easy it is to set up that file. Here we combine everything we learned from the previous examples into one. If any of the
following ideas are unfamiliar, please see the linked examples.

1. :doc:`Setting up general options, molecules, protocols, and experiments <p-xylene-explicit>`
2. :doc:`Setting up implicit solvent simulations <host-guest-implicit>`
3. :doc:`Configuring multiple solvents for hydration free energy calculations <hydration-phenol-explicit>`
4. :doc:`Running combinatorial simulations and constructing molecules from SMILES strings <all-ligand-explicit>`

This example resides in ``{PYTHON SOURCE DIR}/share/hydration/freesolv``.

Examining YAML file
-------------------

Options Header
^^^^^^^^^^^^^^

Here we set an NPT simulation (it will be NVT for implicit solvent by default), where we also have verbose set and
minimize.

Molecules Header
^^^^^^^^^^^^^^^^

Here we target a ``.smiles`` file which behaves identical to a ``.csv`` file. Multiple molecules are specified, one per
line, where the second column is the SMILES string of the molecule.

We get the missing molecule parameters with ANTECHAMBER and charge the molecule with AM1-BCC.


Solvents Header
^^^^^^^^^^^^^^^

We specify three solvents for this example: an explicit solvent (TIP3P with PME), and implicit solvent (GBSA), and
the vacuum solvent. We only need the vacuum and one other one to do a hydration free energy calculation, but we want to
set up both implicit and explicit solvent simulations in the same file.

Systems Header
^^^^^^^^^^^^^^

Here we setup a hydration free energy system a normal, but we specify for ``solvent1`` that we want the ``!Combinatorial``
option so we run the database through both our implicit and explicit solvents.

Protocols Header
^^^^^^^^^^^^^^^^

We specify a standard alchemical pathway which our simulations will run through.

Experiments Header
^^^^^^^^^^^^^^^^^^

Lastly we put together our experiment that will combine our protocol and system to tell YANK what to run.

Running the Simulation
----------------------

Running the simulation is the same as the other examples where you can either run the ``run-yank.sh`` script, or
by running ``yank script --yaml=yank.yaml``. For running on multiple nodes, use ``run-torque-yank.sh`` and
adapt it to your parallel platform.
