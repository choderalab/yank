.. _p-xylene-implicit:

p-xylene binding to T4 lysozyme L99A in implicit solvent
========================================================

This example illustrates the computation of the binding free energy of p-xylene to the T4 lysozyme L99A mutant in implicit solvent.

.. code-block:: none

   $ cd examples/p-xylene-explicit

To set up the calculation, we designate p-xylene (residue name `MOL`) as the ligand.
A harmonic restraint is used to keep the two molecules from drifting off to infinity.
The `OBC2` implicit solvent model is used, and the temperature is set to 300 Kelvin.

.. code-block:: none

   $ yank prepare binding amber --setupdir=setup --ligand="resname MOL" --store=output --iterations=1000 --restraints=harmonic --gbsa=OBC2 --temperature=300*kelvin --verbose

Run the simulation in serial mode with verbose output:

.. code-block:: none

   $ yank run --store=output --verbose

Analyze the results of the simulation:

.. code-block:: none

   $ yank analyze


