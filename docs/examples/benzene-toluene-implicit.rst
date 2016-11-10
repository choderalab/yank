.. _benzene-toluene-implicit:

benzene-toluene in implicit solvent
===================================

This simple example demonstrates the computation of the association constant between benzene and toluene in implicit solvent.

.. code-block:: none

   $ cd examples/benzene-toluene-implicit

To set up the calculation, we designate benzene (residue name `BEN`) as the ligand.
A harmonic restraint is used to keep the two molecules from drifting off to infinity.
The `OBC2` implicit solvent model is used, and the temperature is set to 300 Kelvin.

.. code-block:: none

   $ yank prepare binding amber --setupdir=setup --ligand="resname BEN" --store=output --iterations=1000 --restraints=Harmonic --gbsa=OBC2 --temperature=300*kelvin --verbose

Run the simulation in serial mode with verbose output:

.. code-block:: none

   $ yank run --store=output --verbose

Analyze the results of the simulation:

.. code-block:: none

   $ yank analyze

