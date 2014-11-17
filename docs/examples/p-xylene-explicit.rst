.. _p-xylene-explicit:

p-xylene binding to T4 lysozyme L99A in explicit solvent
========================================================

This example illustrates the computation of the binding free energy of p-xylene to the T4 lysozyme L99A mutant.

.. code-block:: none

   $ cd examples/p-xylene-explicit

To set up the calculation, we designate p-xylene (residue name `MOL`) as the ligand.
No restraint is needed in the case of periodic simulations.
Reaction field electrostatics (`--nbmethod=CutoffPeriodic`) is used, and the temperature is set to 300 Kelvin, with pressure 1 atmosphere.

.. code-block:: none

   $ yank setup binding amber --setupdir=setup --ligname=MOL --store=output --iterations=1000 --nbmethod=CutoffPeriodic --temperature=300*kelvin --pressure=1*atmospheres --verbose

Run the simulation in serial mode with verbose output:

.. code-block:: none

   $ yank run --store=output --verbose

Analyze the results of the simulation:

.. code-block:: none

   $ yank analyze


