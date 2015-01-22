.. _benzene-toluene-explicit:

benzene-toluene in explicit solvent
===================================

This simple example demonstrates the computation of the association constant between benzene and toluene in explicit solvent.

.. code-block:: none

   $ cd examples/benzene-toluene-explicit

To set up the calculation, we designate benzene (residue name `BEN`) as the ligand.
No restraint is needed in the case of periodic simulations.
Reaction field electrostatics (`--nbmethod=CutoffPeriodic`) is used, and the temperature is set to 300 Kelvin, with pressure 1 atmosphere.

.. code-block:: none

   $ yank prepare binding amber --setupdir=setup --ligname=BEN --store=output --iterations=1000 --nbmethod=CutoffPeriodic --temperature=300*kelvin --pressure=1*atmospheres --verbose

Run the simulation in serial mode with verbose output:

.. code-block:: none

   $ yank run --store=output --verbose

Analyze the results of the simulation:

.. code-block:: none

   $ yank analyze

