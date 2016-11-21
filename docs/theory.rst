.. _theory:

Theory
******

This section covers the theory of YANK. We focus on the thermodynamics here and do not delve into the implementation
specific details. For a more detailed description of the theory behind alchemical free energy calculations,
see `alchemistry.org <http://alchemistry.org>`_.

The Thermodynamic Cycle of YANK
===============================

We cover the full thermodynamic cycle that YANK follows to compute the free energy of binding.

.. figure:: docimages/YANKTC-Stage1.png

The fundamental process we want to capture.

The binding process is shown in this simplified diagram where the receptor (red blob), and the ligand (yellow circle),
bind together to create a complex in some solvent medium (blue box). This process gives us the free energy of binding,
|DG|. Lets look at each component of this diagram in more detail to better understand what
we are showing.

The right hand side shows the complexed ligand/receptor system bound together in a solvated box. The system should be
considered to be surrounded by an infinite medium, although we have drawn a physical box around the system for space.

The left hand side is two systems independent from each other. The receptor by itself with some infinite amount of
solvent is one system, and the ligand in a separate amount of solvent is the other system. These two systems do not
interact with one another, which emulates the effect of the receptor and ligand being in the same solvent, but separate
by an infinite distance. We note that all three systems are at the same temperature and (optionally) pressure.

Each side of the arrow makes up the one thermodynamic state per side. We frame the thermodynamic states by an encompassing
black box to show that even though the systems may be separated, they still contribute to the total thermodynamic state.
Each thermodynamic state.

Directly simulating a binding/unbinding process is nearly computationally impossible, so YANK instead uses a computationally
efficient thermodynamic cycle to compute |DG|. Thermodynamics allow this since free energy is a state function, meaning the
free energy will be identical no mater what path is taken. However, even though free energy does not change, the are only
making an estimate, so there will be error associated with our estimate, and that error **does** depend on the path.


.. figure:: docimages/YANKTC-Stage2.png

The thermodynamic cycle YANK follows over its simulations. Left side is the ``solvent`` phase of the simulation,
right is the ``complex`` phase of the simulation. Top is the fully interacting/coupled interactions, and bottom is the
noninteracting/decoupled interactions.

This thermodynamic cycle is what you the user see when running YANK. Before we go into each state and step of this
cycle, we cover the the new objects shown. The first thing is the dashed circle around the systems. This represents the
long-range nonbonded cutoff scheme that we use in simulations to make them more computationally efficient. We show the
cutoff as we will need to make corrections for this approximation later. Second is the ligand changing from yellow to
white. This indicates that the ligand has been decoupled from its surroundings. When decoupled, the ligand does not have
nonbonded interactions with the receptor, or the solvent. Finally, the spring objects indicate restraints acting on the
ligand to keep it close to the receptor.

We start with the ``complex`` phase of the simulation on the right side. We take the ligand bound to the receptor and
having nonbonded interactions with all its surroundings (top right), then decouple it from the receptor and the solvent
(bottom right). We also turn on restraints with the ligand in this step as indicated by the springs.

Next we look at the ``solvent`` phase of the simulation on the left side. At this point the receptor and ligand are now
in separate systems and not interacting. We have also turned off the restraints. Starting on bottom left, the ligand is
still not interacting with solvent. Effectively, the ligand feels like it is in a vacuum. The ``solvent`` leg then turns
on all the nonbonded interactions of the ligand in solvent by itself to complete the leg. We show the receptor in
solvent by itself for completeness, but there is no change in the receptor system of this leg, so :math:`\Delta G = 0`
of that system.

Lastly, we need to compute the free energy of transferring the restrained, decoupled ligand from the system with receptor
into its own system with just solvent. We handle this with the standard state correction of this transfer plus the
analytical contribution of the restraint. This connects the bottom two states, however, this does not complete the
thermodynamic cycle.

.. figure:: docimages/YANKTC-Stage3.png

We expand the cutoff radius to reduce the error introduced from having a reduced cutoff. Both the fully interacting
and noninteracting state are expanded to account for the errors on both ends of the thermodynamic cycle.

We correct for the error introduced by having a cutoff for long-range nonbonded interactions by expanding the cutoff
radius at either end of the thermodynamic cycle. We do this for both the solvent and complex phases. We also change which
step we compute the standard state correction from, although it does not change the actual value of it.

We look now at the receptor on the solvent leg (left side) of the process. We note that the expanded cutoff state at
both the top and bottom of the leg are identical, and thus the free energy difference between these states should be zero.
We could simulate the process of shrinking the cutoff, simulating the receptor and solvent with no changes, then expand the
cutoff again. However, we already know what the change in free energy is, and doing so will only add noise and error
in our estimation of the free energy, so we simply do not simulate this process and take the analytical result.

Finally, with the expanded cutoff, we assume that we have significantly reduced the error from having a smaller long-range
cutoff and that we are approximating the infinite solvent systems. There will be some error introduced because
the simulations are being run at the smaller cutoff, however, this error can be reduced by simulating with larger cutoffs
at the cost of computational efficiency.

We are now ready to complete the thermodynamic cycle, connecting the stages of YANK with our original binding process.

.. figure:: docimages/YANKTC-Complete.png

We now have a completed thermodynamic cycle for YANK which lets us estimate the free energy of binding.


.. Shorthand markers

.. |DG| replace::
    :math:`\Delta G_{\text{binding}}`