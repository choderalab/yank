.. _algorithms:

**********
Algorithms
**********

We describe the various components of modern alchemical free energy calculations with YANK.

Solvent treatment
=================

YANK provides the capability for performing alchemical free energy calculations in both explicit solvent (where a water model such as TIP3P :cite:`Jorgensen1983` or TIP4P-Ew :cite:`Horn2004` is used to fill the simulation box with solvent) and implicit solvent (where a continuum representation of the solvent is used to reduce calculation times at the cost of some accuracy).
While explicit solvent free energy calculations are still considered the "gold standard" in terms of accuracy, implicit solvent free energy calculations may offer a very rapid way to compute approximate binding free energies while still incorporating some of the entropic and enthalpic contributions to binding.

Explicit solvent
----------------

Solvent model
^^^^^^^^^^^^^

Any explicit solvent model that can be constructed via AmberTools or that is distributed along with OpenMM is supported.

Common choices of explicit solvent model include

* **TIP3P** :cite:`Jorgensen1983` - an older but extremely popular three-site model used in many legacy calculations

* **TIP4P-Ew** :cite:`Horn2004` - a high-quality four-site model reparameterized for use with PME electrostatics.
   This water model is recommended for calculations that utilize PME.

.. todo:: What should we recommend for reaction field calculations?

Electrostatics treatment
^^^^^^^^^^^^^^^^^^^^^^^^

OpenMM supports several electrostatics models for the periodic simulation boxes used with explicit solvent calculations, all of which are accessible in YANK:

* ``PME`` - **Particle mesh Ewald (PME)** :cite:`Essmann1995,Toukmaji1996` is the "gold standard" for accurate long-range treatment of electrostatics in periodic solvated systems.
.. warning:: YANK currently has some difficulty with alchemical transformations involving PME because of the inability to represent the reciprocal-space contribution of the alchemically modified ligand, so phase space overlap with the endpoints can be poorer than with other methods.
.. todo:: Levi Naden has a trick we can use to fix this issue.

* ``CutoffPeriodic`` - **Reaction field electrostatics** :cite:`Tironi1995` is a faster, less accurate methods for treating electrostatics in solvated systems that assumes a uniform dielectric outside the nonbonded cutoff distance.

* ``Ewald`` - **Ewald electrostatics**, which is approximated by the much faster ``PME`` method.  It is not recommended that users employ this method for alchemical free energy calculations due to the speed of this method and availability of ``PME``.

Long-range dispersion corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analytical isotropic long-range dispersion correction
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Simulations in explicit solvent will by default add an **analytical isotropic long-range dispersion correction** to correct for the truncation of the nonbonded potential at the cutoff.
Without this correction, significant artifacts in solvent density and other physical properties can occur :cite:`Shirts2007`.

Anisotropic long-range dispersion correction
""""""""""""""""""""""""""""""""""""""""""""

Because this correction assumes that the solvent is isotropic outside of the nonbonded cutoff, however, significant errors in computed binding free energies are possible (up to several kcal/mol for absolute binding free energies of large ligands) if the diameter of the protein is larger than the nonbonded cutoff due to the significant difference in density between protein and solvent :cite:`Shirts2007`.

To correct for this, we utilize the **anisotropic long-range dispersion correction** described in Ref. :cite:`Shirts2007` in which the endpoints of each alchemical leg of the free energy calculation are perturbed to a system where the cutoffs are enlarged to a point where this error is neglible.
Because this contribution is only accumulated when configurations are written to disk, the additional computational overhead is small.
The largest allowable cutoff (slightly smaller than one-half the smallest box edge) is automatically selected for this purpose.

Restraint potential and standard state correction
=================================================

For **implicit solvent** calculations

Alchemical intermediates
========================

Alchemical protocol
===================

Hamiltonian exchange with Gibbs sampling
========================================

Markov chain Monte Carlo
========================

Metropolis Monte Carlo displacement and rotation moves
------------------------------------------------------

Generalized hybrid Monte Carlo
------------------------------

Automated equilibration detection
=================================

Analysis with MBAR
==================

Automated convergence detection
===============================

Will extract information from `here <http://nbviewer.ipython.org/github/choderalab/simulation-health-reports/blob/master/examples/yank/YANK%20analysis%20example.ipynb>`_.

Simulation health report
========================

Autotuning the alchemical protocol
==================================
