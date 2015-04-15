.. _algorithms:

**********
Algorithms
**********

We describe the various components of modern alchemical free energy calculations with YANK.

Solvent treatment
=================

YANK provides the capability for performing alchemical free energy calculations in both explicit solvent (where a water model such as TIP3P :cite:`Jorgensen1983` or TIP4P-Ew :cite:`Horn2004` is used to fill the simulation box with solvent) and implicit solvent (where a continuum representation of the solvent is used to reduce calculation times at the cost of some accuracy).
While explicit solvent free energy calculations are still considered the "gold standard" in terms of accuracy, implicit solvent free energy calculations may offer a very rapid way to compute approximate binding free energies while still incorporating some of the entropic and enthalpic contributions to binding.

Implicit solvent
----------------

YANK provides the somewhat unique ability to perform alchemical free energy calculations in implicit solvent.

Solvent model
^^^^^^^^^^^^^

For the ``yank prepare amber`` command that imports AMBER ``prmtop`` files, any of the AMBER implicit solvent models available in OpenMM are available for use via the :code:`--gbsa <model>`` argument:

.. tabularcolumns:: |l|L|

=============  ==================================================================================================================================
Value          Meaning
=============  ==================================================================================================================================
:code:`None`   No implicit solvent is used.
:code:`HCT`    Hawkins-Cramer-Truhlar GBSA model\ :cite:`Hawkins1995` (corresponds to igb=1 in AMBER)
:code:`OBC1`   Onufriev-Bashford-Case GBSA model\ :cite:`Onufriev2004` using the GB\ :sup:`OBC`\ I parameters (corresponds to igb=2 in AMBER).
:code:`OBC2`   Onufriev-Bashford-Case GBSA model\ :cite:`Onufriev2004` using the GB\ :sup:`OBC`\ II parameters (corresponds to igb=5 in AMBER).
               This is the same model used by the GBSA-OBC files described in Section :ref:`force-fields`.
:code:`GBn`    GBn solvation model\ :cite:`Mongan2007` (corresponds to igb=7 in AMBER).
:code:`GBn2`   GBn2 solvation model\ :cite:`Nguyen2013` (corresponds to igb=8 in AMBER).
=============  ==================================================================================================================================

Alchemical intermediates
^^^^^^^^^^^^^^^^^^^^^^^^

.. warning:: This section is still undergoing revision for accuracy.

In order to permit alchemical free energy calculations to be carried out in implicit solvent, the contribution of atoms in the alchemically annihilated region must also be annihilated.
This is done by introducing a dependence on the alchemical parameter :math:`\lambda` into the GBSA potential terms.
Each particle *i* has an associated indicator function :math:`\eta_i` which assumes the value of 1 if the particle is part of the alchemical region and 0 otherwise.

The modified Generalized Born contribution to the potential energy is given by :cite:`Onufriev2004`

.. math::
   U_{GB}(x; \lambda, \eta) = - \frac{1}{2} C \left(\frac{1}{\epsilon_{\mathit{solute}}}-\frac{1}{\epsilon_{\mathit{solvent}}}\right)\sum _{i,j}\frac{ s_{ij}(\lambda,\eta) \, {q}_{i} {q}_{j}}{{f}_{\text{GB}}\left(||x_i - x_j||_2,{R}_{i},{R}_{j};\lambda, \eta\right)}

where the indices *i* and *j* run over all particles, :math:`\epsilon_\mathit{solute}` and :math:`\epsilon_\mathit{solvent}` are the dielectric constants of the solute and solvent respectively, :math:`q_i` is the charge of particle *i*\ , and :math:`||x_i - x_j||_2` is the distance between particles *i* and *j*.
The electrostatic constant :math:`C` is equal to 138.935485 nm kJ/mol/e\ :sup:`2`\ .
The alchemical attenuation function :math:`s_{ij}(\lambda, \eta)` attenuates interactions involving softcore atoms, and is given by

.. math::
   s_{ij}(\lambda,\eta) = [ \lambda \eta_i + (1-\eta_i) ] \cdot [ \lambda \eta_j + (1-\eta_j) ]

The alchemically-modified GB effective interaction distance function :math:`f_\text{GB}(d_{ij}, R_i, R_j; \lambda, \eta)`, which has units of distance, is defined as

.. math::
   {f}_{\text{GB}}\left({d}_{ij},{R}_{i},{R}_{j};\lambda\right)={\left[{d}_{ij}^2+{R}_{i}{R}_{j}\text{exp}\left(\frac{-{d}_{ij}^2}{{4R}_{i}{R}_{j}}\right)\right]}^{1/2}

:math:`R_i` is the Born radius of particle *i*\ , which calculated as

.. math::
   R_i=\frac{1}{\rho_i^{-1}-r_i^{-1}\text{tanh}\left(\alpha \Psi_{i}-{\beta \Psi}_i^2+{\gamma \Psi}_i^3\right)}

where :math:`\alpha`, :math:`\beta`, and :math:`\gamma` are the GB\ :sup:`OBC`\ II parameters :math:`\alpha` = 1, :math:`\beta` = 0.8, and :math:`\gamma` =
4.85.  :math:`\rho_i` is the adjusted atomic radius of particle *i*\ , which
is calculated from the atomic radius :math:`r_i` as :math:`\rho_i = r_i-0.009` nm.
:math:`\Psi_i` is calculated as an integral over the van der Waals
spheres of all particles outside particle *i*\ :

.. warning:: This integral needs to be rewritten in terms of a sum over atoms *j* with the alchemical modification `math`:s_j(\lambda,\eta)` inserted.

.. math::
   \Psi_i=\frac{\rho_i}{4\pi}\int_{\text{VDW}}\theta\left(|\mathbf{r}|-{\rho }_{i}\right)\frac{1}{{|\mathbf{r}|}^{4}}{d}^{3}\mathbf{r}

where :math:`\theta`\ (\ *r*\ ) is a step function that excludes the interior of particle
\ *i* from the integral.

The alchemically-modified surface area potential term is a modified form of the term given by :cite:`Schaefer1998`\ :cite:`Ponder`

.. math::
   U_{SA}(x;\lambda) = \epsilon_{SA} \cdot 4\pi \sum_{i} \left[\lambda \eta_i + (1-\eta_i)\right] {\left({r}_{i}+{r}_{\mathit{solvent}}\right)}^{2}{\left(\frac{{r}_{i}}{{R}_{i}}\right)}^{6}

where :math:`\epsilon_{SA}` is the surface area energy penalty, :math:`r_i` is the atomic radius of particle *i*\ , :math:`r_i` is its atomic radius, and :math:`r_\mathit{solvent}` is the solvent radius, which is taken to be 0.14 nm.
The default value for the surface area penalty :math:`\epsilon_{SA}` is 2.25936 kJ/mol/nm\ :sup:`2`\ .

Explicit solvent
----------------

Solvent model
^^^^^^^^^^^^^

Any explicit solvent model that can be constructed via AmberTools or that is distributed along with OpenMM is supported.

For the ``yank prepare amber`` command that imports AMBER ``prmtop`` files, any solvent model specified in the ``prmtop`` file is used automatically.

For systems prepared with ``yank prepare systembuilder``, any solvent models available in OpenMM can be specified via the ``--solventmodel <model>`` argument.  Water models available in OpenMM include:

.. tabularcolumns:: |l|L|

===================  ============================================
Model                Water Model
===================  ============================================
:code:`tip3p`        TIP3P water model :cite:`Jorgensen1983` (older model used in many legacy calculations)
:code:`tip4pew`      TIP4P-Ew water model :cite:`Horn2004` (recommended)
:code:`tip3pfb`      TIP3P-FB water model :cite:`Wang2014`
:code:`tip4pfb`      TIP4P-FB water model :cite:`Wang2014`
:code:`tip5p`        TIP5P water model :cite:`Mahoney2000`
:code:`spce`         SPC/E water model :cite:`Berendsen1987`
:code:`swm4ndp`      SWM4-NDP water model :cite:`Lamoureux2006`
===================  ============================================

.. todo:: What should we recommend for reaction field calculations?  Is there a ForceBalance-parameterized version for use with reaction field?

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
