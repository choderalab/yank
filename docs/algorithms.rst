.. _algorithms:

**********
Algorithms
**********

We describe the various components of modern alchemical free energy calculations with YANK.

Solvent treatment
=================

YANK provides the capability for performing alchemical free energy calculations in both explicit solvent (where a water
model such as TIP3P :cite:`Jorgensen1983` or TIP4P-Ew :cite:`Horn2004` is used to fill the simulation box with solvent)
and implicit solvent (where a continuum representation of the solvent is used to reduce calculation times at the cost of
some accuracy).
While explicit solvent free energy calculations are still considered the "gold standard" in terms of accuracy, implicit
solvent free energy calculations may offer a very rapid way to compute approximate binding free energies while still
incorporating some of the entropic and enthalpic contributions to binding.

Implicit solvent
----------------

YANK provides the somewhat unique ability to perform alchemical free energy calculations in implicit solvent.

Solvent model
^^^^^^^^^^^^^

For the ``yank prepare amber`` command that imports AMBER ``prmtop`` files, any of the AMBER implicit solvent models
available in OpenMM are available for use via the :code:`--gbsa <model>`` argument:

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

In order to permit alchemical free energy calculations to be carried out in implicit solvent, the contribution of atoms
in the alchemically annihilated region must also be annihilated.
This is done by introducing a dependence on the alchemical parameter :math:`\lambda` into the GBSA potential terms.
Each particle *i* has an associated indicator function :math:`\eta_i` which assumes the value of 1 if the particle is
part of the alchemical region and 0 otherwise.

The modified Generalized Born contribution to the potential energy is given by :cite:`Onufriev2004`

.. math::
   U_{GB}(x; \lambda, \eta) = - \frac{1}{2} C \left(\frac{1}{\epsilon_{\mathit{solute}}}-\frac{1}{\epsilon_{\mathit{solvent}}}\right)\sum _{i,j}\frac{ s_{ij}(\lambda,\eta) \, {q}_{i} {q}_{j}}{{f}_{\text{GB}}\left(||x_i - x_j||_2,{R}_{i},{R}_{j};\lambda, \eta\right)}

where the indices *i* and *j* run over all particles, :math:`\epsilon_\mathit{solute}` and :math:`\epsilon_\mathit{solvent}`
are the dielectric constants of the solute and solvent respectively, :math:`q_i` is the charge of particle *i*\ ,
and :math:`||x_i - x_j||_2` is the distance between particles *i* and *j*.
The electrostatic constant :math:`C` is equal to 138.935485 nm kJ/mol/e\ :sup:`2`\ .

.. warning:: Add alchemical self-energy terms.

The alchemical attenuation function :math:`s_{ij}(\lambda, \eta)` attenuates interactions involving softcore atoms, and is given by

.. math::
   s_i(\lambda,\eta) &= \lambda \eta_i + (1-\eta_i) \\
   s_{ij}(\lambda,\eta) &= s_i(\lambda,\eta) \cdot s_j(\lambda,\eta)

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
   U_{SA}(x;\lambda) = \epsilon_{SA} \cdot 4\pi erically
   \sum_{i} s_i(\lambda,\eta) {\left({r}_{i}+{r}_{\mathit{solvent}}\right)}^{2}{\left(\frac{{r}_{i}}{{R}_{i}}\right)}^{6}

where :math:`\epsilon_{SA}` is the surface area energy penalty, :math:`r_i` is the atomic radius of particle *i*\ ,
:math:`r_i` is its atomic radius, and :math:`r_\mathit{solvent}` is the solvent radius, which is taken to be 0.14 nm.
The default value for the surface area penalty :math:`\epsilon_{SA}` is 2.25936 kJ/mol/nm\ :sup:`2`\ .

.. warning:: Add description of other GBSA forms.

Explicit solvent
----------------

Solvent model
^^^^^^^^^^^^^

Any explicit solvent model that can be constructed via AmberTools or that is distributed along with OpenMM is supported.

For the ``yank prepare amber`` command that imports AMBER ``prmtop`` files, any solvent model specified in the ``prmtop`` file is used automatically.
This method is depreciated however in favor of the :ref:`YAML method of setting up systems <yaml_head>`.

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
    .. todo:: Levi Naden has a trick we can use to fix this issue.

* ``CutoffPeriodic`` - **Reaction field electrostatics** :cite:`Tironi1995` is a faster, less accurate methods for treating electrostatics in solvated systems that assumes a uniform dielectric outside the nonbonded cutoff distance.
    .. warning:: |EwaldWarn|

.. |EwaldWarn| replace::
    YANK currently has some difficulty with alchemical transformations involving Reaction Field because of the inability
    to represent the long range contribution of the alchemically modified ligand over all alchemical states, so phase
    space overlap with the endpoints can be poorer than with other methods.

* ``Ewald`` - **Ewald electrostatics**, which is approximated by the much faster ``PME`` method.  It is not recommended that users employ this method for alchemical free energy calculations due to the speed of this method and availability of ``PME``.

Long-range dispersion corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analytical isotropic long-range dispersion correction
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Simulations in explicit solvent will by default add an **analytical isotropic long-range dispersion correction** to
correct for the truncation of the nonbonded potential at the cutoff.
Without this correction, significant artifacts in solvent density and other physical properties can occur :cite:`Shirts2007`.

Anisotropic long-range dispersion correction
""""""""""""""""""""""""""""""""""""""""""""

Because this correction assumes that the solvent is isotropic outside of the nonbonded cutoff, however, significant
errors in computed binding free energies are possible (up to several kcal/mol for absolute binding free energies of large
ligands) if the diameter of the protein is larger than the nonbonded cutoff due to the significant difference in density
between protein and solvent :cite:`Shirts2007`.

To correct for this, we utilize the **anisotropic long-range dispersion correction** described in Ref. :cite:`Shirts2007`
in which the endpoints of each alchemical leg of the free energy calculation are perturbed to a system where the cutoffs
are enlarged to a point where this error is negligible.
Because this contribution is only accumulated when configurations are written to disk, the additional computational
overhead is small.
The largest allowable cutoff (slightly smaller than one-half the smallest box edge) is automatically selected for this
purpose, but the user must specify a large enough box that the cutoff can be expanded to at least 16 Angstroms.

Restraints and standard state correction
========================================

Restraints between receptor and ligand are used for two purposes:

* **Defining the bound species**: The theoretical framework for alchemical free energy calculations requires that the bound receptor-ligand complex be defined in some way.
  While this can be done by an indicator function that assumes the value of unity of the receptor and ligand are bound otherwise, it is difficult to restrict the bound complex integral to this region within the context of a molecular dynamics simulation.
  Instead, a fuzzy indicator function that can assume continuous values is equivalent to imposing a restraint that restricts the ligand to be near the receptor to define the bound complex and restrict the configuration integral.

* **Reducing accessible ligand conformations during intermediate alchemical states**: Another function of restraints is to restrict the region of conformation space that must be integrated in the majority of the alchemical states, speeding convergence.
  For example, orientational restraints greatly restrict the number of orientations or binding modes the ligand must visit during intermediate alchemical states, greatly accelerating convergence.
  On the other hand, if multiple orientations are relevant but cannot be sampled during the imposition of additional restraints, this can cause the resulting free energy estimate to be heavily biased.

In principle, both types of restraints would be used in tandem: One restraint would define the bound complex, while another restraint would be turned to reduce the amount of sampling required to evaluate alchemical free energy differences.
In the current version of YANK, only one restraint can be used at a time.
More guidance is given for each restraint type below.

Standard state correction
-------------------------

Since the restraint defines the bound complex, in order to report a standard state binding free energy, we must compute the free energy of releasing the restraint into a volume ``V0`` representing the *standard state volume* to achieve a standard state concentration of 1 Molar.
More detail of how this free energy fits into the thermodynamic cycle can be found in `theory <theory.html>`_.

Restraint types
---------------

``YANK`` currently supports several kinds of receptor-ligand restraints.

No restraints (``null``)
^^^^^^^^^^^^^^^^^^^^^^^^

While it is possible to run a simulation without a restraint in explicit solvent---such that the noninteracting ligand must explore the entire simulation box---this is not possible in implicit solvent since the ligand can drift away into infinite space.
Note that this is not recommended for explicit solvent, since there is a significant entropy bottleneck that must be overcome for the ligand to discover the binding site from the search space of the entire box.

Spherically symmetric restraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Harmonic resstraints (``Harmonic``)
"""""""""""""""""""""""""""""""""""

A harmonic potential is imposed between the closest atom to the center of the receptor and the closest atom to the center of the ligand, given the initial geometry.
The equilibrium distance is zero, while the spring constant is selected such the potential reaches ``kT`` at one radius of gyration.
This allows the ligand to explore multiple binding sites---including internal sites---without drifting away from the receptor.
For implicit and explicit solvent calculations, harmonic restraints should be imposed at full strength and retained
throughout all alchemical states to define the bound complex.
Since the harmonic restraint is significant at the periphery of the receptor, it can lead to bias in estimates of binding affinities on the surface of receptors.

The standard-state correction is computed via numerical quadrature.

.. note:: |ExplicitNote|

.. |ExplicitNote| replace::
    For **explicit** solvent in the fully coupled (non-alchemical) state, these restraints
    should be off to represent the physically bound system. A series of alchemical intermediates should be added to smoothly
    bring these restraints to full before academically modifying any other interactions. In implicit solvent, the restraint
    should always be fully coupled.

Flat-bottom restraints (``FlatBottom``)
"""""""""""""""""""""""""""""""""""""""

A variant of ``Harmonic`` where the restraint potential is zero in the central region and grows as a half-harmonic potential outside of this region.
A lengthscale ``sigma`` is computed from the median absolute distance from the central receptor atom to all atoms, multiplied by 1.4826.
The transition from flat to harmonic occurs at ``r0 = 2*sigma + 5*angstroms``.
A spring constant of ``K = 0.6 * kilocalories_per_mole / angstroms**2`` is used.
This restraint is described in detail in :cite:`Shirts2013:yank`.
For implicit and explicit solvent calculations, flat-bottom restraints should be imposed at full strength and retained
throughout all alchemical states to define the bound complex.

The standard-state correction is computed via numerical quadrature.

Orientational restraints
^^^^^^^^^^^^^^^^^^^^^^^^

Orientational restraints are used to confine the ligand to a single binding pose.

.. warning:: Because the ligand is highly restrained orientationally, the initial configuration should have the ligand well-placed in the binding site; errors in initial pose cannot be easily recovered from.

Boresch restraints (``Boresch``)
""""""""""""""""""""""""""""""""

A common type of **orientational restraints** between receptor and ligand :cite:`Boresch2003`.
These restrain a distance, two angles, and three torsions in an attempt to keep the ligand in a specific relative binding pose.
Default spring constants used in Table 1 of the original paper :cite:`Boresch2003` are used, and a set of atoms is
automatically chosen (three each in the ligand and receptor) to ensure that the distance (:math:`r_{aA0}`) is within [1,4]
Angstroms and the angles (:math:`\theta_{A0}`, :math:`\theta_{B0}`) are several standard deviations away from 0 and
:math:`\pi`.

Standard use of Boresch restraints is to turn on the restraints over several alchemical states and keep the restraints
active while discharging followed by Lennard-Jones decoupling.
This assumes the ligand is already effectively confined to its bound state even when the restraint is off such that
imposing the restraint measures the free energy of additionally confining the *bound* ligand; if this is not the case,
it could lead to problematic free energy estimates.

The standard state correction is computed by evaluating using a combination of numerical and analytical
one-dimensional integrals from Eq. 12 of :cite:`Boresch2003`.
Note that the analytical standard state correction described in Eq. 32 of :cite:`Boresch2003` is inaccurate
(up to several ``kT``) in certain regimes (near :math:`r_{aA0}` and :math:`\theta_{A0}`, :math:`\theta_{B0}` near
0 or :math:`\pi`) and should be avoided.

.. warning:: Symmetry corrections for symmetric ligands are **not** automatically applied; see Ref :cite:`Boresch2003` and :cite:`Mobley2006:orientational-restraints` for more information on correcting for ligand symmetry.

Adding new restraints
---------------------

``YANK`` also makes it easy to add new types of restraints by subclassing the ``yank.restraints.ReceptorLigandRestraint`` class.
Simply subclassing this class (an abstract base class) and implementing the following methods will allow this restraint type to be specified via its classname.

* ``__init__(self, topology, state, system, positions, receptor_atoms, ligand_atoms):``

* ``get_restraint_force(self):``

* ``get_standard_state_correction(self):``

Alchemical intermediates
========================

Alchemical intermediate states are chosen to provide overlap along a one dimensional thermodynamic path for each phase
of the simulation, ultimately making a :ref:`complete thermodynamic cycle of binding/solvation <yank_cycle>`. At this time,
there is no automated way of selecting alchemical intermediates and it is up to the user to define them at this time,
although automatic selection of alchemical intermediates is planned for a future release.

Alchemical protocol
===================

A number of rules of thumb are followed when choosing what order to carry out alchemical intermediates,
examples of which can be found in the
:doc:`Examples Documentation <examples/index>`.

#. Couple restraints before any other alchemical changes
#. Decouple electrostatics first and separate from all other alchemical changes=
#. Decouple Lennard-Jones interactions last.
#. Always define the "fully coupled" state as the first index, 0 in Python, in the list of alchemical states (the state most closely representing the physically bound/fully solvated conditions)
#. Always define the "fully decoupled/annihilated" state as the last index, -1 in Python, in the list of states.

Hamiltonian exchange with Gibbs sampling
========================================

The theory for this section is taken from a single source and summarized here :cite:`Chodera2011`

Hamiltonian Replica Exchange (HREX) is carried out to improve sampling between different alchemical states. In the basic version
of this scheme,
a proposed swap of configurations between two alchemical states, *i* and *j*, made by comparing the energy of each
configuration in each replica and swapping with a basic Metropolis criteria of

.. math::
    P_{\text{accept}}(i, x_i, j, x_j) &= \text{min}\begin{cases}
                               1, \frac{ e^{-\left[u_i(x_j) + u_j(x_i)\right]}}{e^{-\left[u_i(x_i) + u_j(x_j)\right]}}
                               \end{cases} \\
        &= \text{min}\begin{cases}
          1, \exp\left[\Delta u_{ji}(x_i) + \Delta u_{ij}(x_j)\right]
          \end{cases}

where :math:`x` is the configuration of the subscripted states :math:`i` or :math:`j`, and :math:`u` is the reduced
potential energy. We have added the second equality to improve readability.
This scheme is typically carried out on neighboring states only.

YANK's HREX scheme instead samples with Gibbs Sampling by attempting swaps between ALL :math:`K` states simultaneously. However,
instead of trying to directly sample the unnormalized probability distribution across all states and configurations, then
performing an all-state-to-all-state swap, YANK draws from an approximate distribution by attempting :math:`K^5` swaps
between uniformly chosen pairs of states. The acceptance criteria for each swap is the same as above, but you can
show with this state selection scheme and number of swap attempts, that you will effectively draw from the correct
distribution without too much computational overhead :cite:`Chodera2011`.
This speeds up mixing and reduces the total number of samples you need to get uncorrelated samples.

Markov chain Monte Carlo
========================

Metropolis Monte Carlo displacement and rotation moves
------------------------------------------------------

Generalized hybrid Monte Carlo
------------------------------

Automated equilibration detection
=================================

Will extract information from `here <http://nbviewer.ipython.org/github/choderalab/simulation-health-reports/blob/master/examples/yank/YANK%20analysis%20example.ipynb>`_.

Analysis with MBAR
==================

Automated convergence detection
===============================

Will extract information from `here <http://nbviewer.ipython.org/github/choderalab/simulation-health-reports/blob/master/examples/yank/YANK%20analysis%20example.ipynb>`_.

Simulation health report
========================

Autotuning the alchemical protocol
==================================
