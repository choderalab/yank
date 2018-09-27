.. _tutorial:

YAML syntax tutorial
********************

This tutorial assumes you've :ref:`installed <installation>` YANK successfully.

We'll start by creating a YAML file for a hydration free energy calculation in explicit solvent, and then we'll add an
absolute binding free energy experiment in explicit and implicit solvent.

.. contents::

A first example: Hydration free energy
======================================

We'll go line-by-line through the sections of the following YAML file for YANK that sets up and run a hydration binding
free energy of a small molecule. The three dashes (``---``) is the standard way in YAML to indicate the beginning of a
YAML document.

.. code-block:: yaml

    ---

    options:
      verbose: yes
      minimize: yes
      default_timestep: 2.0*femtoseconds
      default_nsteps_per_iteration: 500
      temperature: 300*kelvin
      pressure: 1*atmosphere

    molecules:
      benzene:
        filepath: benzene.tripos.mol2
        antechamber:
          charge_method: bcc
        leap:
          parameters: [leaprc.gaff]

    solvents:
      water:
        nonbonded_method: PME
        nonbonded_cutoff: 9*angstroms
        clearance: 16*angstroms
        solvent_model: tip4pew
        leap:
          parameters: [leaprc.water.tip4pew]
      vacuum:
        nonbonded_method: NoCutoff

    systems:
      hydration-system:
        solute: benzene
        solvent1: water
        solvent2: vacuum
        leap:
          parameters: [leaprc.protein.ff14SB]

    protocols:
      hydration-protocol:
        solvent1:
          alchemical_path: auto
        solvent2:
          alchemical_path: auto

    experiments:
      system: hydration-system
      protocol: hydration-protocol


Prepare the molecules
"""""""""""""""""""""

Let's skip the ``options`` section for now and look at the ``molecules`` block (:ref:`docs <yaml_molecules_head>`). This
is the place where you can specify the pipeline that a molecule should go through before eventually being solvated or
combined with another molecule to form a complex.

.. code-block:: yaml

    molecules:
      benzene:
        filepath: benzene.tripos.mol2
        antechamber:
          charge_method: bcc
        leap:
          parameters: [leaprc.gaff]

The ``benzene`` keyword here is a string identifier so it can be any name we want. The nested ``filepath`` keyword points
to a ``mol2`` file specifying the molecule topology and initial coordinates of the atoms. The automatic pipeline in YANK
is based on `AmberTools <http://ambermd.org/AmberTools.php>`_ so you can determine charges with antechamber (in this case
AM1-BCC point charges) and assign the parameters to the molecule by loading any parameter file accessible to tleap.

The tleap parameters can be assigned in the ``systems`` section as well with identical effects, but when you have multiple
molecules and you want to assign different parameters to them, it is convenient to be able to specify the parameter file
to load in the ``molecules`` section.

Specify the solvent
"""""""""""""""""""

We'll turn off benzene's nonbonded interactions in explicit water and turn them back on in vacuum to close the
thermodynamic cycle so let's create two solvents.

.. code-block:: yaml

    solvents:
      water:
        nonbonded_method: PME
        nonbonded_cutoff: 9*angstroms
        clearance: 16*angstroms
        solvent_model: tip4pew
        leap:
          parameters: [leaprc.water.tip4pew]
      vacuum:
        nonbonded_method: NoCutoff

Again, ``water`` and ``vacuum` are just two arbitrary string identifiers. The ``vacuum`` solvent is quite simple. The
``water`` solvent we use Particle Mesh Ewald for long-range electrostatic treatment with a cutoff of 9A for both Coulomb
and Lennard-Jones interactions.

The ``clearance`` and ``solvent_model`` keywords instruct YANK to solvate the solute with TIP4P-Ew water molecules so that
the solute will be distant from the box boundaries by 16A. Finally, we ask leap to load the TIP4P-Ew parameters when solvating.
Again, this parameter file can be specified in the in ``systems`` section.

Prepare the systems
"""""""""""""""""""

Finally let's put everything together to create the systems in solvent and in vacuum that YANK will use to run the
hydration free energy calculation.

.. code-block:: yaml

    systems:
      hydration-system:
        solute: benzene
        solvent1: water
        solvent2: vacuum
        leap:
          parameters: [leaprc.protein.ff14SB]

``hydration-system`` is, as usual, an arbitrary identifier. Here we are instructing YANK to create two systems by
solvating ``benzene`` in ``solvent1`` (i.e. the ``water`` specified above) and in vacuum. YANK will use these two systems
to compute the free energy of transferring benzene from ``solvent1`` to ``solvent2``. Finally, we load the general amber
parameters that are required by tleap.

If we don't specify the parameters in the ``solvents`` and ``molecules`` sections, we can do it here.

.. code-block:: yaml

    systems:
      hydration-system:
        ...
        leap:
          parameters: [leaprc.protein.ff14SB, leaprc.water.tip4pew, leaprc.gaff]

Alchemical protocol
"""""""""""""""""""

The ``protocols`` section determines the intermediate states to sample during both phases of the the alchemical calculations.
YANK provides a method to determine this automatically by spacing the intermediate states equally in thermodynamic length.
We'll see later in the tutorial how you can specify an alchemical path manually instead.

.. code-block:: yaml

    protocols:
      hydration-protocol:
        solvent1:
          alchemical_path: auto
        solvent2:
          alchemical_path: auto

Experiment
""""""""""

This is where we associate a protocol to the system and tell YANK to run the hydration free energy calculation.

.. code-block:: yaml

    experiments:
      system: hydration-system
      protocol: hydration-protocol

Running options
"""""""""""""""

Finally, the ``options`` section contains some several parameters for the simulation (see :ref:`here <yaml-options-head>`
for a complete list).

.. code-block:: yaml

    options:
      verbose: yes
      minimize: yes
      default_timestep: 2.0*femtoseconds
      default_nsteps_per_iteration: 500
      default_number_of_iterations: 1000
      temperature: 300*kelvin
      pressure: 1*atmosphere

Here we're setting the verbosity of the output, and asking YANK to minimize the systems before running 1000 iterations
of Hamiltonian replica exchange. Each iteration, by default, perform a Monte Carlo rigid translation and rotation of the
ligand followed, in this case, by 1ps of Langevin dynamics (with 2fs time step) keeping the temperature at 300K. The
pressure is controlled by a Monte Carlo barostat.

Since it is possible to specify multiple free energy calculations in a single YAML file, this section is used to keep
the options to use as default, but they can be overwritten by other parts of the YAML document as well. For example, the
``options`` block can be added in the experiment specification to overwrite some or all of the default values.

.. code-block:: yaml

    experiments:
      system: hydration-system
      protocol: hydration-protocol
      options:
        temperature: 310*kelvin
        default_nsteps_per_iteration: 1000


Add an absolute binding free energy experiment
==============================================

We mentioned that it's possible to specify multiple experiments in a single YAML file. Let's add an absolute binding free
energy calculation.

Preparing protein receptors
"""""""""""""""""""""""""""

We start as before by preparing the receptor.

.. code-block:: yaml

    molecules:
      ...
      t4-lysozyme:
        filepath: input/t4.pdb

In this case, we don't specify leap parameters since we'll add them to the system.

.. note::

    Since the PDB file goes through tleap, your PDB file should adopt the residue names consistent with their protonation state as expected by tleap (e.g., CYS for protonated Cysteine, CYM for deprotonated, and CYX for cysteines forming a disulphide bridge).


Add buffer and neutralizing ions
""""""""""""""""""""""""""""""""

In this case, we want to simulate a particular buffer ionic strength so we create a new solvent.

.. code-block:: yaml

    solvents:
      ...
      buffer:
        nonbonded_method: PME
        nonbonded_cutoff: 11*angstroms
        clearance: 12*angstroms
        positive_ion: Na+
        negative_ion: Cl-
        ionic_strength: 100*millimolar
        solvent_model: tip3p
        leap:
          parameters: [leaprc.water.tip3p]

By adding the three new keywords ``positive_ion``, ``negative_ion``, and ``ionic_strength``, YANK will add neutralizing
and buffer Na+ and Cl- ions.

.. note::

    If the molecule that is alchemically decoupled has a net charge, YANK will automatically select a random counterion to decouple together with the ligand.

Create receptor ligand system
"""""""""""""""""""""""""""""

Now we create the system by combining the ``t4-lysozyme`` and ``benzene`` to form a complex.

.. code-block:: yaml

    systems:
      ...
      t4-benzene:
        receptor: t4-lysozyme
        ligand: benzene
        solvent: buffer
        leap:
          parameters: [leaprc.protein.ff14SB, leaprc.gaff2, leaprc.water.tip4pew]

This time instead of ``solute``, ``solvent1``, and ``solvent2`` we have ``receptor``, ``ligand``, and ``solvent``. An
experiment using this system will decouple the ``ligand`` in complex and turn back on the interaction in bulk.

Restraints
""""""""""

Absolute binding calculations often require a restraint to keep the ligand in the binding pocket.

.. code-block:: yaml

    protocol:
      ...
      absolute-binding:
        complex:
          alchemical_path:
            lambda_electrostatics: [1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
            lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]
            lambda_restraints:     [0.00, 0.25, 0.50, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
        solvent:
          alchemical_path: auto

    experiment-t4-benzene:
      system: t4-benzene
      protocol: absolute-binding
      restraint:
        type: Harmonic

    experiment-hydration:
      ...

    experiments: [experiment-t4-benzene, experiment-hydration]

There are three main changes here. First, notice how the ``experiments`` section now points to a list of experiments.
This is necessary since we are using the same YAML file to run multiple experiments. Secondly, the ``experiment-t4-benzene``
block contains a ``restraint.type`` keyword that we can be used to specify the type of restraint we want to use. Other available
restraints are ``FlatBottom``, ``Boresch``, and ``PeriodicTorsionBoresch``.

Finally, we've used a manual protocol instead of ``auto`` to control the intermediate states of the Hamiltonian replica
exchange simulation of the ``complex`` phase. The protocol here first turns on the harmonic restraint (``lambda_restraints``
goes from 0.0 to 1.0), then turn off charges (``lambda_electrostatics`` from 1.0 to 0.0), and finally the Lennard-Jones
interactions are decoupled (``lambda_sterics``).

YANK uses a heuristic to find the parameters of the restraint, but it is possible to specify them manually.

.. code-block:: yaml

    experiment-t4-benzene:
      system: t4-benzene
      protocol: absolute-binding
      restraint:
        type: Harmonic
        spring_constant: 0.2*kilocalories_per_mole/(angstrom**2)
        restrained_receptor_atoms: (resi 77 or resi 90 or resi 98 or resi 110) and (mass > 1.5)
        restrained_ligand_atoms: (resname MOL) and (mass > 1.5)

This example manually set the spring constant, and attach the harmonic restraint to the centroid of the heavy atoms
identified by the `MDTraj DSL <http://mdtraj.org/latest/atom_selection.html>`_ selection expression. In general, ``type``
is the name of a class, and the keywords that is possible to specify to configure the restraints are its constructor
parameters. Check the `Python API documentation <http://getyank.org/0.23.4/api/restraints_api.html#yank.restraints.Harmonic>`_
for a complete overview.

.. rst-class:: html-toggle

The full script after the changes
"""""""""""""""""""""""""""""""""

.. code-block:: yaml

    ---

    options:
      verbose: yes
      minimize: yes
      default_timestep: 2.0*femtoseconds
      default_nsteps_per_iteration: 500
      temperature: 300*kelvin
      pressure: 1*atmosphere

    molecules:
      benzene:
        filepath: benzene.tripos.mol2
        antechamber:
          charge_method: bcc
        leap:
          parameters: [leaprc.gaff]
      t4-lysozyme:
        filepath: input/t4.pdb

    solvents:
      water:
        nonbonded_method: PME
        nonbonded_cutoff: 9*angstroms
        clearance: 16*angstroms
        solvent_model: tip4pew
        leap:
          parameters: [leaprc.water.tip4pew]
      vacuum:
        nonbonded_method: NoCutoff
      buffer:
        nonbonded_method: PME
        nonbonded_cutoff: 11*angstroms
        clearance: 12*angstroms
        positive_ion: Na+
        negative_ion: Cl-
        ionic_strength: 100*millimolar
        solvent_model: tip3p
        leap:
          parameters: [leaprc.water.tip3p]

    systems:
      hydration-system:
        solute: benzene
        solvent1: water
        solvent2: vacuum
        leap:
          parameters: [leaprc.protein.ff14SB]
      t4-benzene:
        receptor: t4-lysozyme
        ligand: benzene
        solvent: buffer
        leap:
          parameters: [leaprc.protein.ff14SB, leaprc.water.tip4pew]

    protocols:
      hydration-protocol:
        solvent1:
          alchemical_path: auto
        solvent2:
          alchemical_path: auto
      absolute-binding:
        complex:
          alchemical_path:
            lambda_electrostatics: [1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
            lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]
            lambda_restraints:     [0.00, 0.25, 0.50, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
        solvent:
          alchemical_path: auto

    experiment-t4-benzene:
      system: t4-benzene
      protocol: absolute-binding
      restraint:
        type: Harmonic

    experiment-hydration:
      system: hydration-system
      protocol: hydration-protocol

    experiments: [experiment-t4-benzene, experiment-hydration]


Combinatorial experiments: implicit and explicit solvents
=========================================================

We can use the ``!Combinatorial`` keyword in many sections to generate multiple experiments from a single YAML file while
maintaining the representation of our computational experiment compact. For example, let's assume we want to compare the
effect of multiple restraint types and the accuracy of the binding free energy estimate changes with TIP3P and the GBSA
implicit solvent model.

.. code-block:: yaml

    solvents:
      ...
      implicit:
        nonbonded_method: NoCutoff
        implicit_solvent: OBC2
        implicit_solvent_salt_conc: 100*millimolar

    systems:
      ...
      t4-benzene:
        receptor: t4-lysozyme
        ligand: benzene
        solvent: !Combinatorial [buffer, implicit]
        leap:
          parameters: [leaprc.protein.ff14SB, leaprc.water.tip4pew]

    experiment-t4-benzene:
      system: t4-benzene
      protocol: absolute-binding
      restraint:
        type: !Combinatorial [FlatBottom, Harmonic, Boresch]

Here we have defined a new solvent that uses the GBSA/OBC2 model, and we have changed the ``systems.t4-benzene.solvent``
keyword from simply ``buffer`` to ``!Combinatorial [buffer, implicit]``. We've also added ``!Combinatorial`` in the
``experiment-t4-benzene`` section, this time specifying several restraint types. When YANK parses ``experiment-t4-benzene``,
it generates 6 different binding free energy calculations using 3 different restraints and 2 different solvent models.

.. rst-class:: html-toggle

The full script after the changes
"""""""""""""""""""""""""""""""""

.. code-block:: yaml

    ---

    options:
      verbose: yes
      minimize: yes
      default_timestep: 2.0*femtoseconds
      default_nsteps_per_iteration: 500
      temperature: 300*kelvin
      pressure: 1*atmosphere

    molecules:
      benzene:
        filepath: benzene.tripos.mol2
        antechamber:
          charge_method: bcc
        leap:
          parameters: [leaprc.gaff]
      t4-lysozyme:
        filepath: input/t4.pdb

    solvents:
      water:
        nonbonded_method: PME
        nonbonded_cutoff: 9*angstroms
        clearance: 16*angstroms
        solvent_model: tip4pew
        leap:
          parameters: [leaprc.water.tip4pew]
      vacuum:
        nonbonded_method: NoCutoff
      buffer:
        nonbonded_method: PME
        nonbonded_cutoff: 11*angstroms
        clearance: 12*angstroms
        positive_ion: Na+
        negative_ion: Cl-
        ionic_strength: 100*millimolar
        solvent_model: tip3p
        leap:
          parameters: [leaprc.water.tip3p]
      implicit:
        nonbonded_method: NoCutoff
        implicit_solvent: OBC2
        implicit_solvent_salt_conc: 100*millimolar

    systems:
      hydration-system:
        solute: benzene
        solvent1: water
        solvent2: vacuum
        leap:
          parameters: [leaprc.protein.ff14SB]
      t4-benzene:
        receptor: t4-lysozyme
        ligand: benzene
        solvent: !Combinatorial [buffer, implicit]
        leap:
          parameters: [leaprc.protein.ff14SB, leaprc.water.tip4pew]

    protocols:
      hydration-protocol:
        solvent1:
          alchemical_path: auto
        solvent2:
          alchemical_path: auto
      absolute-binding:
        complex:
          alchemical_path:
            lambda_electrostatics: [1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
            lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]
            lambda_restraints:     [0.00, 0.25, 0.50, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
        solvent:
          alchemical_path: auto

    experiment-t4-benzene:
      system: t4-benzene
      protocol: absolute-binding
      restraint:
        type: !Combinatorial [FlatBottom, Harmonic, Boresch]

    experiment-hydration:
      system: hydration-system
      protocol: hydration-protocol

    experiments: [experiment-t4-benzene, experiment-hydration]

Changing sampling method
========================

By default, YANK uses a combination of Monte Carlo rigid displacement/rotation of the ligand and Langevin dynamics to
propagate the replicas of the Hamiltonian Replica Exchange algorithm, but both the propagation and the enhanced sampling
algorithm can be configured.

.. code-block:: yaml

    mcmc_moves:
      langevin:
        type: LangevinSplittingDynamicsMove
        timestep: 4.0*femtosecond
        n_steps: 1250
        collision_rate: 1.0/picosecond
        reassign_velocities: no
        splitting: 'V R O R V'
        n_restart_attempts: 4

    samplers:
      repex:
        type: ReplicaExchangeSampler
        mcmc_moves: langevin
        number_of_iterations: 100000
        online_analysis_interval: 10
      sams:
        type: SAMSSampler
        mcmc_moves: langevin
        state_update_scheme: global-jump
        gamma0: 10.0
        flatness_threshold: 10.0
        number_of_iterations: 100000
        online_analysis_interval: 1000

The ``mcmc_moves`` block allows to specify the constructor of any ``MCMCMove`` object available in the ``openmmtools.mcmc``
module (`see API docs there <https://openmmtools.readthedocs.io/en/latest/mcmc.html>`_). The ``type`` keyword is used to
indicate the class name of the ``MCMCMove``, and the other keywords are passed to its constructor. Note that ``timestep``
and ``n_steps`` overwrite the global options ``default_timestep`` and ``default_nsteps_per_iteration`` respectively
defined above.

The ``samplers`` block has a similar structure, but you can invoke the constructor of any sampler in the ``yank.multistate``
package. Similarly, the ``number_of_iterations`` overwrites the global option ``default_number_of_iterations``. Note also
that we can pass to the ``samplers.SAMPLER_ID.mcmc_moves`` keyword any identifier of the ``mcmc_moves`` section.


Starting from different input files
===================================

Files with multiple molecules
"""""""""""""""""""""""""""""

If you have input files describing multiple molecules, you can define ``molecules`` by selecting all or a subset of them.

.. code-block:: yaml

    molecules:
      t4-lysozyme:
        filepath: input/t4-frames.pdb
        select: 0
      binders:
        filepath: input/binders.mol2
        antechamber:
          charge_method: bcc
        select: all

The example above pick the first structure in the ``t4-frames.pdb`` file to define the ``t4-lysozyme`` molecule, while
it reads all the molecules in the ``binders.mol2`` file to generate the binders. In the latter case, YANK will generate
as many experiments as the number of molecules in ``binders.mol2``.

.. note::

    Selecting mol2 and sdf files requires an installation of the OpenEye library.

To select a subset of the structures in a single file you can do

.. code-block:: yaml

    molecules:
      t4-lysozyme:
        filepath: input/t4-frames.pdb
        select: !Combinatorial [0, 1, 4, 5, 8]

SMILES
""""""

If you have OpenEye installed, then you can also generate small molecules from their SMILES representation.

.. code-block:: yaml

    molecules:
      ...
      benzene:
        smiles: c1ccccc1
        openeye:
          quacpac: am1-bcc
        antechamber:
          charge_method: null
      binders:
        filepath: input/L99A-binders.csv
        openeye:
          quacpac: am1-bcc
        antechamber:
          charge_method: null
        select: all

    systems:
      t4-benzene:
        receptor: t4-lysozyme
        ligand: benzene
        pack: yes
        ...

Here we define a ``benzene`` molecules passing directly its SMILES representation to the ``smiles`` keyword. It is also
possible to load a CSV file defining multiple SMILES molecule to generate combinatorial experiments. The charges for
these molecules are assigned using the OpenEye's AM1-BCC charge scheme rather than antechamber.

Finally, note the ``pack: yes`` keyword in the definition of the system. Since no coordinates are specified by the SMILES
representation, the molecule is placed somewhat randomly in the solvation box. Setting the ``pack`` option brings receptor
and ligand close before generating the system with tleap. This is very helpful in explicit solvent since otherwise tleap
may generate a huge solvation box to include both molecules.

Extra directives for tleap
""""""""""""""""""""""""""

In special cases, you may want to add more directives for tleap. For example, you may want to use tleap to rename some
atoms that are not recognized by the force field you want to load. You can do so by creating a ``leaprc.renameatoms``
including the directive to inject into the tleap script

.. code-block:: none

    addPdbAtomMap {
      { 1DH6 DH61 }
      { 2DH6 DH62 }
    }

and then load it normally together with the leap parameters in the YAML file

.. code-block:: yaml

    molecules:
      t4-lysozyme:
        filepath: input/t4.pdb
        leap:
          parameters: [leaprc.renameatoms]

When the automatic pipeline won't do
""""""""""""""""""""""""""""""""""""

The easiest way to set up a free energy calculation in YANK is to use the automatic pipeline based on AmberTools, but
sometimes that is not flexible enough. In this case, you can use any other program to generate parameter/coordinates files
in Amber or Gromacs format or and XML representation of an OpenMM system.

.. code-block:: yaml

  solvents:
      pme:
        nonbonded_method: PME
        nonbonded_cutoff: 11*angstroms
      vacuum:
        nonbonded_method: NoCutoff

    systems:
      t4-benzene:
        phase1_path: [complex.top, complex.gro]
        phase2_path: [solvent.top, solvent.gro]
        ligand_dsl: resname MOL
        solvent: pme
        gromacs_include_dir: include/
      hydration-system:
        phase1_path: [solvent1.prmtop, solvent1.inpcrd]
        phase2_path: [solvent2.prmtop, solvent2.inpcrd]
        solvent1: pme
        solvent2: vacuum

The ``t4-benzene`` system now loads parameter/coordinates files in Gromacs format for the calculation of the ligand in
complex and in bulk. The ligand is identified by the ``ligand_dsl`` MDTraj DSL selection string. Note that the ``solvent``
must still be specified to indicate the long-range interactions treatment.

When ``ligand_dsl`` is not specified, YANK alchemically modifies the whole solute, which is used in ``hydration-system``
to perform a hydration free energy calculation.

.. note::

    If your ligand/solute has a net charged and you're using PME, your systems should contain enough ions to neutralizing. YANK will decouple them together with the system to maintain a neutral solvation box.
