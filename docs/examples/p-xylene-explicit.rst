.. _p-xylene-explicit:

p-xylene binding to T4 lysozyme L99A in explicit solvent
========================================================

This example illustrates the computation of the binding free energy of p-xylene to the T4 lysozyme L99A mutant in an
NPT system of explicit water. From description, we will craft the settings, molecules, and simulations in YANK. This
covers the basics of running YANK and detailed instructions will be given for every step.

This example resides in ``{PYTHON SOURCE DIR}/share/yank/examples/binding/t4-lysozyme``. The rest of the example here
assumes you are in this directory.


Examining YAML file
-------------------

We start this example by looking at the YAML file which controls all of the setting for YANK, ``p-xylene-explicit.yaml``. This file
is what YANK uses to define all simulation parameters, and actually run the experiments. The file is broken down into 6
sections: ``options``, ``molecules``, ``solvents``, ``systems``, ``protocols`` and ``experiments``. We'll go through each of those here
as they pertain to the example, but please see the :ref:`detailed YAML documentation: <yaml_head>` for all possible options
and valid values.

This example goes through all the options we choose and why in explicit detail, future examples will not cover every
option, but instead just the changes.

.. _hydration_phenol_explicit_options:

Options Heading
^^^^^^^^^^^^^^^
The ``options`` heading is where the general simulation settings are chosen. These are experiment wide settings that will
govern the simulation at runtime, and the the global conditions that we care about. Looking back at our description,
it is here that we will enforce the NPT ensemble settings.

.. code-block:: yaml

  options:
    minimize: yes

The ``minimize`` option tells YANK to run energy minimization before MD simulation to get the molecules into a more
energetically stable starting position. This is helpful as many input files are missing components (such as hydrogens),
were taken from a different force field, or do not have the ligand and receptor positions taken from the same crystal
structure. Because each of these small perturbations can cause large energy changes, we minimize the energy to reduce
the odds of causing a simulation crash.

.. code-block:: yaml

   options:
     verbose: yes

The ``verbose`` options tells YANK to be very informative about what it is doing at every step. We turn verbose on to
see what the simulation is doing. As someone learning YANK, this is a very helpful to see every action being taken to
better understand what is happening.

.. code-block:: yaml

   options:
     temperature: 300*kelvin

This option sets the temperature of our system. YANK simulations use Langivin Dynamics to control temperature. If this
option is not set, Velocity Verlet time integration will be used instead.

.. code-block:: yaml

   options:
     pressure: 1*atmosphere

This option sets the pressure of our system. YANK uses OpenMM's Monte Carlo Barostat to regulate the pressure. By default,
pressure is set to 1 atmosphere anyways, we explicitly set it here for this example. If you wanted to instead do NVT
simulations, you would need to explicitly set this option to ``null``.

.. _hydration_phenol_explicit_molecules:

Molecules Heading
^^^^^^^^^^^^^^^^^

The ``molecules`` heading is where we will specify our ligand and receptor molecules. Both the receptor and the ligand
molecules can have arbitrary names, but we prefer to call them helpful names like ``t4-lysozyme`` and ``p-xylene``.

Let us look at the receptor, T4-Lysozyme. We'll look at the whole code block for defining it.

.. code-block:: yaml

   molecules:
     t4-lysozyme:
       filepath: input/receptor.pdbfixer.pdb
       leap:
         parameters: oldff/leaprc.ff14SB

First we define our molecule name ``t4-lysozyme``, this name will come up again in later sections, hence why we have
given it a meaningful name.

Next we tell YANK where the file is, so ``filepath`` points at the file relative to where
the the yaml script.

Lastly we have to tell YANK where to get the molecule's parameters to put into a simulation. The combination of
``leap`` and ``parameters`` tells YANK how to pull in parameters. The path of ``parameters`` is relative to the the
LEaP directory.

Now let us look at the ligand molecule, para-xylene

.. code-block:: yaml

   molecules:
    p-xylene:
      filepath: input/ligand.tripos.mol2
      antechamber:
        charge_method: bcc

First we give our ligand a meaningful name: ``p-xylene``.

Next we tell YANK where the file is with ``filepath``.

The ``antechamber`` command is probably the most loaded command in the YAML file. Several things happen when this command
is invoked. First, specifying ``antechamber`` tells YANK to prep the molecule by running it through ANTECHAMBER to get
missing torsions, bonds, and other parameters that may not be in the file. This creates an FRCMOD file which is automatically
loaded in with the molecule to make the final files, this step will be entirely transparent to you, the user.

Next is
the sub-directive ``charge_method``. This sub-directive is required with ``antechamber`` and tells YANK how to assign
charges to our ligand molecule. The ``bcc`` option tells YANK to get this molecule's charges from AM1-BCC method at
run time. The other valid option is ``null`` which tells YANK to still use ANTECHAMBER to get missing parameters, but
to not attempt to assign charges. This is helpful if you pre-assigned charges in the input file, but still need the
missing bonded parameters.


Solvents Heading
^^^^^^^^^^^^^^^^
The ``solvents`` heading is where we specify what type of environment we want the ligand and receptor to be in.
Looking at the general description of our system, we know we want explicit solvent (instead of implicit/continuous
dielectric). Since there is only one solvent we need to define, we will look at the whole code block at once.

.. code-block:: yaml

 solvents:
   pme:
     nonbonded_method: PME
     nonbonded_cutoff: 9*angstroms
     clearance: 16*angstroms
     positive_ion: Na+
     negative_ion: Cl-

First we define an arbitrary name for our solvent. Here we call it ``pme`` since we will be using Particle Mesh Ewald
to handle our long-range electrostatics. Again, we could have named this anything we want, but we gave it a meaningful
name to go with.

``nonbonded_method`` is where we choose how to handle our nonbonded interactions. Because we have a large, explicit solvent
system, it would be very taxing to compute every interaction over all atoms and every periodic copy. Instead, we
divide the interactions into short- and long- range interactions with the Particle Mesh Ewald method for computing these
interactions efficiently. This option is also the main place where implicit vs explicit solvent is chosen. We do not
show how to set up an implicit solvent in our :doc:`Host-Guest Example <host-guest-implicit>`. This is also where you
can define a vacuum solvent, though that is not covered in this example.

Since we are dividing the system into long and short range interactions, we specify where that cutoff should take place
with ``nonbonded_cutoff``.

``clearance`` defines how to fill the simulation with solvent and how the box vectors should be drawn.
The box vectors are drawn such that the distance between the box edge and any part of the receptor is at
least the distance specified. Next, the space is filled in with TIP3P water. The water molecules are just replicated
copies of a unit cell of water, so you absolutely should specify the ``minimize`` option in the general ``options`` header.

``positive_ion`` and ``negative_ion`` tell the simulation what Ions to add in to make the system neutral. If these
options are not specified, no counter ions will be added.

Systems Heading
^^^^^^^^^^^^^^^
This heading is where we combine the molecules and the solvent to make an actual system that we can simulate. This is
also where we specify the parameters we use system wide to account for missing ones from individual molecules. We also
use the names we set up in the ``molecules`` and ``solvents`` section, hence why it was important to have meaningful names.

This block is very strait forward.

.. code-block:: yaml

   systems:
     t4-xylene:
       receptor: t4-lysozyme
       ligand: p-xylene
       solvent: pme
       leap:
         parameters: [oldff/leaprc.ff14SB, leaprc.gaff2, frcmod.ionsjc_tip3p]

We first define a name for our system, ``t4-xylene`` is the arbitrary name we chose.

Next we define what the receptor is with the ``receptor`` directive. Note that this points at our arbitrary name of
"t4-lysozyme" from before.

Then we specify the ligand with ``ligand``. Note that this points to our arbitrary ligand name of "p-xylene"

Then comes the solvent with ``solvent``. Note this points at the arbitrary named "pme" from before.

Lastly, we need to define where to get parameters for the atoms with the ``leap`` and subsequent ``parameters`` directives.
Even if you specify all the atom parameters for every molecule in ``molecules``, you will still need to specify this
pair of options to parametrize the explicit water. Note that multiple files can be specified as a comma separated list so
long as they are enclosed by brackets, ``[  ]``.


Protocols Heading
^^^^^^^^^^^^^^^^^
The ``protocols`` heading and its options will be the most foreign to those not familiar with alchemical simulations.
Free energy calculations are computationally difficult to compute because in a physical sense, the ligand needs to drift
in and out of the binding pocket. This action happens on the order of milliseconds to seconds, which are simulation
times that are very difficult to achieve with direct simulation. Instead, we use a computationally efficient thermodynamic
cycle along efficient thermodynamic paths to mimic the end states (bound ligand in solvent box -> ligand in solvent
+ receptor in solvent). For more reading, please see the alchemistry.org page on `the thermodynamic cycle <http://www.alchemistry.org/wiki/Thermodynamic_Cycle>`_
and on `choosing intermediate states <http://www.alchemistry.org/wiki/Constructing_a_Pathway_of_Intermediate_States>`_
for more details.

This header controls how many states you will sample, and which values along the thermodynamic paths to sample in each
phase.

.. code-block:: yaml

   protocols:
     absolute-binding:
       complex:
         alchemical_path:
           lambda_electrostatics: [1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
           lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]
           lambda_restraints:     [0.00, 0.25, 0.50, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
       solvent:
         alchemical_path:
           lambda_electrostatics: [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
           lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]

We first defined a name for our protocol. The name is arbitrary and we choose ``absolute-binding``.

Next we define what happens in the ``complex`` phase. Note that the name ``complex`` is semi-arbitrary. It can be called
whatever you would like so long as it contains the string "complex" in it. ``alchemical_path`` is a required argument
as is ``lambda_electrostatics`` and ``lambda_sterics``. We will discuss the optional ``lambda_restraints`` momentarily,
first lets look at the syntax of these arguments.

Each of the ``lambda_...`` arguments takes a list of lambda values where each index corresponds with a single state. E.g.
index 0 (the first entry) of ``lambda_electrostatics`` is the value the electrostatic lambda will take in the first
state. At the same time, index 0 of the ``lambda_sterics`` is what value the sterics lambda will take in the first state.
This also means that both directives must have the same number of values.

The optional ``lambda_restraints`` tells the restraints which we specify in ``experiments`` how coupled they should be
in each state. We will specify a ``Harmonic`` restraint which will keep the ligand close to the centroid of the receptor
through a weak harmonic biasing potential. However, we only want this restraint on when the ligand is decoupled to
prevent it from drifting too far away, so its lambda values actually are coupled, whereas the nonbonded lambdas are
decoupled. Also note how only one phase has ``lambda_restraints`` specified. This is because the restraint only makes
sense in the ``complex`` phase as we actually do want the ligand to explore the other phase.

Lastly, we define what happens in the ``solvent`` phase. This again is a semi-arbitrary name and can be whatever you
want, so long as it contains the string "solvent". In `the thermodynamic cycle <http://www.alchemistry.org/wiki/Thermodynamic_Cycle>`_
for this process, we have to account for the free energy of removing the harmonic restraints and then transferring the
ligand to a standard state ``solvent`` box. YANK automatically accounts for this free energy for you which is part of
the reason ``lambda_restraints`` do not appear in the ``solvent`` phase.


Experiments Heading
^^^^^^^^^^^^^^^^^^^

Finally we have the ``experiments`` header where we combine a system and a protocol to make the final run.

This is the simplest block

.. code-block:: yaml

   experiments:
     system: t4-xylene
     protocol: absolute-binding


This block requires only a ``system`` to target the named system we made, and a ``protocol`` to target the named
protocol.


Running the Simulation
----------------------

We are finally ready to run our first simulation, now that everything has been set up in the YAML file. To run the
simulation, issue the following command:

.. code-block:: bash

 $ yank script --yaml=p-xylene-explicit.yaml

and let the simulation take care of the rest. What happens next is YANK will set up the files as we have specified,
in this running the ligand through ANTECHAMBER, take the prepped ligand and receptor to make a solvated complex, and run
both a complex and solvent simulation to compute the absolute free energy of binding. Several steps will happen over the
simulation, we'll run through them here:

#. The molecules get processed through LEaP and put into OpenMM
#. The output file is pre-processed
#. The different states are all prepared and the output file is updated
#. Each thermodynamic state is minimized for the complex system
#. A Hamiltonian Replica Exchange simulation is run
#. The process is repeated for the solvent simulation

During the simulation you will see that one iteration propagates, then YANK attempts Hamiltonian replica exchange which
you see as a large table of energies. The swap statistics are then shown showing how many times two replicas exchanged
with one another and finally some timing information (how much longer we expect the simulation to take and so on).

Analyzing Results
-----------------

Once both phases of the simulation run, we can compute the final binding free energy by running the following command.

.. code-block:: bash

 $ yank analyze --store=p-xylene-out

This complex and solvent phase will be automatically loaded in, decorrelated, and analyzed to get the free energy. We
use the energies from a simulation box with expanded cutoff radius to reduce the impact a cutoff has from anisotropic
dispersion correction. Free energy itself is then analyzed through the Multistate Bennet Acceptance Ratio (MBAR) to
get the minimally biased free energy estimate across the two phases.

You should make note of how many decorrelated samples are left after analysis, if you feel that there were not enough
samples, run for longer to get more. This can be done by modifying the YAML file in the ``options`` header and adding
the following options:

.. code-block:: yaml

   options:
     default_number_of_iterations: <Some Integer>
     resume_simulation: yes

where you replace ``<Some Integer>`` with a number larger than the number of iterations you just ran.


Other Files in this Example
---------------------------
In this example, we also include an alternate YAML file called ``implicit.yaml`` which uses an implicit solvent instead
of explicit solvent. The other main difference is that this is effectively NVT ensemble since NPT ensemble makes no
sense in implicit solvent. It the execution and analysis of this system are identical, but replace the script target
in command line with ``--yaml=p-xylene-implicit.yaml``. We cover more details of the setup of an implicit solvent system
in the next example involving :doc:`a Host Guest System <host-guest-implicit>`.
