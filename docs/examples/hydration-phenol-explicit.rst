.. _hydration_phenol_explicit:

Hydration Free Energy of Phenol in Explicit TIP3P Water
=======================================================

This example highlights the other major type of free energy difference YANK can compute, hydration free energies. More
generally it is solvation free energies, but TIP3P water is the only explicit water you can specify in YANK right now.

This example resides in ``{PYTHON SOURCE DIR}/share/yank/examples/hydration/phenol``. The rest of the example here
assumes you are in this directory.

This example can actually be considered a subset of the :doc:`FreeSolv database example <freesolv-imp-exp>`, however,
describes in more detail how to set up a hydraiton free energy calculation.

Examining YAML File
-------------------

Hydration free energies are different from binding free energies in that there is only a single molecule and two
separate solvents. The other major difference is that the phases have different names which we will point out in this
example.

Options Heading
^^^^^^^^^^^^^^^

There are no changes in this directive relative to the
:ref:`options in the binding free energy example <hydration_phenol_explicit_options>`.

Molecules Heading
^^^^^^^^^^^^^^^^^

In hydration free energy, only one molecule must be specified, although more could be specified for other simulatons.
Otherwise, there are no changes in this directive relative to the
:ref:`molecules in the binding free energy example <hydration_phenol_explicit_molecules>`.

Solvents Heading
^^^^^^^^^^^^^^^^

The solvents heading differs from :doc:`previous <p-xylene-explicit>` :doc:`examples <host-guest-implicit>` in that there
two solvents must be specified. Let us take a look at this section in the ``explicit.yaml`` file.

.. code-block:: yaml

    solvents:
      water:
        nonbonded_method: PME
        nonbonded_cutoff: 9*angstroms
        clearance: 16*angstroms
      vacuum:
        nonbonded_method: NoCutoff

Here we name two different solvents: ``water`` and ``vacuum``. We then set the options for both solvents. The ``water``
options are the PME nonbonded treatment with a cutoff at 9 Angstroms, then we fill in box with at least 16 angstroms
of water between the solute and the box edge. The ``vacuum`` setting is very simple, we just set a ``NoCutoff`` scheme
WITHOUT setting ``implicit_solvent`` to create a vacuum system.

YANK will know not to add a barostat to the ``vacuum`` system. Even though a ``pressure`` is specified in the general
options, the choice of NoCutoff and no other options implies no barostat (since trying to do a periodic system without
a cutoff would be impossibly resource intensive).

Systems Heading
^^^^^^^^^^^^^^^

The ``systems`` heading in this case has slightly different keys which tell YANK that this is a hydration free energy
system. Nothing in the previous sections indicated that this would be a hydration free energy so lets look at this section
here.

.. code-block:: yaml

    systems:
      hydration-system:
        solute: phenol
        solvent1: water
        solvent2: vacuum
        leap:
          parameters: leaprc.gaff

Lets start by looking at the familiar options.
We start by naming our system whatever we want, so we call it ``hydration-system``. ``leap`` tells YANK where to get
parameters for the force field. Since we don't have a protein to worry about, we can just point at a generalized force
field and run through from there.

Now lets look at the hydration free energy specific directives. We first specify the solute molecule with ``solute``
which points at a user defined molecule. Next we specify our two solvents with ``solvent1`` and ``solvent2`` which
point at the two user defined solvents. Because free energy differences are just a sign convention, the ``solvent1``
phase is considered the ``+`` and ``solvent2`` is the ``-``.

Protocols Heading
^^^^^^^^^^^^^^^^^

The protocol's heading works the same as the binding examples, but requires slightly different phase deffinitions since
there is no complex/solvent phase.

.. code-block:: yaml

    protocols:
      hydration-protocol:
        solvent1:
          alchemical_path:
            lambda_electrostatics: [1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
            lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]
        solvent2:
          alchemical_path:
            lambda_electrostatics: [1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
            lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]

Here we define the two phases by ``solvent1`` which is linked to the ``solvent1`` of the ``systems``, and similalry the
``solvent2`` is linked to ``solvent2``.

Just like in the binding examples, the names ``solvent1`` and ``solvent2`` for
phase names is semi-arbitrary in that the name only has to contain the string. Perfectly valid names would be
``waterphasesolvent1`` and ``VACsolvent2UUM`` pointing to ``solvent1`` and ``solvent2`` respectively.

Experiments Heading
^^^^^^^^^^^^^^^^^^^

Just like in the binding example, we point to a protocol and a system to make our experiment

.. code-block:: yaml

    experiments:
      system: hydration-system
      protocol: hydration-protocol



Running and Analyzing the Simulation
------------------------------------

The execution and analysis of the simulation are handled the same as
:doc:`in the T4 Lysozyme binding example <p-xylene-explicit>`. Please see the documentation on that page for more information.