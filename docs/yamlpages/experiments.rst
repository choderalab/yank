.. py:currentmodule:: yank.restraints

.. _yaml_experiments_head:

Experiments Header in YAML Files
********************************

The ``experiments`` header is the final object that YANK needs to run a free energy simulation.
Multiple experiments can be defined in a single YAML file and each one is considered a fully defined free energy calculation.


----


.. _yaml_experiments_syntax:

Experiments Syntax
==================
.. code-block:: yaml

   experiments:
     system: {UserDefinedSystem}
     protocol: {UserDefinedProtocol}
     sampler: {UserDefinedSampler}
     restraint:
       type: FlatBottom
       {Restraint Parameter}
       ...
     restraint: {UserDefinedRestraint}
     restraint: [{List}, {of}, {UserDefinedRestraints}]
     options:
       {Any Valid Option}
       {Any Valid Option}
       ...

This is the structure of an ``experiment``.

It takes a ``{UserDefinedSystem}`` (see :doc:`systems <systems>`) and a ``{UserDefinedProtocol}`` (see :doc:`protocols <protocols>`)
to create the experiment and are the only required arguments. You can also specify any ``{UserDefinedSampler}`` (see
:doc:`samplers <samplers>`), however this is optional and will fill in with a default sampler.

The ``restraint`` is an **optional** keyword that applies a restraint to the ligand to keep it close to the receptor.
There are multiple ways to invoke this key word,

* Define a singular whole restraint here
* Specify a single restraint defined in the :doc:`restraints <restraints>` block
* Specify a list of restraints, each one an entry in the :doc:`restraints <restraints>` block

In the example block above, you have to choose one of the formats, not all of them. To see how to define a restraint
either here in the ``experiment`` block or as a reference to the ``restraint`` block, please see the
:doc:`restraints page <restraints>`.

The ``options`` directive lets you overwrite :doc:`any global setting <options>` specified in the header ``options`` for
this specific experiment.

One option is to select restrained atoms through :class:`Topgraphical Regions <yank.Topography>` defined as part of your
:ref:`molecule's regions <yaml_molecules_regions>`. You can also select atoms through a
:func:`compound region <yank.Topography.select>` where regions are combined through set operators
``and``/``or``.

.. _yaml_experiments_multiple:

Running Multiple Experiments
============================

A single experiment can be defined by the following example. However, if one would like to run multiple experiments from the same YAML file, then follow these instructions:

#. Create an outermost header above ``experiments`` with whatever name of experiment you want to run. We label this as ``{UserDefinedExperiment}``.
#. Define your ``{UserDefinedExperiment}`` by creating sub-directives just as you would in the main ``experiment`` header (se the :ref:`Experiments Syntax <yaml_experiments_syntax>` for description).
#. Repeat this process for as many experiments as you want.
#. Create an ``experiments`` header below your user defined ones with the syntax: ``experiments: [{UserDefinedExperiment}, {UserDefinedExperiment}, ...]`` where the list is the experiments you defined.

  * **NOTE**: There are no sub-directives under the ``experiments`` header when invoked this way.

Here is an example

.. code-block:: yaml

   {UserDefinedExperiment}:
     system: {UserDefinedSystem}
     protocol: {UserDefinedProtocol}
     restraint:
       type: FlatBottom
   {ASecondUserDefinedExperiment}:
     system: {UserDefinedSystem}
     protocol: {UserDefinedProtocol}
     restraint:
       type: Boresch
   experiments: [{UserDefinedExperiment}, {ASecondUserDefinedExperiment}]


Example YAML File
=================

.. code-block:: yaml

    options:
      minimize: yes
      verbose: yes
      number_of_iterations: 3000
      nsteps_per_iteration: 500
      temperature: 300*kelvin
      pressure: 1*atmosphere
      timestep: 2.0*femtoseconds
      output_dir: explicit
      resume_setup: yes
      resume_simulation: yes

    molecules:
      Abl:
        filepath: input/2HYY-pdbfixer.pdb
      STI:
        filepath: input/STI02.mol2
        epik:
          select: !Combinatorial [0, 1, 2]
          ph: 7.4
          ph_tolerance: 7.0
          tautomerize: no
        openeye:
          quacpac: am1-bcc
        antechamber:
          charge_method: null

    solvents:
      pme:
        nonbonded_method: PME
        switch_distance: 11*angstroms
        nonbonded_cutoff: 12*angstroms
        ewald_error_tolerance: 1.0e-4
        clearance: 9*angstroms
        positive_ion: Na+
        negative_ion: Cl-

    systems:
      Abl-STI:
        receptor: Abl
        ligand: STI
        solvent: pme
        leap:
          parameters: [leaprc.protein.ff14SB, leaprc.gaff, leaprc.water.tip4pew]

    restraints:
        RigidBoresch:
            type: Boresch
            rigidify: 15*degrees
        BasicHarmonic:
            type: Harmonic

    protocols:
      absolute-binding:
        complex:
          alchemical_path: auto
        solvent:
          alchemical_path: auto

    experiments:
      system: Abl-STI
      protocol: absolute-binding
      restraint: [RigidBoresch, BasicHarmonic]
