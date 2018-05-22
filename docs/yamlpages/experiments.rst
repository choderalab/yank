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
     options:
       {Any Valid Option}
       {Any Valid Option}
       ...

This is the structure of an ``experiment``.

It takes a ``{UserDefinedSystem}`` (see :doc:`systems <systems>`) and a ``{UserDefinedProtocol}`` (see :doc:`protocols <protocols>`)
to create the experiment and are the only required arguments. You can also specify any ``{UserDefinedSampler}`` (see
:doc:`samplers <samplers>`), however this is optional and will fill in with a default sampler.

The ``restraint`` is an **optional** keyword that applies a restraint to the ligand to keep it close to the receptor.
The only required keyword is ``type``. Valid types are: ``FlatBottom``/``Harmonic``/``Boresch``/``RMSD``/``PeriodicTorsionBoresch``/``null``. If not
specified, assumes ``null``. Every restraint has his own set of optional parameters that are passed directly to the
Python constructor of the restraint. See the API documentation in ``yank.restraints`` for the available parameters; you
can use the links below to jump to each of individual restraint types, the keyword arguments for each restraint type
are accepted as arguments in the YAML file:

* :class:`FlatBottom Radially-Symmetric Restraints <FlatBottom>`
* :class:`Harmonic Radially-Symmetric Restraints <Harmonic>`
* :class:`Boresch Orientational Restraints <Boresch>`
* :class:`Periodic torsion restrained Boresch-like restraint <PeriodicTorsionBoresch>`
* :class:`RMSD Protein-Ligand Restraints <RMSD>`

The ``options`` directive lets you overwrite :doc:`any global setting <options>` specified in the header ``options`` for
this specific experiment.

One option is to select restrained atoms through :class:`Topgraphical Regions <yank.Topography>` defined as part of your
:ref:`molecule's regions <yaml_molecules_regions>`. You can also select atoms through a
:func:`compound region <yank.Topography.select>` where regions are combined through set operators
``and``/``or``.

**Note:** The Boresch-like and RMSD restraints require that the ligand and receptor are close to each other to make sure
the standard state correction computation is stable. We recommend only using the ``Boresch``,
``PeriodicTorsionBoresch``, or ``RMSD``, options if you know the binding mode of your system already!

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
