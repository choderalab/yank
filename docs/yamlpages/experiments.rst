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
