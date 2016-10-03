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
     options:
       {Any Valid Option}
       {Any Valid Option}
       ...

This is the structure of an ``experimenet``. 
It takes a ``{UserDefinedSystem}`` (see :doc:`systems <systems>`) and a ``{UserDefinedProtocol}`` (see :doc:`protocols <protocols>`)  to create the experiment.

The ``options`` directive lets you overwrite :doc:`any global setting <options>` specified in the header ``options`` for this specific experiment.


.. _yaml_experiments_multiple:

Running Multiple Experiments
============================

A single experiement can be defined by the following example. However, if one would like to run multiple experiments from the same YAML file, then follow these instructions:

#. Create a header above ``experiments`` with whatever name of experiment you want to run. We label this as ``{UserDefinedExperiment}``.
#. Define your ``{UserDefinedExperiment}`` by creating sub-directives just as you would in the main ``experiment`` header (se the :ref:`Experiments Syntax <yaml_experiments_syntax>` for description).
#. Repeat this process for as many experiments as you want.
#. Create an ``experiments`` header below your user defined ones with the syntax: ``experiments: [{UserDefinedExperiment}, {UserDefinedExperiment}, ...]`` where the list is the experiments you defined.

  * **NOTE**: There are no sub-directives under the ``experiments`` header when invoked this way.
