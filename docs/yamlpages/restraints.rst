.. py:currentmodule:: yank.restraints

.. _yaml_restraints_head:

Restraints for YAML files
*************************

The ``restraints`` block is an **optional** block that defines restraints to the ligand to keep it close to the
receptor. These restraints can be individually or collectively, all controlled through the same or different
variables in the :doc:`protocols <protocols>` section. Because this block is optional, the old way of setting a
single restraint in the :ref:`Experiments block <yaml_experiments_syntax>` is still valid to preserve backwards
compatibility (YANK 0.24.0 is compatible with the same YAML files as version 0.23.1).

.. code-block:: yaml

   restraints:
     {UserDefinedRestraint}:
        name: {UserDefinedName or restraints}
        type: {ValidRestraintType}
        {Restraint Parameter}
        {Restraint Parameter}
        ...

The universal options are :ref:`yaml_restraints_type` and :ref:`yaml_restraints_name`. The ``{Restraint Parameter}``
are optional and :ref:`unique to each restraint type <yaml_restraints_other>`.


.. _yaml_restraints_type:

.. rst-class:: html-toggle

``type``
--------

.. code-block:: yaml

   restraints:
     {UserDefinedRestraint}:
        type: {ValidRestraintType}

Select what type of restraint to use, *required*.

Valid types are: ``FlatBottom``/``Harmonic``/``Boresch``/``RMSD``/``PeriodicTorsionBoresch``

Feel free to check how each restraint works and should be applied through their API docs:

* :class:`FlatBottom Radially-Symmetric Restraints <FlatBottom>`
* :class:`Harmonic Radially-Symmetric Restraints <Harmonic>`
* :class:`Boresch Orientational Restraints <Boresch>`
* :class:`Periodic torsion restrained Boresch-like restraint <PeriodicTorsionBoresch>`
* :class:`RMSD Protein-Ligand Restraints <RMSD>`


.. _yaml_restraints_name:

.. rst-class:: html-toggle

``name``
--------

.. code-block:: yaml

   restraints:
     {UserDefinedRestraint}:
        name: {UserDefinedName or restraints}

The only other universal option in this block. The ``name`` setting which will map to the ``lambda_{name}` variable
in the :doc:`protocols <protocols>` block. These do not have to be unique so each restraint can be controlled by the
same variable if so chosen. If this is not set, it defaults to ``restraints`` so the variable controlling it will be
``lambda_restraints`` in the :doc:`protocols <protocols>` block.


.. _yaml_restraints_other:

Other Options
-------------

Every restraint has his own set of optional parameters that are passed directly to the
Python constructor of the restraint. See the API documentation in ``yank.restraints`` for the available parameters; you
can use the links below to jump to each of individual restraint types, the keyword arguments for each restraint type
are accepted as arguments in the YAML file:

* :class:`FlatBottom Radially-Symmetric Restraints <FlatBottom>`
* :class:`Harmonic Radially-Symmetric Restraints <Harmonic>`
* :class:`Boresch Orientational Restraints <Boresch>`
* :class:`Periodic torsion restrained Boresch-like restraint <PeriodicTorsionBoresch>`
* :class:`RMSD Protein-Ligand Restraints <RMSD>`

One option is to select restrained atoms through :class:`Topgraphical Regions <yank.Topography>` defined as part of your
:ref:`molecule's regions <yaml_molecules_regions>`. You can also select atoms through a
:func:`compound region <yank.Topography.select>` where regions are combined through set operators
``and``/``or``.

**Note:** The Boresch-like and RMSD restraints require that the ligand and receptor are close to each other to make sure
the standard state correction computation is stable. We recommend only using the ``Boresch``,
``PeriodicTorsionBoresch``, or ``RMSD``, options if you know the binding mode of your system already!