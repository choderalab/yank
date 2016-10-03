.. _yaml_protocols_head:

Protocols Header for YAML Files
*******************************

The ``protocols`` header tells YANK which alchemical thermodynamic path to draw samples along. Together with a :doc:`system <systems>`, this header makes a complete :doc:`experiment <experiments>`.

Just like in :doc:`molecules <molecules>`, protocols can have arbitrary, user defined names.
In the examples, these user defined names are marked as ``{UserDefinedProtocol}``.


----


.. _yaml_protocols_example:

Protocols Syntax
================
.. code-block:: yaml

   protocols:
     {UserDefinedProtocol:
       {PhaseName1}:
         alchemical_path:
           lambda_electrostatics: [1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
           lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]
       {PhaseName2}:
         alchemical_path:
           lambda_electrostatics: [1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
           lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]

This is the only valid format for the protocols header and consistes of multiple parts which we break down here.

The ``{PhaseName1}`` and ``{PhaseName2}`` headers are ``system`` dependent headers. 
The ``{UserDefinedSystem`` and ``{UserDefinedProtocol}`` are combined in the ``experiements`` header to make on object.
Each ``{PhaseNameX}`` in the ``{UserDefinedProtocol}`` must match with the expected type of system created. 
Below is the mappings of each ``{PhaseNameX}``:

* :ref:`Ligand/Receptor Free Energies Setup by YANK <yaml_systems_receptor_ligand>`

  * ``{PhaseName1}`` -> ``complex``
  * ``{PhaseName2}`` -> ``solvent``

* :ref:`Hydration Free Energies Setup by YANK <yaml_systems_hydration>`

  * ``{PhaseName1}`` -> ``solvent1``
  * ``{PhaseName2}`` -> ``solvent2``

 * :ref:`Arbitrary Phase Free Energies Setup by User <yaml_systems_user_defined>`

  * ``{PhaseName1}`` -> ``phase1``
  * ``{PhaseName2}`` -> ``phase2``


.. _yaml_protocls_alchemical_path:

alchemical_path, lambda_electrostatics, and lambda_sterics
----------------------------------------------------------

The ``lambda_electrostatics`` and ``lambda_sterics`` directives define the alchemical states that YANK will sample at. 
Each directive accepts an equal sized list of floats as arguments and each index of the list corresponds to what value of lambda those interactions will be controlled by at that state.
The index can be thought of as the column if the lists were stacked as a 2D array, and the state is fully described by the column, not a single row by itself.

Syntax is identical to the example above.

Valid Arguments: <Identical Sized List of Floats>
