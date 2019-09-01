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
     {UserDefinedProtocol}:
       {PhaseName1}:
         alchemical_path:
           lambda_electrostatics: [1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
           lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]
       {PhaseName2}:
         alchemical_path:
           lambda_electrostatics: [1.00, 0.75, 0.50, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
           lambda_sterics:        [1.00, 1.00, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00]

This is the only valid format for the protocols header (unless :ref:`restraints are specified <yaml_protocols_alchemical_path>`)
and consists of multiple parts which we break down here.

The ``{PhaseName1}`` and ``{PhaseName2}`` headers are ``system`` dependent headers. 
The ``{UserDefinedSystem`` and ``{UserDefinedProtocol}`` are combined in the ``experiements`` header to make on object.
Each ``{PhaseNameX}`` in the ``{UserDefinedProtocol}`` must match with the expected type of system created. 
Below is the *recommended* mappings of each ``{PhaseNameX}``:

* :ref:`Ligand/Receptor Free Energies Setup by YANK <yaml_systems_receptor_ligand>`

  * ``{PhaseName1}`` -> ``complex``
  * ``{PhaseName2}`` -> ``solvent``

* :ref:`Hydration Free Energies Setup by YANK <yaml_systems_hydration>`

  * ``{PhaseName1}`` -> ``solvent1``
  * ``{PhaseName2}`` -> ``solvent2``

* :ref:`Arbitrary Phase Free Energies Setup by User <yaml_systems_user_defined>`

  * ``{PhaseName1}`` -> ``phase1``
  * ``{PhaseName2}`` -> ``phase2``

More generally, the ``{PhaseNameX}`` only has to contain the keyword for the system you expect.
For instance, in the Ligand/Receptor case, the following would also be a valid structure:

.. code-block:: yaml

   {UserDefinedProtocol}:
     RICKsolventROLL:
       alchemical_path: ..
     my_complex_phase:
       alchemical_path: ..

Where ``RICKsolventROLL`` would correspond to the ``solvent`` phase since it contains the phrase "solvent",
and ``my_complex_phase`` would correspond to the ``complex`` phase for the same reason.

If you want to name them whatever you would like and instead rely on the which sign is used for the free energy evaluation,
you can invoke ``{UserDefinedProtocol}: !Ordered`` to make the first entry the ``+`` sign and the second ``-`` sign in the free energy difference.
Here is an example:

.. code-block:: yaml

   absolute-binding: !Ordered
     plus_sign_phase:
       alchemical_path: ...
     minus_sign_phase:
       alchemical_path: ...


.. _yaml_protocols_alchemical_path:

``alchemical_path``, ``lambda_electrostatics``, ``lambda_sterics``, and ``lambda_restraints``
---------------------------------------------------------------------------------------------

The ``lambda_electrostatics``, ``lambda_sterics``, and ``lambda_restraints`` directives define the alchemical states that YANK will sample at.
Each directive accepts an equal sized list of floats as arguments and each index of the list corresponds to what value of lambda those interactions will be controlled by at that state.
The index can be thought of as the column if the lists were stacked as a 2D array, and the state is fully described by the column, not a single row by itself.

Syntax is identical to the example above.

Only ``lambda_restraints`` are optional and do not need to be specified for each phase and system. Further, the directive
only applies if ``restraint`` :ref:`in experiments is specified <yaml_experiments_syntax>`. How and where the
``lambda_restraints`` should be will be up to the user. To see use cases of this directive, please see any of the following:

* :ref:`The Harmonic restraint in our detailed binding free energy tutorial <p-xylene-explicit>`
* :ref:`The FlatBottom restraint in our host-guest binding free energy tutorial <host_guest_implicit>`
* `The Boresh restraint in our YANK GitHub Examples <https://github.com/choderalab/yank-examples/tree/master/examples/binding/abl-imatinib>`_

Valid Arguments: <Identical Sized List of Floats>/auto


.. _yaml_protocols_auto:

Automatic Path Determination
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   protocols:
     {UserDefinedProtocol}:
       {PhaseName1}:
         alchemical_path: auto


YANK can automatically determine the sequence of values for the ``lambda_X`` values using an algorithm we called
"thermodynamic trailblazing". YANK will distribute states between the fully
coupled and fully decoupled states for you based on the restraint scheme, class of transformation, and input system.
The intermediate states are distributed such that the standard deviation of potential energy differences between the states is equal,
which should improve replica mixing between the states over placing the states yourself.

Specifying the option ``auto`` for your ``alchemical_path`` will start the process as shown in the example. The process
may take a while, but the algorithm is capable of resuming if it gets unexpectedly interrupted, and the final path is
saved as a new `.yaml` file in your :ref:`output_dir/experiments <yaml_options_output_dir>`, and immediately used as
the input for your simulation.

During the calculation, The algorithm generates some snapshot for all intermediate states  samples that are generated during
the path determination are used to seed the replicas of Hamiltonian replica exchange or SAMS calculations. If ``minimize``
in the ``options`` section is set to ``yes``, this is the stage where the minimization of the input structure occurs.


.. _yaml_protocols_auto_options:

More options for automatic determination of the alchemical path
---------------------------------------------------------------

.. code-block:: yaml

   protocols:
     {UserDefinedProtocol}:
       {PhaseName1}:
         trailblazer_options:
           # Number of initial equilibration iterations performed in the first state before running the algorithm.
           n_equilibration_iterations: 1000
           # Whether the heavy atoms of the receptor are positionally restrained with a harmonic potential during the algorithm.
           constrain_receptor: false
           # Number of samples used to estimate the standard deviation of the potential between two states.
           n_samples_per_state: 100
           # The "distance" in potential standard deviation between two intermediate states up to 'threshold_tolerance'.
           std_potential_threshold: 0.5  # in kT
           threshold_tolerance: 0.05  # in kT
           # Whether to traverse the path in the forward (given by 'alchemical_path') or reversed direction.
           reversed_direction: true
           # A variable controlling the path if 'alchemical_path' contains mathematical expressions.
           function_variable_name: lambda

         alchemical_path:
           lambda_electrostatics: 2*(lambda - 0.5) * step(lambda - 0.5)
           lambda_sterics: step_hm(lambda - 0.5) + 2*lambda * step_hm(0.5 - lambda)
           lambda: [1.0, 0.0]

It is possible to set some of the parameters of the algorithm performing the path discretization in the
``trailblazer_options`` tag. Moreover, instead of ``alchemical_path: auto`` you can express the alchemical variables
in terms of mathematical Python expressions depending on a variable that will be discretized by the algorihtm. In this
case, ``alchemical_path`` must contain the value of the end states assumed by the variable. In the example above,
electrostatic interactions are deactivated beetween the variable ``lambda`` 1.0 and 0.5, while Lennard-Jones potential
is decoupled between ``lambda`` 0.5 and 0.0.

All functions available in the Python standard module ``math`` are available together with
* ``step(x)`` : Heaviside step function (1.0 for x=0)
* ``step_hm(x)`` : Heaviside step function with half-maximum convention.
* ``sign(x)`` : sign function (0.0 for x=0.0)


.. _yaml_protocols_thermodynamic_variables:

Temperature and pressure
------------------------

It is possible to vary temperature and pressure along the alchemical path, but the end states must have the same values.
The number of window must be identical to the other lambda variables. A short example:

.. code-block::yaml

   protocols:
     {UserDefinedProtocol}:
       {PhaseName1}:
         alchemical_path:
           lambda_electrostatics: [1.00, 0.50, 0.00, 0.00, 0.00]
           lambda_sterics:        [1.00, 1.00, 1.00, 0.50, 0.00]
           temperature:           [300*kelvin, 310*kelvin, 320*kelvin, 310*kelvin, 300*kelvin]
       {PhaseName2}:
         alchemical_path:
           lambda_electrostatics: [1.00, 0.50, 0.00, 0.00, 0.00]
           lambda_sterics:        [1.00, 1.00, 1.00, 0.50, 0.00]

Valid Arguments: <List of Quantities>


.. _yaml_protocols_video:

Protocols How-To Video
======================

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/nVVl6if6g0w" frameborder="0" allowfullscreen></iframe>

