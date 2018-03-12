.. _yaml_samplers_head:

Samplers Header for YAML Files
*******************************

The ``samplers`` section tells YANK how it should sample multiple thermodynamic states in order to estimate free energies between states of interest.
Together with the :doc:`mcmc <mcmc>` section, this section provides a flexible way to control how thermodynamic states are efficiently sampled.

----


.. _yaml_samplers_example:

Samplers Syntax
================
.. code-block:: yaml

    samplers:
        {UserDefinedSamplerName}:
            type: {Sampler}
            mcmc_moves: {MCMCName}
            number_of_iterations: {NumberOfIterations}
            {SamplerOptions}

The ``{Sampler}`` can be one of the following:
* ``MultistateSampler``: Independent simulations at distinct thermodynamic states
* ``ReplicaExchangeSampler``: Replica exchange among thermodynamic states (also called Hamiltonian exchange if only the Hamiltonian is changing)
* ``SAMSSampler``: Self-adjusted mixture sampling (also known as optimally-adjusted mixture sampling)

The ``{MCMCName}`` denotes the name of an MCMC scheme block to be used to update the replicas at fixed thermodynamic state.
By default, this uses ``LangevinIntegrator`` with default splitting scheme (g-BAOAB [CITE]).

The ``{NumberOfIterations}`` is a nonnegative integer that denotes the maximum number of iterations to be run.

``{SamplerOptions}`` denotes an optional set of sampler-specific options that can individually be specified if user desires to override defaults.

``MultiStateSampler`` options
-----------------------------

The ``MultiStateSampler`` carries out independent simulations at multiple distinct thermodynamic states.

.. code-block:: yaml

    samplers:
        {UserDefinedSamplerName}:
            type: MultiStateSampler
            mcmc_moves: {MCMCName}
            number_of_iterations: {NumberOfIterations}
            locality: {Locality}

The ``{Locality}`` option defaults to global locality.
If the user desires the states at which energies are to be evaluated should be restricted to a neighborhood ``[k-locality, k+locality]`` around the current state ``k``, an integer can be specified.

.. todo::

   Later, we want to allow more complex neighborhoods to be specified via lists of lists.

``ReplicaExchangeSampler`` options
----------------------------------

The ``ReplicaExchangeSampler`` carries out simulations at multiple thermodynamic states, allowing pairs of replica to periodically exchange thermodynamic states.

.. code-block:: yaml

    samplers:
        {UserDefinedSamplerName}:
            type: ReplicaExchangeSampler
            mcmc_moves: {MCMCName}
            replica_mixing_scheme: {ReplicaMixingScheme}

A simple example:

.. code-block:: yaml

    samplers:
        replica-exchange:
            type: ReplicaExchangeSampler
            mcmc_moves: langevin
            replica_mixing_scheme: gibbs

``SAMSSampler`` options
-----------------------

Like ``ReplicaExchangeSampler``, the ``SAMSSampler`` carries out simulations at one or more thermodynamic states, but state updates are performed independently, which can allow for more rapid exploration of the entire set of thermodynamic states.
If multiple replicas are used, all replicas contribute to the update of the log weights for each state, in principle accelerating convergence at a rate proportional to the number of replicas.

.. todo ::

   Provide a way to specify multiple replicas.

.. code-block:: yaml

    samplers:
        {UserDefinedSamplerName}:
            type: SAMSSampler
            mcmc_moves: {MCMCName}
            state_update_scheme: {JumpScheme}
            gamma0: {GammaValue}
            flatness_threshold: {FlatnessThreshold}
            log_target_probabilities: {LogTargetProbabilities}

Several ``{JumpScheme}`` state update schemes are available:
* ``global-jump`` (default): The sampler can jump to any thermodynamic state (RECOMMENDED)
* ``restricted-range-jump``: The sampler can jump to any thermodynamic state within the specified local neighborhood (EXPERIMENTAL)
* ``local-jump``: Only proposals within the specified neighborhood are considered, but rejection rates may be high

``{GammaValue}`` controls the rate at which the initial heuristic stage accumulates log weight, and defaults to 1.0.

``{FlatnessThreshold}`` controls the fractional log weight that must be accumulated for each thermoydnamic state before the weight adjustment scheme switches from the initial heuristic adjustment scheme to the asymptotically optimal scheme.

By default the log target probabilities are all equal, resulting in SAMS attempting to adjust the log weights to equally sample all thermodynamic states.

A simple example:

.. code-block:: yaml

    samplers:
        sams:
            type: SAMSSampler
            mcmc_moves: langevin
            state_update_scheme: global-jump
            flatness_threshold: 2.0
            number_of_iterations: 10000
            gamma0: 10.0
