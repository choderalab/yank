.. _yaml_samplers_head:

Samplers Header for YAML Files
*******************************

The ``samplers`` section tells YANK how it should sample multiple thermodynamic states in order to estimate free
energies between states of interest.
Together with the :doc:`mcmc <mcmc>` section, this section provides a flexible way to control how
thermodynamic states are efficiently sampled.

This block is fully optional for those who do not wish to fiddle with such settings and to
support backwards compatible YAML scripts. If this no ``sampler`` is given to ``experiment``, then a DEFAULT one is
used.

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

* :ref:`MultistateSampler <yaml_samplers_multistatesampler>`: Independent simulations at distinct thermodynamic states
* :ref:`ReplicaExchangeSampler <yaml_samplers_repexsampler>`: Replica exchange among thermodynamic states (also called Hamiltonian exchange if only the Hamiltonian is changing)
* :ref:`SAMSSampler <yaml_samplers_samssampler>`: Self-adjusted mixture sampling (also known as optimally-adjusted mixture sampling)

The above block is the **minimum syntax** needed for any definition for any of the options.

.. todo::

    Add link to the MCMC block when docs are ready.

The ``{MCMCName}`` denotes the name of an MCMC scheme block to be used to update the replicas at fixed thermodynamic state.
By default, this uses ``LangevinIntegrator`` with default splitting scheme (g-BAOAB [CITE]).

The ``{NumberOfIterations}`` is a non-negative integer that denotes the maximum number of iterations to be run.
When this block is not given and :ref:`default_number_of_iterations <yaml_options_default_number_of_iterations>` is set
in the main :ref:`yaml-options-head` block, then that number is used instead. See that description for more information

``{SamplerOptions}`` denotes an optional set of sampler-specific options that can individually be specified if user
desires to override defaults. The ``MultiStateSampler`` is the base class for all other samples and options which
can be provided to it are global to all other samplers. Below the ``MultiStateSampler`` are the individual options
available for every other choice of ``type``.


.. _yaml_samplers_multistatesampler:

.. rst-class:: html-toggle

``MultiStateSampler`` options
-----------------------------

The ``MultiStateSampler`` carries out independent simulations at multiple distinct thermodynamic states.

.. code-block:: yaml

    samplers:
        {UserDefinedSamplerName}:
            type: MultiStateSampler
            mcmc_moves: {MCMCName}
            number_of_iterations: {NumberOfIterations}
            {MoreGlobalSamplerOptions}

The ``{MoreGlobalSamplerOptions}`` are detailed below and are valid for any other ``type`` of Sampler.


.. _yaml_samplers_locality:

.. rst-class:: html-toggle

``locality``
""""""""""""

.. code-block:: yaml

   samplers:
        {UserDefinedSamplerName}:
            type: MultiStateSampler
            mcmc_moves: {MCMCName}
            number_of_iterations: {NumberOfIterations}
            locality: {Locality}

Specify the number of states around the sampled state to compute energies between.

By default this is set to ``null`` for global locality and all samples are computed in all states.

If the user desires the states at which energies are to be evaluated should be restricted to a neighborhood
``[k-locality, k+locality]`` around the current state ``k``, an integer can be specified. This is a non-wrapping
locality; e.g. For 10 states, State 0 (first state) with a ``locality: 2`` will include states ``1`` and ``2`` but
NOT ``9`` and ``8``. If ``locality`` is greater than or equal to the number of states, then the behavior is the same
as ``null``.

Valid Options: [``null``]/``int`` > 0

.. todo::

   Later, we want to allow more complex neighborhoods to be specified via lists of lists.



.. _yaml_samplers_online_analysis_parameters:

.. rst-class:: html-toggle

Online Analysis Parameters
""""""""""""""""""""""""""

YANK's samplers also supports an online free energy analysis framework which allows running simulations up to some
target error in the free energy. Note that this will pause the simulation to run this analysis. The longer the
simulation gets, the slower this process becomes.


.. _yaml_samplers_online_analysis_interval:

.. rst-class:: html-toggle

``online_analysis_interval``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: yaml

   samplers:
        {UserDefinedSamplerName}:
            type: MultiStateSampler
            mcmc_moves: {MCMCName}
            number_of_iterations: {NumberOfIterations}
            online_analysis_interval: 100

Both the toggle and iteration count between online analysis operations. Every interval, the Multistate Bennet Acceptance
Ratio estimate for the free energy is calculated and the error is computed. Some data is preserved each iteration to
speed up future calculations, but this operation will still slow down as more iterations are added. We recommend
choosing an interval of *at least* 100, if not more.

If set to ``null`` (default), then online analysis is not run.

Valid Options (``null``): ``null`` or <Int >= 1>


.. rst-class:: html-toggle

.. _yaml_samplers_online_analysis_target_error:

``online_analysis_target_error``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: yaml

   samplers:
        {UserDefinedSamplerName}:
            type: MultiStateSampler
            mcmc_moves: {MCMCName}
            number_of_iterations: {NumberOfIterations}
            online_analysis_target_error: 1.0

The target error for the online analysis measured in kT per phase. Once the free energy is at or below this value,
the phase will be considered complete.
This value should be a number greater than 0, even though 0 is a valid option. The error free energy estimate between states
is never zero except in very rare cases, so your simulation may never converge if you set this to 0.

If :ref:`yaml_samplers_online_analysis_interval` is ``null``, this option does nothing.

Valid Options (0.2): <Float >= 0>



.. _yaml_samplers_online_analysis_minimum_iterations:

.. rst-class:: html-toggle

``online_analysis_minimum_iterations``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: yaml

   samplers:
        {UserDefinedSamplerName}:
            type: MultiStateSampler
            mcmc_moves: {MCMCName}
            number_of_iterations: {NumberOfIterations}
            online_analysis_minimum_iterations: 50

Number of iterations that are skipped at the beginning of the simulation before online analysis is attempted. This is
a speed option since most of the initial iterations will be either equilibration or under sampled. We recommend choosing
an initial number that is *at least* one or two :ref:`yaml_samplers_online_analysis_interval`'s for speed's sake.

The first iteration at which online analysis is performed is not affected by this number and always tracked as the
modulo of the current iteration. E.g. if you have ``online_analysis_interval: 100`` and
``online_analysis_minimum_iterations: 150``, online analysis would happen at iteration 200 first, not iteration 250.

If :ref:`yaml_samplers_online_analysis_interval` is ``null``, this option does nothing.

Valid Options (50): <Int >=1>



|
|

.. _yaml_samplers_repexsampler:

.. rst-class:: html-toggle

``ReplicaExchangeSampler`` options
----------------------------------

The ``ReplicaExchangeSampler`` carries out simulations at multiple thermodynamic states, allowing pairs of replica to
periodically exchange thermodynamic states. If :ref:`yaml_samplers_locality` is specified (i.e. not ``null``), then
:ref:`yaml_samplers_replica_mixing_scheme` must be ``swap-neighbors``.

with this scheme, you must use
in replica exchange because there exists
one replica per thermodynamic state, and global locality is required for replica exchange to work.

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
            replica_mixing_scheme: swap-all


.. _yaml_samplers_replica_mixing_scheme:

.. rst-class:: html-toggle

``replica_mixing_scheme``
"""""""""""""""""""""""""

.. code-block:: yaml

   options:
     replica_mixing_scheme: swap-all

Specifies how the Hamiltonian Replica Exchange attempts swaps between replicas.
``swap-all`` will attempt to exchange every state with every other state. ``swap-neighbors``  will attempt only
exchanges between adjacent states. If ``null`` is specified, no mixing is done, and effectively disables all replica
exchange functionality.

Valid Options: [swap-all]/swap-neighbors/null


|
|

.. _yaml_samplers_samssampler:

.. rst-class:: html-toggle

``SAMSSampler`` options
-----------------------

Like ``ReplicaExchangeSampler``, the ``SAMSSampler`` carries out simulations at one or more thermodynamic states, but
state updates are performed independently, which can allow for more rapid exploration of the entire set of thermodynamic
states.
If multiple replicas are used, all replicas contribute to the update of the log weights for each state, in principle
accelerating convergence at a rate proportional to the number of replicas.

Many of the default options for this sampler should be considered acceptable and you should not need to manually set
them, however, the ability to do so is present.

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


.. _yaml_samplers_state_update_scheme:

.. rst-class:: html-toggle

``state_update_scheme``
"""""""""""""""""""""""

.. code-block:: yaml

    samplers:
        sams:
            type: SAMSSampler
            mcmc_moves: langevin
            state_update_scheme: global-jump

The scheme of how SAMS chooses to jump between sampled thermodynamic states, the behavior depends on which scheme
is chosen:

* ``global-jump`` (default): The sampler can jump to any thermodynamic state (RECOMMENDED)
* ``restricted-range-jump``: The sampler can jump to any thermodynamic state within the specified local neighborhood (EXPERIMENTAL)
* ``local-jump``: Only proposals within the specified neighborhood are considered, but rejection rates may be high

Valid Options: ``global-jump`` (Others are experimental and disabled for now)


.. _yaml_samplers_gamm0:

.. rst-class:: html-toggle

``gamma0``
""""""""""

.. code-block:: yaml

    samplers:
        sams:
            type: SAMSSampler
            mcmc_moves: langevin
            gamma0: 1.0

Controls the rate at which the initial heuristic stage accumulates log weight

Valid Options (1.0): float > 0


.. _yaml_samplers_flatness_threshold:

.. rst-class:: html-toggle

``flatness_threshold``
""""""""""""""""""""""

.. code-block:: yaml

    samplers:
        sams:
            type: SAMSSampler
            mcmc_moves: langevin
            flatness_threshold: 0.2


Controls the fractional log weight that must be accumulated for each thermodynamic state before the weight adjustment
scheme switches from the initial heuristic adjustment scheme to the asymptotically optimal scheme.

By default the log target probabilities are all equal, resulting in SAMS attempting to adjust the log weights to equally
sample all thermodynamic states.

Valid Options (0.2): float > 0

