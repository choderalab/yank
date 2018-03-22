.. _yaml_mcmc_head:

MCMC Moves Header for YAML Files
********************************

In the ``mcmc_moves`` section, you can define the propagators used to move forward the replicas of a
:doc:`sampler <samplers>`. This block is optional. By default, YANK uses a sequence of MCMC moves that mix MC rigid
rotation and translation of the ligand with Langevin dynamics.

----


.. _yaml_mcmc_example:

MCMC Moves Syntax
=================
MCMC moves definitions must adhere to the following syntax

.. code-block:: yaml

    mcmc_moves:
        {UserDefinedMCMCMoveName1}:
            type: {MCMCMove}
            {MCMCMoveOptions}
        {UserDefinedMCMCMoveName2}:
            type: {MCMCMOve}
            {MCMCMoveOptions}
        ...

where ``{UserDefinedMCMCMoveName}`` is a unique string identifier within the ``mcmc_moves`` block. The only mandatory
keyword is ``type``. YANK supports all the ``MCMCMove`` classes defined in the
`openmmtools.mcmc <http://openmmtools.readthedocs.io/en/latest/mcmc.html#mcmc-move-types>`_ module, which currently are:

* ``SequenceMove``: Container MCMC move describing a sequence of other moves.
* ``LangevinDynamicsMove``
* ``LangevinSplittingDynamicsMove``
* ``GHMCMove``
* ``HMCMove``
* ``MonteCarloBarostatMove``
* ``MCDisplacementMove``
* ``MCRotationMove``

Each move accepts a specific set of options (see the constructor documentation of the classes in the
`openmmtools.mcmc <http://openmmtools.readthedocs.io/en/latest/mcmc.html#mcmc-move-types>`_ module. The default moves
used by YANK are equivalent to the following.

.. code-block:: yaml

    mcmc_moves:
        default1:
            type: LangevinSplittingDynamicsMove
            timestep: 2.0*femtoseconds,
            collision_rate: 1.0 / picosecond,
            n_steps: 500,
            reassign_velocities: yes,
            n_restart_attempts: 6,
            splitting: 'VRORV'
        default2:
            type: SequenceMove
            move_list:
                - type: MCDisplacementMove
                - type: MCRotationMove
                - type: LangevinSplittingDynamicsMove
                  timestep: 2.0*femtoseconds,
                  collision_rate: 1.0 / picosecond,
                  n_steps: 500,
                  reassign_velocities: yes,
                  n_restart_attempts: 6,
                  splitting: 'VRORV'


``default1`` is used for the solvent phase and for complex phases using a ``Boresch`` restraint. For complex phases
using any other restraint, ``default2`` is used.

Each iteration of the sampler applies the MCMC move once. In ``default2`` example above, one iteration of the algorithm
consists of one MC ligand rigid translation, followed by one MC ligand rigid rotation, and 500 steps of Langevin
dynamics.
