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
* ``LangevinSplittingDynamicsMove``: High-quality Langevin integrator family based on symmetric Strang splittings, using g-BAOAB :cite:`LeimkuhlerMatthews2016:g-BAOAB` as default
* ``LangevinDynamicsMove``: Leapfrog Langevin dynamics integrator from OpenMM (``simtk.openmm.LangevinIntegrator``); not recommended
* ``GHMCMove``: Metropolized Langevin integrator, when exact sampling is required; not recommended for large systems
* ``HMCMove``: Hybrid Monte Carlo integrator, when exact sampling is required; not recommended for large systems
* ``MonteCarloBarostatMove``: Explicit Monte Carlo barostat; not recommended, since this can be incorporated within Langevin dynamics moves instead
* ``MCDisplacementMove``: Monte Carlo displacement of ligand; useful for fragment-sized ligands in implicit solvent
* ``MCRotationMove``: Monte Carlo rotation of ligand; useful for fragment-sized ligands in implicit solvent

Each move accepts a specific set of options (see the constructor documentation of the classes in the `openmmtools.mcmc <http://openmmtools.readthedocs.io/en/latest/mcmc.html#mcmc-move-types>`_ module.
The default moves used by YANK are equivalent to the following:

.. _yaml_mcmc_defaults:

.. code-block:: yaml

    mcmc_moves:
        default1:
            type: LangevinSplittingDynamicsMove
            timestep: 2.0*femtoseconds, # 2 fs timestep
            collision_rate: 1.0 / picosecond, # weak collision rate
            n_steps: 500, # 500 steps/iteration
            reassign_velocities: yes, # reassign Maxwell-Boltzmann velocities each iteration
            n_restart_attempts: 6, # attempt to recover from NaNs
            splitting: 'VRORV' # use the high-quality BAOAB integrator
        default2:
            type: SequenceMove
            move_list:
                - type: MCDisplacementMove # Monte Carlo ligand displacement
                - type: MCRotationMove # Monte Carlo ligand rotation
                - type: LangevinSplittingDynamicsMove
                  timestep: 2.0*femtoseconds, # 2 fs timestep
                  collision_rate: 1.0 / picosecond, # weak collision rate
                  n_steps: 500, # 500 steps/iteration
                  reassign_velocities: yes, # reassign Maxwell-Boltzmann velocities each iteration
                  n_restart_attempts: 6, # attempt to recover from NaNs
                  splitting: 'VRORV' # use the high-quality BAOAB integrator


``default1`` is used for the solvent phase and for complex phases using a ``Boresch`` restraint.
For complex phases using any other restraint, ``default2`` is used.

Each iteration of the sampler applies the MCMC move once.
In ``default2`` example above, one iteration of the algorithm consists of one MC ligand rigid translation, followed by one MC ligand rigid rotation, and 500 steps of Langevin dynamics.
