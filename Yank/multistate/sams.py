#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
SamsSampler
===========

Self-adjusted mixture sampling (SAMS), also known as optimally-adjusted mixture sampling.

This implementation uses stochastic approximation to allow one or more replicas to sample the whole range of thermodynamic states
for rapid online computation of free energies.

COPYRIGHT

Written by John D. Chodera <john.chodera@choderalab.org> while at Memorial Sloan Kettering Cancer Center.

LICENSE

This code is licensed under the latest available version of the MIT License.

"""

import copy
import math
import logging
import numpy as np
import openmmtools as mmtools

from .. import mpi
from .multistatesampler import MultiStateSampler
from .multistatereporter import MultiStateReporter
from .multistateanalyzer import MultiStateSamplerAnalyzer

logger = logging.getLogger(__name__)


# ==============================================================================
# PARALLEL TEMPERING
# ==============================================================================

class SAMSSampler(MultiStateSampler):
    """Self-adjusted mixture sampling (SAMS), also known as optimally-adjusted mixture sampling.

    This class provides a facility for self-adjusted mixture sampling simulations.
    One or more replicas use the method of expanded ensembles [1] to sample multiple thermodynamic states within each replica,
    with log weights for each thermodynamic state adapted on the fly [2] to achieve the desired target probabilities for each state.

    Attributes
    ----------
    log_target_probabilities : array-like
        log_target_probabilities[state_index] is the log target probability for state ``state_index``
    update_scheme : str
        Thermodynamic state sampling scheme. One of ['global-jump', 'local-jump', 'restricted-range']
    locality : int
        Number of neighboring states on either side to consider for local update schemes
    update_stages : str
        Number of stages to use for update. One of ['one-stage', 'two-stage']
    update_method : str
        Method to use for updating log weights in SAMS. One of ['optimal', 'rao-blackwellized']
    adapt_target_probabilities : bool
            If True, target probabilities will be adapted to achieve minimal thermodynamic length between terminal thermodynamic states.

    References
    ----------
    [1] Lyubartsev AP, Martsinovski AA, Shevkunov SV, and Vorontsov-Velyaminov PN. New approach to Monte Carlo calculation of the free energy: Method of expanded ensembles. JCP 96:1776, 1992
    http://dx.doi.org/10.1063/1.462133

    [2] Tan, Z. Optimally adjusted mixture sampling and locally weighted histogram analysis, Journal of Computational and Graphical Statistics 26:54, 2017.
    http://dx.doi.org/10.1080/10618600.2015.1113975

    Examples
    --------
    SAMS simulation of alanine dipeptide in implicit solvent at different temperatures.

    Create the system:

    >>> import math
    >>> from simtk import unit
    >>> from openmmtools import testsystems, states, mcmc
    >>> testsystem = testsystems.AlanineDipeptideImplicit()

    Create thermodynamic states for parallel tempering with exponentially-spaced schedule:

    >>> n_replicas = 3  # Number of temperature replicas.
    >>> T_min = 298.0 * unit.kelvin  # Minimum temperature.
    >>> T_max = 600.0 * unit.kelvin  # Maximum temperature.
    >>> temperatures = [T_min + (T_max - T_min) * (math.exp(float(i) / float(nreplicas-1)) - 1.0) / (math.e - 1.0)
    ...                 for i in range(n_replicas)]
    >>> thermodynamic_states = [states.ThermodynamicState(system=testsystem.system, temperature=T)
    ...                         for T in temperatures]

    Initialize simulation object with options. Run with a GHMC integrator:

    >>> move = mcmc.GHMCMove(timestep=2.0*unit.femtoseconds, n_steps=50)
    >>> simulation = SAMSSampler(mcmc_moves=move, number_of_iterations=2,
    >>>                          update_scheme='restricted-range', locality=5,
    >>>                          update_stages='two-stage', flatness_threshold=0.2,
    >>>                          update_method='rao-blackwellized',
    >>>                          adapt_target_probabilities=False)


    Create a single-replica SAMS simulation bound to a storage file and run:

    >>> storage_path = tempfile.NamedTemporaryFile(delete=False).name + '.nc'
    >>> reporter = MultiStateReporter(storage_path, checkpoint_interval=1)
    >>> simulation.create(thermodynamic_states=thermodynamic_states,
    >>>                   sampler_states=states.SamplerState(testsystem.positions),
    >>>                   storage=reporter)
    >>> simulation.run()  # This runs for a maximum of 2 iterations.
    >>> simulation.iteration
    2
    >>> simulation.run(n_iterations=1)
    >>> simulation.iteration
    2

    To resume a simulation from an existing storage file and extend it beyond
    the original number of iterations.

    >>> del simulation
    >>> simulation = ReplicaExchangeSampler.from_storage(reporter)
    >>> simulation.extend(n_iterations=1)
    >>> simulation.iteration
    3

    You can extract several information from the NetCDF file using the Reporter
    class while the simulation is running. This reads the SamplerStates of every
    run iteration.

    >>> reporter = MultiStateReporter(storage=storage_path, open_mode='r', checkpoint_interval=1)
    >>> sampler_states = reporter.read_sampler_states(iteration=range(1, 4))
    >>> len(sampler_states)
    3
    >>> sampler_states[-1].positions.shape  # Alanine dipeptide has 22 atoms.
    (22, 3)

    Clean up.

    >>> os.remove(storage_path)

    See Also
    --------
    ReplicaExchangeSampler

    """

    _TITLE_TEMPLATE = ('Self-adjusted mixture sampling (SAMS) simultion using SAMSSampler '
                       'class of yank.multistate on {}')

    def __init__(self,
                log_target_probabilities=None,
                update_scheme='restricted-range', locality=5,
                update_stages='two-stage', flatness_threshold=0.2,
                update_method='rao-blackwellized',
                adapt_target_probabilities=False,
                **kwargs):
        """Initialize a SAMS sampler.

        Parameters
        ----------
        log_target_probabilities : array-like or None
            ``log_target_probabilities[state_index]`` is the log target probability for thermodynamic state ``state_index``
            When converged, each state should be sampled with the specified log probability.
            If None, uniform probabilities for all states will be assumed.
        update_scheme : str, optional, default='global-jump'
            Specifies the scheme used to sample new thermodynamic states given fixed sampler states.
            One of ['global-jump', 'local-jump', 'restricted-range']
            ``global_jump`` will allow the sampler to access any thermodynamic state
            ``local-jump`` will propose a move to one of the local neighborhood states, and accept or reject.
            ``restricted-range`` will compute the probabilities for each of the states in the local neighborhood, increasing jump probability
        locality : int, optional, default=1
            Number of neighboring states on either side to consider for local update schemes.
        update_stages : str, optional, default='two-stage'
            One of ['one-stage', 'two-stage']
            ``one-stage`` will use the asymptotically optimal scheme throughout the entire simulation (not recommended due to slow convergence)
            ``two-stage`` will use a heuristic first stage to achieve flat histograms before switching to the asymptotically optimal scheme
        flatness_threshold : float, optiona, default=0.2
            Histogram relative flatness threshold to use for first stage of two-stage scheme.
        update_method : str, optional, default='rao-blackwellized'
            Method to use for updating log weights in SAMS. One of ['optimal', 'rao-blackwellized']
            ``rao-blackwellized`` will update log free energy estimate for all states for which energies were computed
            ``optimal`` will use integral counts to update log free energy estimate of current state only
        adapt_target_probabilities : bool, optional, default=False
            If True, target probabilities will be adapted to achieve minimal thermodynamic length between terminal thermodynamic states.
            (EXPERIMENTAL)
        """
        # Initialize multi-state sampler
        super(SAMSSampler, self).__init__(**kwargs)
        self.log_target_probabilities = log_target_probabilities
        self.update_scheme = update_scheme
        self.locality = locality
        self.update_stages = update_stages
        self.flatness_threshold = flatness_threshold
        self.update_method = update_method
        self.adapt_target_probabilities = adapt_target_probabilities

    class _StoredProperty(MultiStateSampler._StoredProperty):

        @staticmethod
        def _update_scheme_validator(instance, scheme):
            supported_schemes = ['global-jump', 'local-jump', 'restricted-range']
            if scheme not in supported_schemes:
                raise ValueError("Unknown update scheme '{}'. Supported values "
                                 "are {}.".format(scheme, supported_schemes))
            return scheme

    log_target_probabilities = _StoredProperty('log_target_probabilities', validate_function=None)

    update_scheme = _StoredProperty('update_scheme', validate_function=_StoredProperty._update_scheme_validator)

    # TODO: Implement validate functions
    locality = _StoredProperty('locality', validate_function=None)
    update_stages = _StoredProperty('update_stages', validate_function=None)
    flatness_threshold = _StoredProperty('flatness_threshold', validate_function=None)
    update_method = _StoredProperty('update_method', validate_function=None)
    adapt_target_probabilities = _StoredProperty('adapt_target_probabilities', validate_function=None)

    def create(self, thermodynamic_states: list, sampler_states:list, storage,
               **kwargs):
        """Initialize SAMS sampler.

        Parameters
        ----------
        thermodynamic_states : list of openmmtools.states.ThermodynamicState
            Thermodynamic states to simulate, where one replica is allocated per state.
            Each state must have a system with the same number of atoms.
        sampler_states : list of openmmtools.states.SamplerState
            One or more sets of initial sampler states.
            The number of replicas is determined by the number of sampler states provided,
            and does not need to match the number of thermodynamic states.
            Most commonly, a single sampler state is provided.
        storage : str or Reporter
            If str: path to the storage file, checkpoint options are default
            If Reporter: Instanced :class:`Reporter` class, checkpoint information is read from
            In the future this will be able to take a Storage class as well.
        initial_thermodynamic_states : None or list or array-like of int of length len(sampler_states), optional,
            default: None.
            Initial thermodynamic_state index for each sampler_state.
            If no initial distribution is chosen, ``sampler_states`` are distributed between the
            ``thermodynamic_states`` following these rules:

                * If ``len(thermodynamic_states) == len(sampler_states)``: 1-to-1 distribution

                * If ``len(thermodynamic_states) > len(sampler_states)``: First and last state distributed first
                  remaining ``sampler_states`` spaced evenly by index until ``sampler_states`` are depleted.
                  If there is only one ``sampler_state``, then the only first ``thermodynamic_state`` will be chosen

                * If ``len(thermodynamic_states) < len(sampler_states)``, each ``thermodynamic_state`` receives an
                  equal number of ``sampler_states`` until there are insufficient number of ``sampler_states`` remaining
                  to give each ``thermodynamic_state`` an equal number. Then the rules from the previous point are
                  followed.
        metadata : dict, optional
           Simulation metadata to be stored in the file.
        """
        # TODO: Initialize SAMS-specific statistics

        # Initialize replica-exchange simulation.
        super(SAMSSampler, self).create(thermodynamic_states, sampler_states, storage=storage, **kwargs)


    # TODO: Implement SAMS-specific state and weight update schemes

class SAMSAnalyzer(MultiStateSamplerAnalyzer):
    """
    The SAMSAnalyzer is the analyzer for a simulation generated from a SAMSSampler simulation.

    See Also
    --------
    ReplicaExchangeAnalyzer
    PhaseAnalyzer

    """
    pass

# ==============================================================================
# MAIN AND TESTS
# ==============================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
