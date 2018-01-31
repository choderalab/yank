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
from scipy.misc import logsumexp

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
    state_update_scheme : str
        Thermodynamic state sampling scheme. One of ['global-jump', 'local-jump', 'restricted-range']
    locality : int
        Number of neighboring states on either side to consider for local update schemes
    update_stages : str
        Number of stages to use for update. One of ['one-stage', 'two-stage']
    weight_update_method : str
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
    >>>                          state_update_scheme='restricted-range', locality=5,
    >>>                          update_stages='two-stage', flatness_threshold=0.2,
    >>>                          weight_update_method='rao-blackwellized',
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
                state_update_scheme='restricted-range', locality=5,
                update_stages='two-stage', flatness_threshold=0.2,
                weight_update_method='rao-blackwellized',
                adapt_target_probabilities=False,
                **kwargs):
        """Initialize a SAMS sampler.

        Parameters
        ----------
        log_target_probabilities : array-like or None
            ``log_target_probabilities[state_index]`` is the log target probability for thermodynamic state ``state_index``
            When converged, each state should be sampled with the specified log probability.
            If None, uniform probabilities for all states will be assumed.
        state_update_scheme : str, optional, default='global-jump'
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
        weight_update_method : str, optional, default='rao-blackwellized'
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
        self.state_update_scheme = state_update_scheme
        self.locality = locality
        self.update_stages = update_stages
        self.flatness_threshold = flatness_threshold
        self.weight_update_method = weight_update_method
        self.adapt_target_probabilities = adapt_target_probabilities

    class _StoredProperty(MultiStateSampler._StoredProperty):

        @staticmethod
        def _state_update_scheme_validator(instance, scheme):
            supported_schemes = ['global-jump', 'local-jump', 'restricted-range']
            if scheme not in supported_schemes:
                raise ValueError("Unknown update scheme '{}'. Supported values "
                                 "are {}.".format(scheme, supported_schemes))
            return scheme

        @staticmethod
        def _update_stages_validator(instance, scheme):
            supported_schemes = ['one-stage', 'two-stage']
            if scheme not in supported_schemes:
                raise ValueError("Unknown update scheme '{}'. Supported values "
                                 "are {}.".format(scheme, supported_schemes))
            return scheme

        @staticmethod
        def _weight_update_method_validator(instance, scheme):
            supported_schemes = ['optimal', 'rao-blackwellized']
            if scheme not in supported_schemes:
                raise ValueError("Unknown update scheme '{}'. Supported values "
                                 "are {}.".format(scheme, supported_schemes))
            return scheme

        @staticmethod
        def _adapt_target_probabilities_validator(instance, scheme):
            supported_schemes = [True, False]
            if scheme not in supported_schemes:
                raise ValueError("Unknown update scheme '{}'. Supported values "
                                 "are {}.".format(scheme, supported_schemes))
            return scheme

    log_target_probabilities = _StoredProperty('log_target_probabilities', validate_function=None)
    state_update_scheme = _StoredProperty('state_update_scheme', validate_function=_StoredProperty._state_update_scheme_validator)
    locality = _StoredProperty('locality', validate_function=None)
    update_stages = _StoredProperty('update_stages', validate_function=_StoredProperty._update_stages_validator)
    flatness_threshold = _StoredProperty('flatness_threshold', validate_function=None)
    weight_update_method = _StoredProperty('weight_update_method', validate_function=_StoredProperty._weight_update_method_validator)
    adapt_target_probabilities = _StoredProperty('adapt_target_probabilities', validate_function=_StoredProperty._adapt_target_probabilities_validator)

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
        # Initialize replica-exchange simulation.
        super(SAMSSampler, self).create(thermodynamic_states, sampler_states, storage=storage, **kwargs)

    # TODO: Implement SAMS-specific state and weight update schemes

    @mpi.on_single_node(0, broadcast_result=True)
    def _mix_replicas(self):
        """Update thermodynamic states according to user-specified scheme."""
        logger.debug("Updating thermodynamic states...")

        # Reset storage to keep track of swap attempts this iteration.
        self._n_accepted_matrix[:, :] = 0
        self._n_proposed_matrix[:, :] = 0

        # Perform swap attempts according to requested scheme.
        with mmtools.utils.time_it('Mixing of replicas'):
            if self.state_update_scheme == 'global-jump':
                pass
            elif self.state_update_scheme == 'local-jump':
                pass
            elif self.state_update_scheme == 'restricted-range':
                pass
            else:
                assert self.replica_mixing_scheme is None

        # Determine fraction of swaps accepted this iteration.
        n_swaps_proposed = self._n_proposed_matrix.sum()
        n_swaps_accepted = self._n_accepted_matrix.sum()
        swap_fraction_accepted = 0.0
        if n_swaps_proposed > 0:
            # TODO drop casting to float when dropping Python 2 support.
            swap_fraction_accepted = float(n_swaps_accepted) / n_swaps_proposed
        logger.debug("Accepted {}/{} attempted swaps ({:.1f}%)".format(n_swaps_accepted, n_swaps_proposed,
                                                                       swap_fraction_accepted * 100.0))

        # Return new states indices for MPI broadcasting.
        return self._replica_thermodynamic_states

    def _local_jump(self):
        n_replica, n_states, locality = self.n_replicas, self.n_states, self.locality
        for (replica_index, current_state_index) in enumerate(self._replica_thermodynamic_states):
            u_k = np.zeros([n_states], np.float64)
            log_P_k = np.zeros([n_states], np.float64)
            # Determine current neighborhood.
            neighborhood = range(max(0, current_state_index - locality), min(n_states, current_state_index + locality + 1))
            neighborhood_size = len(neighborhood)
            # Propose a move from the current neighborhood.
            proposed_state_index = np.random.choice(neighborhood, p=np.ones([len(neighborhood)], np.float64) / float(neighborhood_size))
            # Determine neighborhood for proposed state.
            proposed_neighborhood = range(max(0, proposed_state_index - locality), min(n_states, proposed_state_index + locality + 1))
            proposed_neighborhood_size = len(proposed_neighborhood)
            # Compute state log weights.
            log_Gamma_j_L = - float(proposed_neighborhood_size) # log probability of proposing return
            log_Gamma_L_j = - float(neighborhood_size)          # log probability of proposing new state
            L = current_state_index
            # Compute potential for all states in neighborhood
            for j in neighborhood:
                u_k[j] = self._energy_thermodynamic_states[replica_index, j]
            # Compute log of probability of selecting each state in neighborhood
            for j in neighborhood:
                if j != L:
                    log_P_k[j] = log_Gamma_L_j + min(0.0, log_Gamma_j_L - log_Gamma_L_j + (self.log_weights[j] - u_k[j]) - (self.log_weights[L] - u_k[L]))
            P_k = np.zeros([n_states], np.float64)
            P_k[neighborhood] = np.exp(log_P_k[neighborhood])
            # Compute probability to return to current state L
            P_k[L] = 0.0
            P_k[L] = 1.0 - P_k[neighborhood].sum()
            log_P_k[L] = np.log(P_k[L])
            # Update context.
            thermodynamic_state_index = np.random.choice(neighborhood, p=P_k[neighborhood])
            self._replica_thermodynamic_states[replica_index] = thermodynamic_state

            # TODO: Update log weights

    def _global_jump(self):
        """
        Global jump scheme.
        This method is described after Eq. 3 in [2]
        """
        n_replica, n_states = self.n_replicas, self.n_states
        for (replica_index, current_state_index) in enumerate(self._thermodynamic_states):
            u_k = np.zeros([n_states], np.float64)
            log_P_k = np.zeros([n_states], np.float64)
            # Compute unnormalized log probabilities for all thermodynamic states
            neighborhood = range(n_states)
            for state_index in neighborhood:
                u_k[state_index] = self._energy_thermodynamic_states[replica_index, state_index]
                log_P_k[state_index] = self.log_weights[state_index] - u_k[state_index]
            log_P_k -= logsumexp(log_P_k)
            # Update sampler Context to current thermodynamic state.
            P_k = np.exp(log_P_k[neighborhood])
            thermodynamic_state_index = np.random.choice(neighborhood, p=P_k)
            self._replica_thermodynamic_states[replica_index] = thermodynamic_state_index

        # TODO: Update log weights

    def _restricted_range(self):
        n_replica, n_states, locality = self.n_replicas, self.n_states, self.locality
        for (replica_index, current_state_index) in enumerate(self._thermodynamic_states):
            u_k = np.zeros([n_states], np.float64)
            log_P_k = np.zeros([n_states], np.float64)
            # Propose new state from current neighborhood.
            neighborhood = range(max(0, current_state_index - locality), min(n_states, current_state_index + locality + 1))
            for j in neighborhood:
                u_k[j] = self._energy_thermodynamic_states[replica_index, j]
                log_P_k[j] = self.log_weights[j] - u_k[j]
            log_P_k[neighborhood] -= logsumexp(log_P_k[neighborhood])
            P_k = np.exp(log_P_k[neighborhood])
            proposed_state_index = np.random.choice(neighborhood, p=P_k)
            # Determine neighborhood of proposed state.
            proposed_neighborhood = range(max(0, proposed_state_index - locality), min(n_states, proposed_state_index + locality + 1))
            for j in proposed_neighborhood:
                if j not in neighborhood:
                    u_k[j] = self._energy_thermodynamic_states[replica_index, j]
            # Accept or reject.
            log_P_accept = logsumexp(self.log_weights[neighborhood] - u_k[neighborhood]) - logsumexp(self.log_weights[proposed_neighborhood] - u_k[proposed_neighborhood])
            if (log_P_accept >= 0.0) or (np.random.rand() < np.exp(log_P_accept)):
                thermodynamic_state_index = proposed_state_index
            self._replica_thermodynamic_states[replica_index] = thermodynamic_state_index

            # TODO: Update log weights

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
