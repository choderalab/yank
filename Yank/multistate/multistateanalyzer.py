#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
MultiStateAnalyzers
===================

Analysis tools and module for MultiStateSampler simulations. Provides programmatic and automatic
"best practices" integration to determine free energy and other observables.

Fully extensible to support new samplers and observables.


"""

# =============================================================================================
# MODULE IMPORTS
# =============================================================================================

import abc
import copy
import logging
import re
from typing import Optional, NamedTuple, Union

import numpy as np
import simtk.unit as units
from scipy.misc import logsumexp
from pymbar import MBAR, timeseries


from . import utils

ABC = abc.ABC
logger = logging.getLogger(__name__)

__all__ = [
    'PhaseAnalyzer',
    'MultiStateSamplerAnalyzer',
    'MultiPhaseAnalyzer',
    'ObservablesRegistry',
    'default_observables_registry'
]

# =============================================================================================
# PARAMETERS
# =============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA


# =============================================================================================
# MODULE CLASSES
# =============================================================================================

class ObservablesRegistry(object):
    """
    Registry of computable observables.

    This is a class accessed by the :class:`PhaseAnalyzer` objects to check
    which observables can be computed, and then provide a regular categorization of them.

    This registry is a required linked component of any PhaseAnalyzer and especially of the MultiPhaseAnalyzer.
    This is not an internal class to the PhaseAnalyzer however because it can be instanced, extended, and customized
    as part of the API for this module.

    To define your own methods:
    1) Choose a unique observable name.
    2) Categorize the observable in one of the following ways by adding to the list in the "observables_X" method:

        2a) "defined_by_phase":
            Depends on the Phase as a whole (state independent)

        2b) "defined_by_single_state":
            Computed entirely from one state, e.g. Radius of Gyration

        2c) "defined_by_two_states":
            Property is relative to some reference state, such as Free Energy Difference

    3) Optionally categorize the error category calculation in the "observables_with_error_adding_Y" methods
       If not placed in an error category, the observable will be assumed not to carry error
       Examples: A, B, C are the observable in 3 phases, eA, eB, eC are the error of the observable in each phase

        3a) "linear": Error between phases adds linearly.
            If C = A + B, eC = eA + eB

        3b) "quadrature": Error between phases adds in the square.
            If C = A + B, eC = sqrt(eA^2 + eB^2)

    4) Finally, to add this observable to the phase, implement a "get_{method name}" method to the subclass of
       :class:`YankPhaseAnalyzer`. Any :class:`MultiPhaseAnalyzer` composed of this phase will automatically have the
       "get_{method name}" if all other phases in the :class:`MultiPhaseAnalyzer` have the same method.
    """

    def __init__(self):
        """Register Defaults"""
        # Create empty registry
        self._observables = {'two_state': set(),
                             'one_state': set(),
                             'phase': set()}
        self._errors = {'quad': set(),
                        'linear': set(),
                        None: set()}

    def register_two_state_observable(self, name: str,
                                      error_class: Optional[str]=None,
                                      re_register: bool=False):
        """
        Register a new two state observable, or re-register an existing one.

        Parameters
        ----------
        name: str
            Name of the observable, will be cast to all lower case and spaces replaced with underscores
        error_class: "quad", "linear", or None
            How the error of the observable is computed when added with other errors from the same observable.

            * "quad": Adds in the quadrature, Observable C = A + B, Error eC = sqrt(eA**2 + eB**2)

            * "linear": Adds linearly,  Observable C = A + B, Error eC = eA + eB

            * None: Does not carry error

        re_register: bool, optional, Default: False
            Re-register an existing observable
        """

        self._register_observable(name, "two_state", error_class, re_register=re_register)

    def register_one_state_observable(self, name: str,
                                      error_class: Optional[str]=None,
                                      re_register: bool=False):
        """
        Register a new one state observable, or re-register an existing one.

        Parameters
        ----------
        name: str
            Name of the observable, will be cast to all lower case and spaces replaced with underscores
        error_class: "quad", "linear", or None
            How the error of the observable is computed when added with other errors from the same observable.

            * "quad": Adds in the quadrature, Observable C = A + B, Error eC = sqrt(eA**2 + eB**2)

            * "linear": Adds linearly,  Observable C = A + B, Error eC = eA + eB

            * None: Does not carry error

        re_register: bool, optional, Default: False
            Re-register an existing observable
        """

        self._register_observable(name, "one_state", error_class, re_register=re_register)

    def register_phase_observable(self, name: str,
                                  error_class: Optional[str]=None,
                                  re_register: bool=False):
        """
        Register a new observable defined by phaee, or re-register an existing one.

        Parameters
        ----------
        name: str
            Name of the observable, will be cast to all lower case and spaces replaced with underscores
        error_class: 'quad', 'linear', or None
            How the error of the observable is computed when added with other errors from the same observable.

            * 'quad': Adds in the quadrature, Observable C = A + B, Error eC = sqrt(eA**2 + eB**2)

            * 'linear': Adds linearly,  Observable C = A + B, Error eC = eA + eB

            * None: Does not carry error

        re_register: bool, optional, Default: False
            Re-register an existing observable

        """

        self._register_observable(name, "phase", error_class, re_register=re_register)

    ########################
    # Define the observables
    ########################
    @property
    def observables(self):
        """
        Set of observables which are derived from the subsets below
        """
        observables = set()
        for subset_key in self._observables:
            observables |= self._observables[subset_key]
        return tuple(observables)

    # ------------------------------------------------
    # Exclusive Observable categories
    # The intersection of these should be the null set
    # ------------------------------------------------

    @property
    def observables_defined_by_two_states(self):
        """
        Observables that require an i and a j state to define the observable accurately between phases
        """
        return self._get_observables('two_state')

    @property
    def observables_defined_by_single_state(self):
        """
        Defined observables which are fully defined by a single state, and not by multiple states such as differences
        """
        return self._get_observables('one_state')

    @property
    def observables_defined_by_phase(self):
        """
        Observables which are defined by the phase as a whole, and not defined by any 1 or more states
        e.g. Standard State Correction
        """
        return self._get_observables('phase')

    ##########################################
    # Define the observables which carry error
    # This should be a subset of observables
    ##########################################

    @property
    def observables_with_error(self):
        """Determine which observables have error by inspecting the the error subsets"""
        observables = set()
        for subset_key in self._errors:
            if subset_key is not None:
                observables |= self._errors[subset_key]
        return tuple(observables)

    # ------------------------------------------------
    # Exclusive Error categories
    # The intersection of these should be the null set
    # ------------------------------------------------

    @property
    def observables_with_error_adding_quadrature(self):
        """Observable C = A + B, Error eC = sqrt(eA**2 + eB**2)"""
        return self._get_errors('quad')

    @property
    def observables_with_error_adding_linear(self):
        """Observable C = A + B, Error eC = eA + eB"""
        return self._get_errors('linear')

    @property
    def observables_without_error(self):
        return self._get_errors(None)

    # ------------------
    # Internal functions
    # ------------------

    def _get_observables(self, key):
        return tuple(self._observables[key])

    def _get_errors(self, key):
        return tuple(self._errors[key])

    @staticmethod
    def _cast_observable_name(name) -> str:
        return re.sub(" +", "_", name.lower())

    def _register_observable(self, obs_name: str,
                             obs_calc_class: str,
                             obs_error_class: Union[None, str],
                             re_register: bool=False):
        obs_name = self._cast_observable_name(obs_name)
        if not re_register and obs_name in self.observables:
            raise ValueError("{} is already a registered observable! "
                             "Consider setting re_register key!".format(obs_name))
        self._check_obs_class(obs_calc_class)
        self._check_obs_error_class(obs_error_class)
        obs_name_set = {obs_name}  # set(single_object) throws an error, set(string) splits each char
        # Throw out existing observable if present (set difference)
        for obs_key in self._observables:
            self._observables[obs_key] -= obs_name_set
        for obs_err_key in self._errors:
            self._errors[obs_err_key] -= obs_name_set
        # Add new observable to correct classifiers (set union)
        self._observables[obs_calc_class] |= obs_name_set
        self._errors[obs_error_class] |= obs_name_set

    def _check_obs_class(self, obs_class):
        assert obs_class in self._observables, "{} not a known observable class!".format(obs_class)

    def _check_obs_error_class(self, obs_error):
        assert obs_error is None or obs_error in self._errors, \
            "{} not a known observable error class!".format(obs_error)


# Create a default registry and register some stock values
default_observables_registry = ObservablesRegistry()
default_observables_registry.register_two_state_observable('free_energy', error_class='quad')
default_observables_registry.register_two_state_observable('entropy', error_class='quad')
default_observables_registry.register_two_state_observable('enthalpy', error_class='quad')


# ---------------------------------------------------------------------------------------------
# Phase Analyzers
# ---------------------------------------------------------------------------------------------

class PhaseAnalyzer(ABC):
    """
    Analyzer for a single phase of a MultiState simulation.

    Uses the reporter from the simulation to determine the location
    of all variables.

    To compute a specific observable in an implementation of this class, add it to the ObservableRegistry and then
    implement a ``get_X`` where ``X`` is the name of the observable you want to compute. See the ObservablesRegistry for
    information about formatting the observables.

    Analyzer works in units of kT unless specifically stated otherwise. To convert back to a unit set, just multiply by
    the .kT property.

    A PhaseAnalyzer also needs an ObservablesRegistry to track how to handle each observable given implemented within
    for things like error and cross-phase analysis.

    Parameters
    ----------
    reporter : MultiStateReporter instance
        Reporter from MultiState which ties to the simulation data on disk.
    name : str, Optional
        Unique name you want to assign this phase, this is the name that will appear in :class:`MultiPhaseAnalyzer`'s.
        If not set, it will be given the arbitrary name "phase#" where # is an integer, chosen in order that it is
        assigned to the :class:`MultiPhaseAnalyzer`.
    reference_states : tuple of ints, length 2, Optional, Default: (0,-1)
        Integers ``i`` and ``j`` of the state that is used for reference in observables, "O". These values are only used
        when reporting single numbers or combining observables through :class:`MultiPhaseAnalyzer` (since the number of
        states between phases can be different). Calls to functions such as ``get_free_energy`` in a single Phase
        results in the O being returned for all states.

            For O completely defined by the state itself (i.e. no differences between states, e.g. Temperature),
            only O[i] is used

            For O where differences between states are required (e.g. Free Energy): O[i,j] = O[j] - O[i]

            For O defined by the phase as a whole, the reference states are not needed.

    analysis_kwargs : None or dict, optional
        Dictionary of extra keyword arguments to pass into the analysis tool, typically MBAR.
        For instance, the initial guess of relative free energies to give to MBAR would be something like:
        ``{'initial_f_k':[0,1,2,3]}``

    registry : ObservablesRegistry instance
        Instanced ObservablesRegistry with all observables implemented through a ``get_X`` function classified and
        registered. Any cross-phase analysis must use the same instance of an ObservablesRegistry


    Attributes
    ----------
    name
    observables
    mbar
    reference_states
    kT
    reporter
    registry

    See Also
    --------
    ObservablesRegistry

    """
    def __init__(self, reporter,
                 name=None, reference_states=(0, -1), analysis_kwargs=None,
                 registry=default_observables_registry):
        """
        The reporter provides the hook into how to read the data, all other options control where differences are
        measured from and how each phase interfaces with other phases.
        """
        if type(reporter) is str:
            raise ValueError('reporter must be a MultiStateReporter instance')

        if not isinstance(registry, ObservablesRegistry):
            raise ValueError("Registry must be an instanced ObservablesRegistry")
        self.registry = registry
        if not reporter.is_open():
            reporter.open(mode='r')
        self._reporter = reporter
        # Internal properties
        self._name = name
        self._initialize_observables()
        # Start as default sign +, handle all sign conversion at preparation time
        self._sign = '+'
        self._equilibration_data = None  # Internal tracker so the functions can get this data without recalculating it
        # External properties
        self._reference_states = None  # initialize the cache object
        self.reference_states = reference_states
        self._mbar = None
        self._kT = None
        if type(analysis_kwargs) not in [type(None), dict]:
            raise ValueError('analysis_kwargs must be either None or a dictionary')
        self._extra_analysis_kwargs = analysis_kwargs if (analysis_kwargs is not None) else dict()

    def _initialize_observables(self):
        observables = []
        # Auto-determine the computable observables by inspection of non-flagged methods
        # We determine valid observables by negation instead of just having each child implement the method to enforce
        # uniform function naming conventions.
        self._computed_observables = {}  # Cache of observables so the phase can be retrieved once computed
        for observable in self.registry.observables:
            if hasattr(self, "get_" + observable):
                observables.append(observable)
                self._computed_observables[observable] = None
        # Cast observables to an immutable
        self._observables = tuple(observables)

    def clear(self):
        """Reset the MBAR and observables object"""
        self._initialize_observables()
        self._mbar = None

    @property
    def name(self):
        """User-readable string name of the phase"""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def observables(self):
        """
        List of observables that the instanced analyzer can compute/fetch.

        This list is automatically compiled upon class initialization based on the functions implemented in the subclass
        """
        return self._observables

    @property
    def mbar(self):
        """MBAR object tied to this phase"""
        if self._mbar is None:
            self._create_mbar_from_scratch()
        return self._mbar

    @property
    def reference_states(self):
        """Tuple of reference states ``i`` and ``j`` for :class:`MultiPhaseAnalyzer` instances"""
        return self._reference_states

    @reference_states.setter
    def reference_states(self, value):
        """Provide a way to re-assign the ``i, j`` states in a protected way"""
        i, j = value[0], value[1]
        if type(i) is not int or type(j) is not int:
            raise ValueError("reference_states must be a length 2 iterable of ints")
        self._reference_states = (i, j)

    @property
    def kT(self):
        """
        Quantity of boltzmann constant times temperature of the phase in units of energy per mol

        Allows conversion between dimensionless energy and unit bearing energy
        """
        if self._kT is None:
            thermodynamic_states, _ = self._reporter.read_thermodynamic_states()
            temperature = thermodynamic_states[0].temperature
            self._kT = kB * temperature
        return self._kT

    @property
    def reporter(self):
        """Sampler Reporter tied to this object."""
        return self._reporter

    @reporter.setter
    def reporter(self, value):
        """Make sure users cannot overwrite the reporter."""
        raise ValueError("You cannot re-assign the reporter for this analyzer!")

    def read_energies(self):
        """
        Extract energies from the ncfile and order them by replica, state, iteration

        Returns
        -------
        energy_matrix : np.ndarray of shape [n_replicas, n_states, n_iterations]
            Potential energy matrix of the sampled states
        unsampled_energy_matrix : np.ndarray of shape [n_replicas, n_unsamped_states, n_iterations]
            Potential energy matrix of the unsampled states
            Energy from each drawn sample n, evaluated at unsampled state l
            If no unsampled states were drawn, this will be shape (0,N)
        neighborhoods : np.ndarray of shape [n_replicas, n_states, n_iterations]
            Neighborhood energies were computed at, uses a boolean mask over the energy_matrix
        sampled_states : np.ndarray of shape [n_replicas, n_iterations]
            States sampled by the replicas in the energy_matrix
        """
        logger.info("Reading energies...")
        energy_thermodynamic_states, neighborhoods, energy_unsampled_states = self._reporter.read_energies()
        # n_iterations, n_replicas, n_states = energy_thermodynamic_states.shape
        # _, _, n_unsampled_states = energy_unsampled_states.shape
        # energy_matrix = np.zeros([n_replicas, n_states, n_iterations], np.float64)
        # unsampled_energy_matrix = np.zeros([n_replicas, n_unsampled_states, n_iterations], np.float64)
        energy_matrix = np.moveaxis(energy_thermodynamic_states, 0, -1)
        unsampled_energy_matrix = np.moveaxis(energy_unsampled_states, 0, -1)
        # for n in range(n_iterations):
        #     energy_matrix[:, :, n] = energy_thermodynamic_states[n, :, :]
        #     unsampled_energy_matrix[:, :, n] = energy_unsampled_states[n, :, :]
        # 2D matrix, can transpose to get the matrix in the right place
        sampled_states = self._reporter.read_replica_thermodynamic_states().T
        logger.info("Done.")

        # TODO: Figure out what format we need the data in to be useful for both global and local MBAR/WHAM
        # For now, we simply can't handle analysis of non-global calculations.
        if np.any(neighborhoods == 0):
            raise Exception('Non-global MBAR analysis not implemented yet.')

        return energy_matrix, unsampled_energy_matrix, neighborhoods, sampled_states

    @property
    def has_log_weights(self):
        """
        Return True if the storage has log weights, False otherwise
        """
        try:
            # Check that logZ and log_weights have per-iteration data
            # If either of these return a ValueError, then no history data are available
            _ = self._reporter.read_logZ(0)
            _ = self._reporter.read_online_analysis_data(0, 'log_weights')
            return True
        except ValueError:
            return False

    def read_log_weights(self):
        """
        Extract log weights from the ncfile, if present.
        Returns ValueError if not present.

        Returns
        -------
        log_weights : np.ndarray of shape [n_states, n_iterations]
            log_weights[l,n] is the log weight applied to state ``l``
            during the collection of samples at iteration ``n``

        """
        log_weights = np.array(
            self._reporter.read_online_analysis_data(slice(None, None), 'log_weights')['log_weights'])
        log_weights = np.moveaxis(log_weights, 0, -1)
        return log_weights

    def read_logZ(self, iteration=None):
        """
        Extract logZ estimates from the ncfile, if present.
        Returns ValueError if not present.

        Parameters
        ----------
        iteration : int or slice, optional, default=None
            If specified, iteration or slice of iterations to extract

        Returns
        -------
        logZ : np.ndarray of shape [n_states, n_iterations]
            logZ[l,n] is the online logZ estimate for state ``l`` at iteration ``n``

        """
        if iteration == -1:
            log_z = self._reporter.read_logZ(iteration)
        else:
            if iteration is not None:
                log_z = self._reporter.read_online_analysis_data(iteration, "logZ")["logZ"]
            else:
                log_z = self._reporter.read_online_analysis_data(slice(0, None), "logZ")["logZ"]
            log_z = np.moveaxis(log_z, 0, -1)
        return log_z

    @abc.abstractmethod
    def _create_mbar_from_scratch(self):
        """
        This method should automatically do everything needed to make the MBAR object from file. It should make all
        the assumptions needed to make the MBAR object.  Typically many of these functions will be needed for the
        :func:`analyze_phase` function.

        Should call the :func:`_prepare_mbar_input_data` to get the data ready for

        Returns nothing, but the self.mbar object should be set after this.
        """
        raise NotImplementedError()

    def _prepare_mbar_input_data(self, sampled_energy_matrix, unsampled_energy_matrix, sampled_states):
        """
        Prepare a set of data for MBAR given sampled and unsampled energy

        Parameters
        ----------
        sampled_energy_matrix : np.ndarray of shape [n_replicas, n_sampled_states, n_iterations]
            Energy of the sampled thermodynamic states by the replicas
        unsampled_energy_matrix : np.ndarray of shale [n_replicas, n_unsampled_states, n_iterations]
            Energy of the unsampled thermodynamic states by every replica
        sampled_states : np.ndarray of shape [n_replicas, n_iterations
            Integer array of of the state sampled by each replica at each iteration

        Returns
        -------
        energy_matrix : energy matrix of shape (K,N) indexed by k,n
            K is the total number of states observables are desired
            N is the total number of samples drawn from ALL states
            The nth configuration is the energy evaluated in the kth thermodynamic state
        samples_per_state : 1-D iterable of shape K
            The number of samples drawn from each kth state
            The \sum samples_per_state = N
        """
        n_replica, n_sampled_states, n_iterations = sampled_energy_matrix.shape
        _, n_unsampled_states, _ = unsampled_energy_matrix.shape
        # Initialize the states
        total_states = n_sampled_states + n_unsampled_states
        energy_matrix = np.zeros([total_states, n_iterations*n_replica])
        samples_per_state = np.zeros([total_states], dtype=int)
        # Compute shift index for how many unsampled states there were
        first_sampled_state = int(n_unsampled_states/2.0)
        last_sampled_state = total_states - first_sampled_state
        # Cast the sampled states into the energy matrix
        energy_matrix[first_sampled_state:last_sampled_state, :] = self.reformat_energies_for_mbar(sampled_energy_matrix)
        # Determine how many samples and which states they were drawn from
        unique_sampled_states, counts = np.unique(sampled_states, return_counts=True)
        # Assign those counts to the correct range of states
        samples_per_state[first_sampled_state:last_sampled_state][unique_sampled_states] = counts
        if n_unsampled_states > 0:
            energy_matrix[[0, -1], :] = self.reformat_energies_for_mbar(unsampled_energy_matrix)
            logger.info("Found expanded cutoff states in the energies!")
            logger.info("Free energies will be reported relative to them instead!")
        return energy_matrix, samples_per_state

    @abc.abstractmethod
    def get_effective_energy_timeseries(self):
        """
        Generate the effective energy (negative log deviance) timeseries that is generated for this phase

        The effective energy for a series of samples x_n, n = 1..N, is defined as

        u_n = - \ln \pi(x_n) + c

        where \pi(x) is the probability density being sampled, and c is an arbitrary constant.

        Returns
        -------
        u_n : ndarray of shape (N,)
            u_n[n] is the negative log deviance of the same from iteration ``n``
            Timeseries used to determine equilibration time and statistical inefficiency.

        """
        raise NotImplementedError("This class has not implemented this function")

    @staticmethod
    def reformat_energies_for_mbar(u_kln: np.ndarray, n_k: Optional[np.ndarray]=None):
        """
        Convert [replica, state, iteration] data into [state, total_iteration] data

        This method assumes that the first dimension are all samplers,
        the second dimension are all the thermodynamic states energies were evaluated at
        and an equal number of samples were drawn from each k'th sampler, UNLESS n_k is specified.

        Parameters
        ----------
        u_kln : np.ndarray of shape (K,L,N')
            K = number of replica samplers
            L = number of thermodynamic states,
            N' = number of iterations from state k
        n_k : np.ndarray of shape K or None
            Number of samples each _SAMPLER_ (k) has drawn
            This allows you to have trailing entries on a given kth row in the n'th (n prime) index
            which do not contribute to the conversion.

            If this is None, assumes ALL samplers have the same number of samples
            such that N_k = N' for all k

            **WARNING**: N_k is number of samples the SAMPLER drew in total,
            NOT how many samples were drawn from each thermodynamic state L.
            This method knows nothing of how many samples were drawn from each state.

        Returns
        -------
        u_ln : np.ndarray of shape (L, N)
            Reduced, non-sparse data format
            L = number of thermodynamic states
            N = \sum_k N_k. note this is not N'
        """
        k, l, n = u_kln.shape
        if n_k is None:
            n_k = np.ones(k, dtype=np.int32)*n
        u_ln = np.zeros([l, n_k.sum()])
        n_counter = 0
        for k_index in range(k):
            u_ln[:, n_counter:n_counter + n_k[k_index]] = u_kln[k_index, :, :n_k[k_index]]
            n_counter += n_k[k_index]
        return u_ln

    # Private Class Methods
    def _create_mbar(self, energy_matrix, samples_per_state):
        """
        Initialize MBAR for Free Energy and Enthalpy estimates, this may take a while.
        This function is helpful for those who want to create a slightly different mbar object with different
        parameters.

        This function is hidden from the user unless they really, really need to create their own mbar object

        Parameters
        ----------
        energy_matrix : array of numpy.float64, optional, default=None
           Reduced potential energies of the replicas; if None, will be extracted from the ncfile
        samples_per_state : array of ints, optional, default=None
           Number of samples drawn from each kth state; if None, will be extracted from the ncfile

        """

        # Delete observables cache since we are now resetting the estimator
        for observable in self.observables:
            self._computed_observables[observable] = None

        # Initialize MBAR (computing free energy estimates, which may take a while)
        logger.info("Computing free energy differences...")
        mbar = MBAR(energy_matrix, samples_per_state, **self._extra_analysis_kwargs)
        self._mbar = mbar

    def _combine_phases(self, other, operator='+'):
        """
        Workhorse function when creating a :class:`MultiPhaseAnalyzer` object by combining single
        :class:`PhaseAnalyzer`s
        """
        phases = [self]
        names = []
        signs = [self._sign]
        # Reset self._sign
        self._sign = '+'
        if self.name is None:
            names.append(utils.generate_phase_name(self.name, []))
        else:
            names.append(self.name)
        if isinstance(other, MultiPhaseAnalyzer):
            new_phases = other.phases
            new_signs = other.signs
            new_names = other.names
            final_new_names = []
            for name in new_names:
                other_names = [n for n in new_names if n != name]
                final_new_names.append(utils.generate_phase_name(name, other_names + names))
            names.extend(final_new_names)
            for new_sign in new_signs:
                if operator != '+' and new_sign == '+':
                    signs.append('-')
                else:
                    signs.append('+')
            phases.extend(new_phases)
        elif isinstance(other, PhaseAnalyzer):
            names.append(utils.generate_phase_name(other.name, names))
            if operator != '+' and other._sign == '+':
                signs.append('-')
            else:
                signs.append('+')
            # Reset the other's sign if it got set to negative
            other._sign = '+'
            phases.append(other)
        else:
            base_err = "cannot {} 'PhaseAnalyzer' and '{}' objects"
            if operator == '+':
                err = base_err.format('add', type(other))
            else:
                err = base_err.format('subtract', type(other))
            raise TypeError(err)
        phase_pass = {'phases': phases, 'signs': signs, 'names': names}
        return MultiPhaseAnalyzer(phase_pass)

    def __add__(self, other):
        return self._combine_phases(other, operator='+')

    def __sub__(self, other):
        return self._combine_phases(other, operator='-')

    def __neg__(self):
        """Internally handle the internal sign"""
        if self._sign == '+':
            self._sign = '-'
        else:
            self._sign = '+'
        return self


class MultiStateSamplerAnalyzer(PhaseAnalyzer):

    """
    The MultiStateSamplerAnalyzer is the analyzer for a simulation generated from a MultiStateSampler simulation,
    implemented as an instance of the :class:`PhaseAnalyzer`.

    See Also
    --------
    PhaseAnalyzer

    """

    # TODO use class syntax and add docstring after dropping python 3.5 support.
    _MixingStatistics = NamedTuple('MixingStatistics', [
        ('transition_matrix', np.ndarray),
        ('eigenvalues', np.ndarray),
        ('statistical_inefficiency', np.ndarray)
    ])

    def generate_mixing_statistics(self, number_equilibrated: Union[int, None] = None) -> NamedTuple:
        """
        Compute and return replica mixing statistics.

        Compute the transition state matrix, its eigenvalues sorted from
        greatest to least, and the state index correlation function.

        Parameters
        ----------
        number_equilibrated : int, optional, default=None
            If specified, only samples ``number_equilibrated:end`` will
            be used in analysis. If not specified, automatically retrieves
            the number from equilibration data or generates it from the
            internal energy.

        Returns
        -------
        mixing_statistics : namedtuple
            A namedtuple containing the following attributes:
            - ``transition_matrix``: (nstates by nstates ``np.array``)
            - ``eigenvalues``: (nstates-dimensional ``np.array``)
            - ``statistical_inefficiency``: float
        """
        # Read data from disk
        if number_equilibrated is None:
            if self._equilibration_data is None:
                self._get_equilibration_data_auto()
            number_equilibrated, _, _ = self._equilibration_data
        states = self._reporter.read_replica_thermodynamic_states()
        n_iterations, n_replicas = states.shape
        n_states = self._reporter.n_states
        n_ij = np.zeros([n_states, n_states], np.int64)

        # Compute empirical transition count matrix.
        for iteration in range(number_equilibrated, n_iterations - 1):
            for i_replica in range(n_replicas):
                i_state = states[iteration, i_replica]
                j_state = states[iteration + 1, i_replica]
                n_ij[i_state, j_state] += 1

        # Compute transition matrix estimate.
        # TODO: Replace with maximum likelihood reversible count estimator from msmbuilder or pyemma.
        t_ij = np.zeros([n_states, n_states], np.float64)
        for i_state in range(n_states):
            # Cast to float to ensure we don't get integer division
            denominator = float((n_ij[i_state, :].sum() + n_ij[:, i_state].sum()))
            if denominator > 0:
                for j_state in range(n_states):
                    t_ij[i_state, j_state] = (n_ij[i_state, j_state] + n_ij[j_state, i_state]) / denominator
            else:
                t_ij[i_state, i_state] = 1.0

        # Estimate eigenvalues
        mu = np.linalg.eigvals(t_ij)
        mu = -np.sort(-mu)  # Sort in descending order

        # Compute state index statistical inefficiency of stationary data.
        # states[n][k] is the state index of replica k at iteration n, but
        # the functions wants a list of timeseries states[k][n].
        states_kn = np.transpose(states[number_equilibrated:])
        g = timeseries.statisticalInefficiencyMultiple(states_kn)

        return self._MixingStatistics(transition_matrix=t_ij, eigenvalues=mu,
                                      statistical_inefficiency=g)

    def show_mixing_statistics(self, cutoff=0.05, number_equilibrated=None):
        """
        Print summary of mixing statistics. Passes information off to generate_mixing_statistics then prints it out to
        the logger

        Parameters
        ----------
        cutoff : float, optional, default=0.05
           Only transition probabilities above 'cutoff' will be printed
        number_equilibrated : int, optional, default=None
           If specified, only samples number_equilibrated:end will be used in analysis
           If not specified, it uses the internally held statistics best

        """

        mixing_statistics = self.generate_mixing_statistics(number_equilibrated=number_equilibrated)

        # Print observed transition probabilities.
        nstates = mixing_statistics.transition_matrix.shape[1]
        logger.info("Cumulative symmetrized state mixing transition matrix:")
        str_row = "{:6s}".format("")
        for jstate in range(nstates):
            str_row += "{:6d}".format(jstate)
        logger.info(str_row)

        for istate in range(nstates):
            str_row = ""
            str_row += "{:-6d}".format(istate)
            for jstate in range(nstates):
                P = mixing_statistics.transition_matrix[istate, jstate]
                if P >= cutoff:
                    str_row += "{:6.3f}".format(P)
                else:
                    str_row += "{:6s}".format("")
            logger.info(str_row)

        # Estimate second eigenvalue and equilibration time.
        perron_eigenvalue = mixing_statistics.eigenvalues[1]
        if perron_eigenvalue >= 1:
            logger.info('Perron eigenvalue is unity; Markov chain is decomposable.')
        else:
            equilibration_timescale = 1.0 / (1.0 - perron_eigenvalue)
            logger.info('Perron eigenvalue is {0:.5f}; state equilibration timescale '
                        'is ~ {1:.1f} iterations'.format(perron_eigenvalue, equilibration_timescale)
            )

        # Print information about replica state index statistical efficiency.
        logger.info('Replica state index statistical inefficiency is '
                    '{:.3f}'.format(mixing_statistics.statistical_inefficiency))

    def get_effective_energy_timeseries(self):
        """
        Generate the effective energy (negative log deviance) timeseries that is generated for this phase

        The effective energy for a series of samples x_n, n = 1..N, is defined as

        u_n = - \ln \pi(x_n) + c

        where \pi(x) is the probability density being sampled, and c is an arbitrary constant.

        Returns
        -------
        u_n : ndarray of shape (N,)
            u_n[n] is the negative log deviance of the same from iteration ``n``
            Timeseries used to determine equilibration time and statistical inefficiency.

        """
        energies, _, _, states = self.read_energies()
        n_replicas, n_states, n_iterations = energies.shape

        # Check for log weights
        has_log_weights = False
        if self.has_log_weights:
            has_log_weights = True
            log_weights = self.read_log_weights()
            f_l = - self.read_logZ(iteration=-1)  # use last (best) estimate of free energies

        u_n = np.zeros([n_iterations], np.float64)
        # Slice of all replicas, have to use this as : is too greedy
        replicas_slice = range(n_replicas)
        for iteration in range(n_iterations):
            states_slice = states[:, iteration]  # slice of current sampled states by those replicas
            u_n[iteration] = np.sum(energies[replicas_slice, states_slice, iteration])

            # Correct for potentially-changing log weights
            if has_log_weights:
                u_n[iteration] += - np.sum(log_weights[states_slice, iteration]) + (
                        n_replicas * logsumexp(-f_l[:] + log_weights[:, iteration]))

        return u_n

    def _compute_free_energy(self):
        """
        Estimate free energies of all alchemical states.
        """

        # Create MBAR object if not provided
        if self._mbar is None:
            self._create_mbar_from_scratch()

        nstates = self.mbar.N_k.size

        # Get matrix of dimensionless free energy differences and uncertainty estimate.
        logger.info("Computing covariance matrix...")

        try:
            # pymbar 2
            (Deltaf_ij, dDeltaf_ij) = self.mbar.getFreeEnergyDifferences()
        except ValueError:
            # pymbar 3
            (Deltaf_ij, dDeltaf_ij, _) = self.mbar.getFreeEnergyDifferences()

        # Matrix of free energy differences
        logger.info("Deltaf_ij:")
        for i in range(nstates):
            str_row = ""
            for j in range(nstates):
                str_row += "{:8.3f}".format(Deltaf_ij[i, j])
            logger.info(str_row)

        # Matrix of uncertainties in free energy difference (expectations standard
        # deviations of the estimator about the true free energy)
        logger.info("dDeltaf_ij:")
        for i in range(nstates):
            str_row = ""
            for j in range(nstates):
                str_row += "{:8.3f}".format(dDeltaf_ij[i, j])
            logger.info(str_row)

        # Return free energy differences and an estimate of the covariance.
        free_energy_dict = {'value': Deltaf_ij, 'error': dDeltaf_ij}
        self._computed_observables['free_energy'] = free_energy_dict

    def get_free_energy(self):
        """
        Compute the free energy and error in free energy from the MBAR object

        Output shape changes based on if there are unsampled states detected in the sampler

        Returns
        -------
        DeltaF_ij : ndarray of floats, shape (K,K) or (K+2, K+2)
            Difference in free energy from each state relative to each other state
        dDeltaF_ij : ndarray of floats, shape (K,K) or (K+2, K+2)
            Error in the difference in free energy from each state relative to each other state
        """
        if self._computed_observables['free_energy'] is None:
            self._compute_free_energy()
        free_energy_dict = self._computed_observables['free_energy']
        return free_energy_dict['value'], free_energy_dict['error']

    def _compute_enthalpy_and_entropy(self):
        """Function to compute the cached values of enthalpy and entropy"""
        if self._mbar is None:
            self._create_mbar_from_scratch()
        (f_k, df_k, H_k, dH_k, S_k, dS_k) = self.mbar.computeEntropyAndEnthalpy()
        enthalpy = {'value': H_k, 'error': dH_k}
        entropy = {'value': S_k, 'error': dS_k}
        self._computed_observables['enthalpy'] = enthalpy
        self._computed_observables['entropy'] = entropy

    def get_enthalpy(self):
        """
        Compute the difference in enthalpy and error in that estimate from the MBAR object

        Output shape changes based on if there are unsampled states detected in the sampler

        Returns
        -------
        DeltaH_ij : ndarray of floats, shape (K,K) or (K+2, K+2)
            Difference in enthalpy from each state relative to each other state
        dDeltaH_ij : ndarray of floats, shape (K,K) or (K+2, K+2)
            Error in the difference in enthalpy from each state relative to each other state
        """
        if self._computed_observables['enthalpy'] is None:
            self._compute_enthalpy_and_entropy()
        enthalpy_dict = self._computed_observables['enthalpy']
        return enthalpy_dict['value'], enthalpy_dict['error']

    def get_entropy(self):
        """
        Compute the difference in entropy and error in that estimate from the MBAR object

        Output shape changes based on if there are unsampled states detected in the sampler

        Returns
        -------
        DeltaH_ij : ndarray of floats, shape (K,K) or (K+2, K+2)
            Difference in enthalpy from each state relative to each other state
        dDeltaH_ij : ndarray of floats, shape (K,K) or (K+2, K+2)
            Error in the difference in enthalpy from each state relative to each other state
        """
        if self._computed_observables['entropy'] is None:
            self._compute_enthalpy_and_entropy()
        entropy_dict = self._computed_observables['entropy']
        return entropy_dict['value'], entropy_dict['error']

    def _get_equilibration_data_auto(self):
        """
        Automatically generate the equilibration data from best practices,
        part of the :func:`_create_mbar_from_scratch` routine.

        Returns nothing, but sets self._equilibration_data
        """
        u_n = self.get_effective_energy_timeseries()

        # Discard equilibration samples.
        # TODO: if we include u_n[0] (the energy right after minimization) in the equilibration detection,
        # TODO:         then number_equilibrated is 0. Find a better way than just discarding first frame.
        self._equilibration_data = utils.get_equilibration_data(u_n[1:])

    def _create_mbar_from_scratch(self):
        # Extract energies
        energy_sampled, energy_unsampled, neighborhood, sampled_states = self.read_energies()
        # Generate decorrelation data
        self._get_equilibration_data_auto()
        number_equilibrated, g_t, Neff_max = self._equilibration_data
        # Remove equilibrated data
        energy_sampled = utils.remove_unequilibrated_data(energy_sampled, number_equilibrated, -1)
        energy_unsampled = utils.remove_unequilibrated_data(energy_unsampled, number_equilibrated, -1)
        sampled_states = utils.remove_unequilibrated_data(sampled_states, number_equilibrated, -1)
        neighborhood = utils.remove_unequilibrated_data(neighborhood, number_equilibrated, -1)
        # Subsample along the decorrelation data
        energy_sampled = utils.subsample_data_along_axis(energy_sampled, g_t, -1)
        energy_unsampled = utils.subsample_data_along_axis(energy_unsampled, g_t, -1)
        sampled_states = utils.subsample_data_along_axis(sampled_states, g_t, -1)
        neighborhood = utils.subsample_data_along_axis(neighborhood, g_t, -1)
        mbar_kn, mbar_N_k = self._prepare_mbar_input_data(energy_sampled, energy_unsampled, sampled_states)
        self._create_mbar(mbar_kn, mbar_N_k)


# https://choderalab.slack.com/files/levi.naden/F4G6L9X8S/quick_diagram.png

class MultiPhaseAnalyzer(object):
    """
    Multiple Phase Analyzer creator, not to be directly called itself, but instead called by adding or subtracting
    different implemented :class:`PhaseAnalyzer` or other :class:`MultiPhaseAnalyzers`'s. The individual Phases of
    the :class:`MultiPhaseAnalyzer` are only references to existing Phase objects, not copies. All
    :class:`PhaseAnalyzer` and :class:`MultiPhaseAnalyzer` classes support ``+`` and ``-`` operations.

    The observables of this phase are determined through inspection of all the passed in phases and only observables
    which are shared can be computed. For example:

        ``PhaseA`` has ``.get_free_energy`` and ``.get_entropy``

        ``PhaseB`` has ``.get_free_energy`` and ``.get_enthalpy``,

        ``PhaseAB = PhaseA + PhaseB`` will only have a ``.get_free_energy`` method

    Because each Phase may have a different number of states, the ``reference_states`` property of each phase
    determines which states from each phase to read the data from.

    For observables defined by two states, the i'th and j'th reference states are used:

        If we define ``PhaseAB = PhaseA - PhaseB``

        Then ``PhaseAB.get_free_energy()`` is roughly equivalent to doing the following:

            ``A_i, A_j = PhaseA.reference_states``

            ``B_i, B_j = PhaseB.reference_states``

            ``PhaseA.get_free_energy()[A_i, A_j] - PhaseB.get_free_energy()[B_i, B_j]``

        The above is not exact since get_free_energy returns an error estimate as well

    For observables defined by a single state, only the i'th reference state is used

        Given ``PhaseAB = PhaseA + PhaseB``, ``PhaseAB.get_temperature()`` is equivalent to:

            ``A_i = PhaseA.reference_states[0]``

            ``B_i = PhaseB.reference_states[0]``

            ``PhaseA.get_temperature()[A_i] + PhaseB.get_temperature()[B_i]``

    For observables defined entirely by the phase, no reference states are needed.

        Given ``PhaseAB = PhaseA + PhaseB``, ``PhaseAB.get_standard_state_correction()`` gives:

            ``PhaseA.get_standard_state_correction() + PhaseB.get_standard_state_correction()``

    Each phase MUST use the same ObservablesRegistry, otherwise an error is raised

    This class is public to see its API.

    Parameters
    ----------
    phases : dict
        has keys "phases", "names", and "signs"

    Attributes
    ----------
    observables
    phases
    names
    signs
    registry

    See Also
    --------
    PhaseAnalyzer
    ObservablesRegistry

    """
    def __init__(self, phases):
        """
        Create the compound phase which is any combination of phases to generate a new MultiPhaseAnalyzer.

        """
        # Compare ObservableRegistries
        ref_registry = phases['phases'][0].registry
        for phase in phases['phases'][1:]:
            # Use is comparison since we are checking same insetance
            if phase.registry is not ref_registry:
                raise ValueError("Not all phases have the same ObservablesRegistry! Observable calculation "
                                 "will be inconsistent!")
        self.registry = ref_registry
        # Determine available observables
        observables = []
        for observable in self.registry.observables:
            shared_observable = True
            for phase in phases['phases']:
                if observable not in phase.observables:
                    shared_observable = False
                    break
            if shared_observable:
                observables.append(observable)
        if len(observables) == 0:
            raise RuntimeError("There are no shared computable observable between the phases, combining them will do "
                               "nothing.")
        self._observables = tuple(observables)
        self._phases = phases['phases']
        self._names = phases['names']
        self._signs = phases['signs']
        # Set the methods shared between both objects
        for observable in self.observables:
            setattr(self, "get_" + observable, self._spool_function(observable))

    def _spool_function(self, observable):
        """
        Dynamic observable calculator layer

        Must be in its own function to isolate the variable name space

        If you have this in the __init__, the "observable" variable colides with any others in the list, causing a
        the wrong property to be fetched.
        """
        return lambda: self._compute_observable(observable)

    @property
    def observables(self):
        """List of observables this :class:`MultiPhaseAnalyzer` can generate"""
        return self._observables

    @property
    def phases(self):
        """List of implemented :class:`PhaseAnalyzer`'s objects this :class:`MultiPhaseAnalyzer` is tied to"""
        return self._phases

    @property
    def names(self):
        """
        Unique list of string names identifying this phase. If this :class:`MultiPhaseAnalyzer` is combined with
        another, its possible that new names will be generated unique to that :class:`MultiPhaseAnalyzer`, but will
        still reference the same phase.

        When in doubt, use :func:`MultiPhaseAnalyzer.phases` to get the actual phase objects.
        """
        return self._names

    @property
    def signs(self):
        """
        List of signs that are used by the :class:`MultiPhaseAnalyzer` to
        """
        return self._signs

    def clear(self):
        """
        Clear the individual phases of their observables and estimators for re-computing quantities
        """
        for phase in self.phases:
            phase.clear()

    def _combine_phases(self, other, operator='+'):
        """
        Function to combine the phases regardless of operator to reduce code duplication. Creates a new
        :class:`MultiPhaseAnalyzer` object based on the combined phases of the other. Accepts either a
        :class:`PhaseAnalyzer` or a :class:`MultiPhaseAnalyzer`.

        If the names have collision, they are re-named with an extra digit at the end.

        Parameters
        ----------
        other : :class:`MultiPhaseAnalyzer` or :class:`PhaseAnalyzer`
        operator : sign of the operator connecting the two objects

        Returns
        -------
        output : :class:`MultiPhaseAnalyzer`
            New :class:`MultiPhaseAnalyzer` where the phases are the combined list of the individual phases from each
            component. Because the memory pointers to the individual phases are the same, changing any
            single :class:`PhaseAnalyzer`'s
            reference_state objects updates all :class:`MultiPhaseAnalyzer` objects they are tied to

        """
        phases = []
        names = []
        signs = []
        # create copies
        phases.extend(self.phases)
        names.extend(self.names)
        signs.extend(self.signs)
        if isinstance(other, MultiPhaseAnalyzer):
            new_phases = other.phases
            new_signs = other.signs
            new_names = other.names
            final_new_names = []
            for name in new_names:
                other_names = [n for n in new_names if n != name]
                final_new_names.append(utils.generate_phase_name(name, other_names + names))
            names.extend(final_new_names)
            for new_sign in new_signs:
                if (operator == '-' and new_sign == '+') or (operator == '+' and new_sign == '-'):
                    signs.append('-')
                else:
                    signs.append('+')
            signs.extend(new_signs)
            phases.extend(new_phases)
        elif isinstance(other, PhaseAnalyzer):
            names.append(utils.generate_phase_name(other.name, names))
            if (operator == '-' and other._sign == '+') or (operator == '+' and other._sign == '-'):
                signs.append('-')
            else:
                signs.append('+')
            other._sign = '+'  # Recast to positive if negated
            phases.append(other)
        else:
            baseerr = "cannot {} 'MultiPhaseAnalyzer' and '{}' objects"
            if operator == '+':
                err = baseerr.format('add', type(other))
            else:
                err = baseerr.format('subtract', type(other))
            raise TypeError(err)
        phase_pass = {'phases': phases, 'signs': signs, 'names': names}
        return MultiPhaseAnalyzer(phase_pass)

    def __add__(self, other):
        return self._combine_phases(other, operator='+')

    def __sub__(self, other):
        return self._combine_phases(other, operator='-')

    def __neg__(self):
        """
        Return a SHALLOW copy of self with negated signs so that the phase objects all still point to the same
        objects
        """
        new_signs = []
        for sign in self._signs:
            if sign == '+':
                new_signs.append('-')
            else:
                new_signs.append('+')
        # return a *shallow* copy of self with the signs reversed
        output = copy.copy(self)
        output._signs = new_signs
        return output

    def __str__(self):
        """Simplified string output"""
        header = "MultiPhaseAnalyzer<{}>"
        output_string = ""
        for phase_name, sign in zip(self.names, self.signs):
            if output_string == "" and sign == '-':
                output_string += '{}{} '.format(sign, phase_name)
            elif output_string == "":
                output_string += '{} '.format(phase_name)
            else:
                output_string += '{} {} '.format(sign, phase_name)
        return header.format(output_string)

    def __repr__(self):
        """Generate a detailed representation of the MultiPhase"""
        header = "MultiPhaseAnalyzer <\n{}>"
        output_string = ""
        for phase, phase_name, sign in zip(self.phases, self.names, self.signs):
            if output_string == "" and sign == '-':
                output_string += '{}{} ({})\n'.format(sign, phase_name, phase)
            elif output_string == "":
                output_string += '{} ({})\n'.format(phase_name, phase)
            else:
                output_string += '    {} {} ({})\n'.format(sign, phase_name, phase)
        return header.format(output_string)

    def _compute_observable(self, observable_name):
        """
        Helper function to compute arbitrary observable in both phases

        Parameters
        ----------
        observable_name : str
            Name of the observable as its defined in the ObservablesRegistry

        Returns
        -------
        observable_value
            The observable as its combined between all the phases

        """
        def prepare_phase_observable(single_phase):
            """Helper function to cast the observable in terms of observable's registry"""
            observable = getattr(single_phase, "get_" + observable_name)()
            if isinstance(single_phase, MultiPhaseAnalyzer):
                if observable_name in self.registry.observables_with_error:
                    observable_payload = dict()
                    observable_payload['value'], observable_payload['error'] = observable
                else:
                    observable_payload = observable
            else:
                raise_registry_error = False
                if observable_name in self.registry.observables_with_error:
                    observable_payload = {}
                    if observable_name in self.registry.observables_defined_by_phase:
                        observable_payload['value'], observable_payload['error'] = observable
                    elif observable_name in self.registry.observables_defined_by_single_state:
                        observable_payload['value'] = observable[0][single_phase.reference_states[0]]
                        observable_payload['error'] = observable[1][single_phase.reference_states[0]]
                    elif observable_name in self.registry.observables_defined_by_two_states:
                        observable_payload['value'] = observable[0][single_phase.reference_states[0],
                                                                    single_phase.reference_states[1]]
                        observable_payload['error'] = observable[1][single_phase.reference_states[0],
                                                                    single_phase.reference_states[1]]
                    else:
                        raise_registry_error = True
                else:  # No error
                    if observable_name in self.registry.observables_defined_by_phase:
                        observable_payload = observable
                    elif observable_name in self.registry.observables_defined_by_single_state:
                        observable_payload = observable[single_phase.reference_states[0]]
                    elif observable_name in self.registry.observables_defined_by_two_states:
                        observable_payload = observable[single_phase.reference_states[0],
                                                        single_phase.reference_states[1]]
                    else:
                        raise_registry_error = True
                if raise_registry_error:
                    raise RuntimeError("You have requested an observable that is improperly registered in the "
                                       "ObservablesRegistry!")
            return observable_payload

        def modify_final_output(passed_output, payload, sign):
            if observable_name in self.registry.observables_with_error:
                if sign == '+':
                    passed_output['value'] += payload['value']
                else:
                    passed_output['value'] -= payload['value']
                if observable_name in self.registry.observables_with_error_adding_linear:
                    passed_output['error'] += payload['error']
                elif observable_name in self.registry.observables_with_error_adding_quadrature:
                    passed_output['error'] = (passed_output['error']**2 + payload['error']**2)**0.5
            else:
                if sign == '+':
                    passed_output += payload
                else:
                    passed_output -= payload
            return passed_output

        if observable_name in self.registry.observables_with_error:
            final_output = {'value': 0, 'error': 0}
        else:
            final_output = 0
        for phase, phase_sign in zip(self.phases, self.signs):
            phase_observable = prepare_phase_observable(phase)
            final_output = modify_final_output(final_output, phase_observable, phase_sign)
        if observable_name in self.registry.observables_with_error:
            # Cast output to tuple
            final_output = (final_output['value'], final_output['error'])
        return final_output
