#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
Analyze
=======

Analysis tools and module for MultiStateSampler simulations. Provides programmatic and automatic
"best practices" integration to determine free energy and other observables.

Fully extensible to support new samplers and observables.


"""

# =============================================================================================
# MODULE IMPORTS
# =============================================================================================

import os
import abc
import copy
import yaml
import mdtraj
import logging
import numpy as np
import simtk.unit as units
import openmmtools as mmtools

from . import analyzerutils as autils
from ..multistatereporter import MultiStateReporter

from pymbar import MBAR, timeseries
from typing import Optional, NamedTuple, Union

ABC = abc.ABC
logger = logging.getLogger(__name__)

__all__ = [
    'get_analyzer',
    'PhaseAnalyzer',
    'MultiStateSamplerAnalyzer',
    'ReplicaExchangeAnalyzer',
    'ParallelTemperingAnalyzer',
    'MultiPhaseAnalyzer',
    'analyze_directory',
    'extract_u_n',
    'extract_trajectory'
]

# =============================================================================================
# PARAMETERS
# =============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA


# =============================================================================================
# MODULE FUNCTIONS
# =============================================================================================

def get_analyzer(file_base_path):
    """
    Utility function to convert storage file to a Reporter and Analyzer by reading the data on file

    For now this is mostly placeholder functions since there is only the implemented :class:`ReplicaExchangeAnalyzer`,
    but creates the API for the user to work with.

    Parameters
    ----------
    file_base_path : string
        Complete path to the storage file with filename and extension.

    Returns
    -------
    analyzer : instance of implemented :class:`PhaseAnalyzer`
        Analyzer for the specific phase.
    """
    # Eventually extend this to get more reporters, but for now simple placeholder
    reporter = MultiStateReporter(file_base_path, open_mode='r')
    """
    storage = infer_storage_format_from_extension('complex.nc')  # This is always going to be nc for now.
    metadata = storage.metadata
    sampler_class = metadata['sampler_full_name']
    module_name, cls_name = sampler_full_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    reporter = cls.create_reporter('complex.nc')
    """
    # Eventually change this to auto-detect simulation from reporter:
    if True:
        analyzer = ReplicaExchangeAnalyzer(reporter)
    else:
        raise RuntimeError("Cannot automatically determine analyzer for Reporter: {}".format(reporter))
    return analyzer


# =============================================================================================
# MODULE CLASSES
# =============================================================================================

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


    Attributes
    ----------
    name
    observables
    mbar
    reference_states
    kT
    reporter

    See Also
    --------
    ObservableRegistry

    """
    def __init__(self, reporter, name=None, reference_states=(0, -1), analysis_kwargs=None):
        """
        The reporter provides the hook into how to read the data, all other options control where differences are
        measured from and how each phase interfaces with other phases.
        """
        if not reporter.is_open():
            reporter.open(mode='r')
        self._reporter = reporter
        observables = []
        # Auto-determine the computable observables by inspection of non-flagged methods
        # We determine valid observables by negation instead of just having each child implement the method to enforce
        # uniform function naming conventions.
        self._computed_observables = {}  # Cache of observables so the phase can be retrieved once computed
        for observable in autils.ObservablesRegistry.observables:
            if hasattr(self, "get_" + observable):
                observables.append(observable)
                self._computed_observables[observable] = None
        # Cast observables to an immutable
        self._observables = tuple(observables)
        # Internal properties
        self._name = name
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

    # Abstract methods
    @abc.abstractmethod
    def analyze_phase(self, *args, **kwargs):
        """
        Auto-analysis function for the phase

        Function which broadly handles "auto-analysis" for those that do not wish to call all the methods on their own.

        Returns a dictionary of analysis objects
        """
        raise NotImplementedError()

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

    @abc.abstractmethod
    def _prepare_mbar_input_data(self, *args, **kwargs):
        """
        Prepare a set of data for MBAR, because each analyzer may need to do something else to prepare for MBAR, it
        should have its own function to do that with.

        Parameters
        ----------
        args : arguments needed to generate the appropriate Returns
        kwargs : keyword arguments needed to generate the appropriate Returns

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
        raise NotImplementedError()

    @abc.abstractmethod
    def get_states_energies(self):
        """
        Extract the deconvoluted energies from a phase.

        Energies from this are NOT decorrelated.

        Returns
        -------
        sampled_energy_matrix : numpy.ndarray of shape K,L,N'
            Deconvoluted energy of sampled states evaluated at other sampled states.

            Has shape (K,L,N') = (number of replica samplers,
                                 number of sampled thermodynamic states,
                                 number of iterations from state k)

            Indexed by [k,l,n] where an energy drawn from replica sampler [k] is evaluated in thermodynamic state [l] at
            iteration [n]
        unsampled_energy_matrix : numpy.ndarray of shape K,L,N
            Has shape (K, L, N) = (number of replica samplers,
                                   number of UN-sampled thermodynamic states,
                                   number of iterations)

            Indexed by [k,l,n]
            where an energy drawn from replica state [k] is evaluated in un-sampled state [l] at iteration [n]
        """
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def get_timeseries(passed_timeseries):
        """
        Generate the timeseries that is generated for this phase

        Returns
        -------
        generated_timeseries : 1-D iterable
            timeseries which can be fed into get_decorrelation_time to get the decorrelation
        """

        raise NotImplementedError("This class has not implemented this function")

    @staticmethod
    def reformat_energies_for_mbar(u_kln: np.ndarray, n_k: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Convert u_kln formatted energies into u_ln formatted energies.

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

            **WARNING**: N_k is number of samples the SAMPLER drew,
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
            names.append(autils.generate_phase_name(self.name, []))
        else:
            names.append(self.name)
        if isinstance(other, MultiPhaseAnalyzer):
            new_phases = other.phases
            new_signs = other.signs
            new_names = other.names
            final_new_names = []
            for name in new_names:
                other_names = [n for n in new_names if n != name]
                final_new_names.append(autils.generate_phase_name(name, other_names + names))
            names.extend(final_new_names)
            for new_sign in new_signs:
                if operator != '+' and new_sign == '+':
                    signs.append('-')
                else:
                    signs.append('+')
            phases.extend(new_phases)
        elif isinstance(other, PhaseAnalyzer):
            names.append(autils.generate_phase_name(other.name, names))
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

    def get_states_energies(self):
        """
        Extract and decorrelate energies from the ncfile to gather energies common data for other functions

        Returns
        -------
        energy_matrix : ndarray of shape (K,N)
            Potential energy matrix of the sampled states
            Energy is from each drawn sample n, evaluated at every sampled state k
        unsampled_energy_matrix : ndarray of shape (L,N)
            Potential energy matrix of the unsampled states
            Energy from each drawn sample n, evaluated at unsampled state l
            If no unsampled states were drawn, this will be shape (0,N)

        """
        logger.info("Reading energies...")
        # Returns the energies in kln format
        energy_thermodynamic_states, energy_unsampled_states = self._reporter.read_energies()
        n_iterations, n_replicas, n_states = energy_thermodynamic_states.shape
        _, _, n_unsampled_states = energy_unsampled_states.shape
        energy_matrix_replica = np.zeros([n_replicas, n_states, n_iterations], np.float64)
        unsampled_energy_matrix_replica = np.zeros([n_replicas, n_unsampled_states, n_iterations], np.float64)
        for n in range(n_iterations):
            energy_matrix_replica[:, :, n] = energy_thermodynamic_states[n, :, :]
            unsampled_energy_matrix_replica[:, :, n] = energy_unsampled_states[n, :, :]
        logger.info("Done.")

        logger.info("Deconvoluting replicas...")
        energy_matrix = np.zeros([n_states, n_states, n_iterations], np.float64)
        unsampled_energy_matrix = np.zeros([n_states, n_unsampled_states, n_iterations], np.float64)
        for iteration in range(n_iterations):
            state_indices = self._reporter.read_replica_thermodynamic_states(iteration)
            energy_matrix[state_indices, :, iteration] = energy_matrix_replica[:, :, iteration]
            unsampled_energy_matrix[state_indices, :, iteration] = unsampled_energy_matrix_replica[:, :, iteration]
        logger.info("Done.")

        return energy_matrix, unsampled_energy_matrix

    @staticmethod
    def get_timeseries(passed_timeseries):
        """
        Compute the timeseries of a simulation from the Replica Exchange simulation. This is the sum of energies
        for each sample from the state it was drawn from.

        Parameters
        ----------
        passed_timeseries : ndarray of shape (K,L,N), indexed by k,l,n
            K is the total number of sampled states

            L is the total states we want MBAR to analyze

            N is the total number of samples

            The kth sample was drawn from state k at iteration n, the nth configuration of kth state is evaluated in
            thermodynamic state l

        Returns
        -------
        u_n : ndarray of shape (N,)
            Timeseries to compute decorrelation and equilibration data from.
        """
        niterations = passed_timeseries.shape[-1]
        u_n = np.zeros([niterations], np.float64)
        # Compute total negative log probability over all iterations.
        for iteration in range(niterations):
            u_n[iteration] = np.sum(np.diagonal(passed_timeseries[:, :, iteration]))
        return u_n

    def _prepare_mbar_input_data(self, sampled_energy_matrix, unsampled_energy_matrix):
        """Convert the sampled and unsampled energy matrices into MBAR ready data"""
        nstates, _, niterations = sampled_energy_matrix.shape
        _, nunsampled, _ = unsampled_energy_matrix.shape
        # Subsample data to obtain uncorrelated samples
        N_k = np.zeros(nstates, np.int32)
        N = niterations  # number of uncorrelated samples
        N_k[:] = N
        mbar_ready_energy_matrix = self.reformat_energies_for_mbar(sampled_energy_matrix)
        if nunsampled > 0:
            new_energy_matrix = np.zeros([nstates + 2, N_k.sum()])
            N_k_new = np.zeros(nstates + 2, np.int32)
            unsampled_kn = self.reformat_energies_for_mbar(unsampled_energy_matrix)
            # Add augmented unsampled energies to the new matrix
            new_energy_matrix[[0, -1], :] = unsampled_kn[[0, -1], :]
            # Fill in the old energies to the middle states
            new_energy_matrix[1:-1, :] = mbar_ready_energy_matrix
            N_k_new[1:-1] = N_k
            # Notify users
            logger.info("Found expanded cutoff states in the energies!")
            logger.info("Free energies will be reported relative to them instead!")
            # Reset values, last step in case something went wrong so we dont overwrite u_kn on accident
            mbar_ready_energy_matrix = new_energy_matrix
            N_k = N_k_new
        return mbar_ready_energy_matrix, N_k

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

    def get_standard_state_correction(self):
        """
        Compute the standard state correction free energy associated with the Phase.

        This usually is just a stored variable, but it may need other calculations.

        Returns
        -------
        standard_state_correction : float
            Free energy contribution from the standard_state_correction

        """
        if self._computed_observables['standard_state_correction'] is None:
            ssc = self._reporter.read_dict('metadata')['standard_state_correction']
            self._computed_observables['standard_state_correction'] = ssc
        return self._computed_observables['standard_state_correction']

    def _get_equilibration_data_auto(self, input_data=None):
        """
        Automatically generate the equilibration data from best practices, part of the :func:`_create_mbar_from_scratch`
        routine.

        Parameters
        ----------
        input_data : np.ndarray-like, Optional, Default: None
            Optionally provide the data to look at. If not provided, uses energies from :func:`extract_energies()`

        Returns nothing, but sets self._equilibration_data
        """
        if input_data is None:
            input_data, _ = self.get_states_energies()
        u_n = self.get_timeseries(input_data)
        # Discard equilibration samples.
        # TODO: if we include u_n[0] (the energy right after minimization) in the equilibration detection,
        # TODO:         then number_equilibrated is 0. Find a better way than just discarding first frame.
        self._equilibration_data = autils.get_equilibration_data(u_n[1:])

    def _create_mbar_from_scratch(self):
        u_kln, unsampled_u_kln = self.get_states_energies()
        self._get_equilibration_data_auto(input_data=u_kln)
        number_equilibrated, g_t, Neff_max = self._equilibration_data
        u_kln = autils.remove_unequilibrated_data(u_kln, number_equilibrated, -1)
        unsampled_u_kln = autils.remove_unequilibrated_data(unsampled_u_kln, number_equilibrated, -1)
        # decorrelate_data subsample the energies only based on g_t so both ends up with same indices.
        u_kln = autils.subsample_data_along_axis(u_kln, g_t, -1)
        unsampled_u_kln = autils.subsample_data_along_axis(unsampled_u_kln, g_t, -1)
        mbar_ukn, mbar_N_k = self._prepare_mbar_input_data(u_kln, unsampled_u_kln)
        self._create_mbar(mbar_ukn, mbar_N_k)

    def analyze_phase(self, cutoff=0.05):
        if self._mbar is None:
            self._create_mbar_from_scratch()
        number_equilibrated, g_t, _ = self._equilibration_data
        self.show_mixing_statistics(cutoff=cutoff, number_equilibrated=number_equilibrated)
        data = {}
        # Accumulate free energy differences
        Deltaf_ij, dDeltaf_ij = self.get_free_energy()
        DeltaH_ij, dDeltaH_ij = self.get_enthalpy()
        data['DeltaF'] = Deltaf_ij[self.reference_states[0], self.reference_states[1]]
        data['dDeltaF'] = dDeltaf_ij[self.reference_states[0], self.reference_states[1]]
        data['DeltaH'] = DeltaH_ij[self.reference_states[0], self.reference_states[1]]
        data['dDeltaH'] = dDeltaH_ij[self.reference_states[0], self.reference_states[1]]
        data['DeltaF_standard_state_correction'] = self.get_standard_state_correction()
        return data


class ReplicaExchangeAnalyzer(MultiStateSamplerAnalyzer):

    """
    The ReplicaExchangeAnalyzer is the analyzer for a simulation generated from a Replica Exchange sampler simulation,
    implemented as an instance of the :class:`PhaseAnalyzer`.

    See Also
    --------
    PhaseAnalyzer

    """
    pass


class ParallelTemperingAnalyzer(ReplicaExchangeAnalyzer):
    """
    The ParallelTemperingAnalyzer is the analyzer for a simulation generated from a Parallel Tempering sampler
    simulation, implemented as an instance of the :class:`ReplicaExchangeAnalyzer` as the sampler is a subclass of
    the :class:`yank.multistate.ReplicaExchangeSampler`

    See Also
    --------
    ReplicaExchangeAnalyzer
    PhaseAnalyzer

    """
    pass


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

    """
    def __init__(self, phases):
        """
        Create the compound phase which is any combination of phases to generate a new MultiPhaseAnalyzer.

        """
        # Determine
        observables = []
        for observable in autils.ObservablesRegistry.observables:
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
                final_new_names.append(autils.generate_phase_name(name, other_names + names))
            names.extend(final_new_names)
            for new_sign in new_signs:
                if (operator == '-' and new_sign == '+') or (operator == '+' and new_sign == '-'):
                    signs.append('-')
                else:
                    signs.append('+')
            signs.extend(new_signs)
            phases.extend(new_phases)
        elif isinstance(other, PhaseAnalyzer):
            names.append(autils.generate_phase_name(other.name, names))
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
                if observable_name in autils.ObservablesRegistry.observables_with_error:
                    observable_payload = dict()
                    observable_payload['value'], observable_payload['error'] = observable
                else:
                    observable_payload = observable
            else:
                raise_registry_error = False
                if observable_name in autils.ObservablesRegistry.observables_with_error:
                    observable_payload = {}
                    if observable_name in autils.ObservablesRegistry.observables_defined_by_phase:
                        observable_payload['value'], observable_payload['error'] = observable
                    elif observable_name in autils.ObservablesRegistry.observables_defined_by_single_state:
                        observable_payload['value'] = observable[0][single_phase.reference_states[0]]
                        observable_payload['error'] = observable[1][single_phase.reference_states[0]]
                    elif observable_name in autils.ObservablesRegistry.observables_defined_by_two_states:
                        observable_payload['value'] = observable[0][single_phase.reference_states[0],
                                                                    single_phase.reference_states[1]]
                        observable_payload['error'] = observable[1][single_phase.reference_states[0],
                                                                    single_phase.reference_states[1]]
                    else:
                        raise_registry_error = True
                else:  # No error
                    if observable_name in autils.ObservablesRegistry.observables_defined_by_phase:
                        observable_payload = observable
                    elif observable_name in autils.ObservablesRegistry.observables_defined_by_single_state:
                        observable_payload = observable[single_phase.reference_states[0]]
                    elif observable_name in autils.ObservablesRegistry.observables_defined_by_two_states:
                        observable_payload = observable[single_phase.reference_states[0],
                                                        single_phase.reference_states[1]]
                    else:
                        raise_registry_error = True
                if raise_registry_error:
                    raise RuntimeError("You have requested an observable that is improperly registered in the "
                                       "ObservablesRegistry!")
            return observable_payload

        def modify_final_output(passed_output, payload, sign):
            if observable_name in autils.ObservablesRegistry.observables_with_error:
                if sign == '+':
                    passed_output['value'] += payload['value']
                else:
                    passed_output['value'] -= payload['value']
                if observable_name in autils.ObservablesRegistry.observables_with_error_adding_linear:
                    passed_output['error'] += payload['error']
                elif observable_name in autils.ObservablesRegistry.observables_with_error_adding_quadrature:
                    passed_output['error'] = (passed_output['error']**2 + payload['error']**2)**0.5
            else:
                if sign == '+':
                    passed_output += payload
                else:
                    passed_output -= payload
            return passed_output

        if observable_name in autils.ObservablesRegistry.observables_with_error:
            final_output = {'value': 0, 'error': 0}
        else:
            final_output = 0
        for phase, phase_sign in zip(self.phases, self.signs):
            phase_observable = prepare_phase_observable(phase)
            final_output = modify_final_output(final_output, phase_observable, phase_sign)
        if observable_name in autils.ObservablesRegistry.observables_with_error:
            # Cast output to tuple
            final_output = (final_output['value'], final_output['error'])
        return final_output


def analyze_directory(source_directory):
    """
    Analyze contents of store files to compute free energy differences.

    This function is needed to preserve the old auto-analysis style of YANK. What it exactly does can be refined when
    more analyzers and simulations are made available. For now this function exposes the API.

    Parameters
    ----------
    source_directory : string
       The location of the simulation storage files.

    """
    analysis_script_path = os.path.join(source_directory, 'analysis.yaml')
    if not os.path.isfile(analysis_script_path):
        err_msg = 'Cannot find analysis.yaml script in {}'.format(source_directory)
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    with open(analysis_script_path, 'r') as f:
        analysis = yaml.load(f)
    phase_names = [phase_name for phase_name, sign in analysis]
    data = dict()
    for phase_name, sign in analysis:
        phase_path = os.path.join(source_directory, phase_name + '.nc')
        phase = get_analyzer(phase_path)
        data[phase_name] = phase.analyze_phase()
        kT = phase.kT

    # Compute free energy and enthalpy
    DeltaF = 0.0
    dDeltaF = 0.0
    DeltaH = 0.0
    dDeltaH = 0.0
    for phase_name, sign in analysis:
        DeltaF -= sign * (data[phase_name]['DeltaF'] + data[phase_name]['DeltaF_standard_state_correction'])
        dDeltaF += data[phase_name]['dDeltaF']**2
        DeltaH -= sign * (data[phase_name]['DeltaH'] + data[phase_name]['DeltaF_standard_state_correction'])
        dDeltaH += data[phase_name]['dDeltaH']**2
    dDeltaF = np.sqrt(dDeltaF)
    dDeltaH = np.sqrt(dDeltaH)

    # Attempt to guess type of calculation
    calculation_type = ''
    for phase in phase_names:
        if 'complex' in phase:
            calculation_type = ' of binding'
        elif 'solvent1' in phase:
            calculation_type = ' of solvation'

    # Print energies
    logger.info('')
    logger.info('Free energy{:<13}: {:9.3f} +- {:.3f} kT ({:.3f} +- {:.3f} kcal/mol)'.format(
        calculation_type, DeltaF, dDeltaF, DeltaF * kT / units.kilocalories_per_mole,
        dDeltaF * kT / units.kilocalories_per_mole))
    logger.info('')

    for phase in phase_names:
        logger.info('DeltaG {:<17}: {:9.3f} +- {:.3f} kT'.format(phase, data[phase]['DeltaF'],
                                                                 data[phase]['dDeltaF']))
        if data[phase]['DeltaF_standard_state_correction'] != 0.0:
            logger.info('DeltaG {:<17}: {:18.3f} kT'.format('restraint',
                                                            data[phase]['DeltaF_standard_state_correction']))
    logger.info('')
    logger.info('Enthalpy{:<16}: {:9.3f} +- {:.3f} kT ({:.3f} +- {:.3f} kcal/mol)'.format(
        calculation_type, DeltaH, dDeltaH, DeltaH * kT / units.kilocalories_per_mole,
        dDeltaH * kT / units.kilocalories_per_mole))

# ==========================================
# HELPER FUNCTIONS FOR TRAJECTORY EXTRACTION
# ==========================================

def extract_u_n(ncfile):
    """
    Extract timeseries of u_n = - log q(X_n) from store file

    where q(X_n) = \pi_{k=1}^K u_{s_{nk}}(x_{nk})

    with X_n = [x_{n1}, ..., x_{nK}] is the current collection of replica configurations
    s_{nk} is the current state of replica k at iteration n
    u_k(x) is the kth reduced potential

    TODO: Figure out a way to remove this function

    Parameters
    ----------
    ncfile : netCDF4.Dataset
       Open NetCDF file to analyze

    Returns
    -------
    u_n : numpy array of numpy.float64
       u_n[n] is -log q(X_n)
    """

    # Get current dimensions.
    niterations = ncfile.variables['energies'].shape[0]
    nstates = ncfile.variables['energies'].shape[1]
    natoms = ncfile.variables['energies'].shape[2]

    # Extract energies.
    logger.info("Reading energies...")
    energies = ncfile.variables['energies']
    u_kln_replica = np.zeros([nstates, nstates, niterations], np.float64)
    for n in range(niterations):
        u_kln_replica[:, :, n] = energies[n, :, :]
    logger.info("Done.")

    # Deconvolute replicas
    logger.info("Deconvoluting replicas...")
    u_kln = np.zeros([nstates, nstates, niterations], np.float64)
    for iteration in range(niterations):
        state_indices = ncfile.variables['states'][iteration, :]
        u_kln[state_indices,:,iteration] = energies[iteration, :, :]
    logger.info("Done.")

    # Compute total negative log probability over all iterations.
    u_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        u_n[iteration] = np.sum(np.diagonal(u_kln[:, :, iteration]))

    return u_n


# ==============================================================================
# Extract trajectory from NetCDF4 file
# ==============================================================================

def extract_trajectory(nc_path, nc_checkpoint_file=None, state_index=None, replica_index=None,
                       start_frame=0, end_frame=-1, skip_frame=1, keep_solvent=True,
                       discard_equilibration=False, image_molecules=False):
    """Extract phase trajectory from the NetCDF4 file.

    Parameters
    ----------
    nc_path : str
        Path to the primary nc_file storing the analysis options
    nc_checkpoint_file : str or None, Optional
        File name of the checkpoint file housing the main trajectory
        Used if the checkpoint file is differently named from the default one chosen by the nc_path file.
        Default: None
    state_index : int, optional
        The index of the alchemical state for which to extract the trajectory.
        One and only one between state_index and replica_index must be not None
        (default is None).
    replica_index : int, optional
        The index of the replica for which to extract the trajectory. One and
        only one between state_index and replica_index must be not None (default
        is None).
    start_frame : int, optional
        Index of the first frame to include in the trajectory (default is 0).
    end_frame : int, optional
        Index of the last frame to include in the trajectory. If negative, will
        count from the end (default is -1).
    skip_frame : int, optional
        Extract one frame every skip_frame (default is 1).
    keep_solvent : bool, optional
        If False, solvent molecules are ignored (default is True).
    discard_equilibration : bool, optional
        If True, initial equilibration frames are discarded (see the method
        pymbar.timeseries.detectEquilibration() for details, default is False).

    Returns
    -------
    trajectory: mdtraj.Trajectory
        The trajectory extracted from the netcdf file.

    """
    # Check correct input
    if (state_index is None) == (replica_index is None):
        raise ValueError('One and only one between "state_index" and '
                         '"replica_index" must be specified.')
    if not os.path.isfile(nc_path):
        raise ValueError('Cannot find file {}'.format(nc_path))

    # Import simulation data
    try:
        reporter = MultiStateReporter(nc_path, open_mode='r', checkpoint_storage=nc_checkpoint_file)
        metadata = reporter.read_dict('metadata')
        reference_system = mmtools.utils.deserialize(metadata['reference_state']).system
        topology = mmtools.utils.deserialize(metadata['topography']).topology

        # Determine if system is periodic
        is_periodic = reference_system.usesPeriodicBoundaryConditions()
        logger.info('Detected periodic boundary conditions: {}'.format(is_periodic))

        # Get dimensions
        # Assume full iteration until proven otherwise
        full_iteration = True
        trajectory_storage = reporter._storage_checkpoint
        if not keep_solvent:
            # If tracked solute particles, use any last iteration, set with this logic test
            full_iteration = len(reporter.analysis_particle_indices) == 0
            if not full_iteration:
                trajectory_storage = reporter._storage_analysis
                topology = topology.subset(reporter.analysis_particle_indices)

        n_iterations = reporter.read_last_iteration(full_iteration=full_iteration)
        n_frames = trajectory_storage.variables['positions'].shape[0]
        n_atoms = trajectory_storage.variables['positions'].shape[2]
        logger.info('Number of frames: {}, atoms: {}'.format(n_frames, n_atoms))

        # Determine frames to extract
        if start_frame <= 0:
            # Discard frame 0 with minimized energy which
            # throws off automatic equilibration detection.
            start_frame = 1
        if end_frame < 0:
            end_frame = n_frames + end_frame + 1
        frame_indices = range(start_frame, end_frame, skip_frame)
        if len(frame_indices) == 0:
            raise ValueError('No frames selected')
        logger.info('Extracting frames from {} to {} every {}'.format(
            start_frame, end_frame, skip_frame))

        # Discard equilibration samples
        if discard_equilibration:
            u_n = extract_u_n(reporter._storage_analysis)
            n_equil_iterations, g, n_eff = timeseries.detectEquilibration(u_n)
            logger.info(("Discarding initial {} equilibration samples (leaving {} "
                         "effectively uncorrelated samples)...").format(n_equil_iterations, n_eff))
            # Find first frame post-equilibration.
            if not full_iteration:
                for iteration in range(n_equil_iterations, n_iterations):
                    n_equil_frames = reporter._calculate_checkpoint_iteration(iteration)
                    if n_equil_frames is not None:
                        break
            else:
                n_equil_frames = n_equil_iterations
            frame_indices = frame_indices[n_equil_frames:-1]

        # Extract state positions and box vectors.
        # MDTraj Cython code expects float32 positions.
        positions = np.zeros((len(frame_indices), n_atoms, 3), dtype=np.float32)
        if is_periodic:
            box_vectors = np.zeros((len(frame_indices), 3, 3), dtype=np.float32)
        if state_index is not None:
            logger.info('Extracting positions of state {}...'.format(state_index))

            # Deconvolute state indices
            state_indices = np.zeros(len(frame_indices))
            for i, iteration in enumerate(frame_indices):
                replica_indices = reporter._storage_analysis.variables['states'][iteration, :]
                state_indices[i] = np.where(replica_indices == state_index)[0][0]

            # Extract state positions and box vectors
            for i, iteration in enumerate(frame_indices):
                replica_index = state_indices[i]
                positions[i, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                if is_periodic:
                    box_vectors[i, :, :] = trajectory_storage.variables['box_vectors'][iteration, replica_index, :, :].astype(np.float32)

        else:  # Extract replica positions and box vectors
            logger.info('Extracting positions of replica {}...'.format(replica_index))

            for i, iteration in enumerate(frame_indices):
                positions[i, :, :] = trajectory_storage.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                if is_periodic:
                    box_vectors[i, :, :] = trajectory_storage.variables['box_vectors'][iteration, replica_index, :, :].astype(np.float32)
    finally:
        reporter.close()

    # Create trajectory object
    logger.info('Creating trajectory object...')
    trajectory = mdtraj.Trajectory(positions, topology)
    if is_periodic:
        trajectory.unitcell_vectors = box_vectors

    # Force periodic boundary conditions to molecules positions
    if image_molecules:
        logger.info('Applying periodic boundary conditions to molecules positions...')
        trajectory.image_molecules(inplace=True)

    return trajectory