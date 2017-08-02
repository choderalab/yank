#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
Analyze
=======

Analysis tools and module for YANK simulations. Provides programmatic and automatic "best practices" integration to
determine free energy and other observables.

Fully extensible to support new samplers and observables.

"""


# =============================================================================================
# Analyze datafiles produced by YANK.
# =============================================================================================

import os
import os.path

import abc
import copy
import yaml
import numpy as np

import openmmtools as mmtools
from .repex import Reporter

from pymbar import MBAR  # multi-state Bennett acceptance ratio
from pymbar import timeseries  # for statistical inefficiency analysis

import mdtraj
import simtk.unit as units

import logging
logger = logging.getLogger(__name__)

ABC = abc.ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3

# =============================================================================================
# PARAMETERS
# =============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA


# =============================================================================================
# MODULE FUNCTIONS
# =============================================================================================

def generate_phase_name(current_name, name_list):
    """
    Provide a regular way to generate unique human-readable names from base names.

    Given a base name and a list of existing names, a number will be appended to the base name until a unique string
    is generated.

    Parameters
    ----------
    current_name : string
        The base name you wish to ensure is unique. Numbers will be appended to this string until a unique string
        not in the name_list is provided
    name_list : iterable of strings
        The current_name, and its modifiers, are compared against this list until a unique string is found

    Returns
    -------
    name : string
        Unique string derived from the current_name that is not in name_list.
        If the parameter current_name is not already in the name_list, then current_name is returned unmodified.
    """
    base_name = 'phase{}'
    counter = 0
    if current_name is None:
        name = base_name.format(counter)
        while name in name_list:
            counter += 1
            name = base_name.format(counter)
    elif current_name in name_list:
        name = current_name + str(counter)
        while name in name_list:
            counter += 1
            name = current_name + str(counter)
    else:
        name = current_name
    return name


def get_analyzer(file_base_path):
    """
    Utility function to convert storage file to a Reporter and Analyzer by reading the data on file

    For now this is mostly placeholder functions since there is only the implemented :class:`RepexPhase`, but creates
    the API for the user to work with.

    Parameters
    ----------
    file_base_path : string
        Complete path to the storage file with filename and extension.

    Returns
    -------
    analyzer : instance of implemented :class:`YankPhaseAnalyzer`
        Analyzer for the specific phase.
    """
    # Eventually extend this to get more reporters, but for now simple placeholder
    reporter = Reporter(file_base_path, open_mode='r')
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
        analyzer = RepexPhase(reporter)
    else:
        raise RuntimeError("Cannot automatically determine analyzer for Reporter: {}".format(reporter))
    return analyzer


# =============================================================================================
# MODULE CLASSES
# =============================================================================================

class _ObservablesRegistry(object):
    """
    Registry of computable observables.

    This is a hidden class accessed by the :class:`YankPhaseAnalyzer` and :class:`MultiPhaseAnalyzer` objects to check
    which observables can be computed, and then provide a regular categorization of them. This is a static registry.

    To define your own methods:
    1) Choose a unique observable name.
    2) Categorize the observable in one of the following ways by adding to the list in the "observables_X" method:
     2a) "defined_by_phase": Depends on the Phase as a whole (state independent)
     2b) "defined_by_single_state": Computed entirely from one state, e.g. Radius of Gyration
     2c) "defined_by_two_states": Property is relative to some reference state, such as Free Energy Difference
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

    ########################
    # Define the observables
    ########################
    @staticmethod
    def observables():
        """
        Set of observables which are derived from the subsets below
        """
        observables = set()
        for subset in (_ObservablesRegistry.observables_defined_by_two_states(),
                       _ObservablesRegistry.observables_defined_by_single_state(),
                       _ObservablesRegistry.observables_defined_by_phase()):
            observables = observables.union(set(subset))
        return tuple(observables)

    # ------------------------------------------------
    # Exclusive Observable categories
    # The intersection of these should be the null set
    # ------------------------------------------------
    @staticmethod
    def observables_defined_by_two_states():
        """
        Observables that require an i and a j state to define the observable accurately between phases
        """
        return 'entropy', 'enthalpy', 'free_energy'

    @staticmethod
    def observables_defined_by_single_state():
        """
        Defined observables which are fully defined by a single state, and not by multiple states such as differences
        """
        return tuple()

    @staticmethod
    def observables_defined_by_phase():
        """
        Observables which are defined by the phase as a whole, and not defined by any 1 or more states
        e.g. Standard State Correction
        """
        return 'standard_state_correction',

    ##########################################
    # Define the observables which carry error
    # This should be a subset of observables()
    ##########################################
    @staticmethod
    def observables_with_error():
        """Determine which observables have error by inspecting the the error subsets"""
        observables = set()
        for subset in (_ObservablesRegistry.observables_with_error_adding_quadrature(),
                       _ObservablesRegistry.observables_with_error_adding_linear()):
            observables = observables.union(set(subset))
        return tuple(observables)

    # ------------------------------------------------
    # Exclusive Error categories
    # The intersection of these should be the null set
    # ------------------------------------------------
    @staticmethod
    def observables_with_error_adding_quadrature():
        """Observable C = A + B, Error eC = sqrt(eA**2 + eB**2)"""
        return 'entropy', 'enthalpy', 'free_energy'

    @staticmethod
    def observables_with_error_adding_linear():
        """Observable C = A + B, Error eC = eA + eB"""
        return tuple()


# ---------------------------------------------------------------------------------------------
# Phase Analyzers
# ---------------------------------------------------------------------------------------------

class YankPhaseAnalyzer(ABC):
    """
    Analyzer for a single phase of a YANK simulation.

    Uses the reporter from the simulation to determine the location
    of all variables.

    To compute a specific observable in an implementation of this class, add it to the ObservableRegistry and then
    implement a ``get_X`` where ``X`` is the name of the observable you want to compute. See the ObservablesRegistry for
    information about formatting the observables.

    Analyzer works in units of kT unless specifically stated otherwise. To convert back to a unit set, just multiply by
    the .kT property.

    Parameters
    ----------
    reporter : Reporter instance
        Reporter from Repex which ties to the simulation data on disk.
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
        for observable in _ObservablesRegistry.observables():
            if hasattr(self, "get_" + observable):
                observables.append(observable)
                self._computed_observables[observable] = None
        # Cast observables to an immutable
        self._observables = tuple(observables)
        # Internal properties
        self._name = name
        # Start as default sign +, handle all sign conversion at peration time
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
        This should be have like the old "analyze" function from versions of YANK pre-1.0.

        Returns a dictionary of analysis objects
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_mbar_from_scratch(self):
        """
        This method should automatically do everything needed to make the MBAR object from file. It should make all
        the assumptions needed to make the MBAR object.  Typically alot of these functions will be needed for the
        analyze_phase function,

        Returns nothing, but the self.mbar object should be set after this.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_energies(self):
        """
        Extract the deconvoluted energies from a phase.

        Energies from this are NOT decorrelated.

        Returns
        -------
        sampled_energy_matrix : numpy.ndarray of shape K,K,N
            Deconvoluted energy of sampled states evaluated at other sampled states.

            Has shape (K,K,N) = (number of sampled states, number of sampled states, number of iterations)

            Indexed by [k,l,n] where an energy drawn from sampled state [k] is evaluated in sampled state [l] at
            iteration [n]
        unsampled_energy_matrix : numpy.ndarray of shape K,L,N
            Has shape (K, L, N) = (number of sampled states, number of UN-sampled states, number of iterations)

            Indexed by [k,l,n]
            where an energy drawn from sampled state [k] is evaluated in un-sampled state [l] at iteration [n]
        """
        raise NotImplementedError()

    # This SHOULD be an abstract static method since its related to the analyzer, but could handle any input data
    # Until we drop Python 2.7, we have to keep this method
    # @abc.abstractmethod
    @staticmethod
    def get_timeseries(passed_timeseries):
        """
        Generate the timeseries that is generated for this phase

        Returns
        -------
        generated_timeseries : 1-D iterable
            timeseries which can be fed into get_decorrelation_time to get the decorrelation
        """

        raise NotImplementedError("This class has not implemented this function")

    # Static methods
    @staticmethod
    def get_decorrelation_time(timeseries_to_analyze):
        """
        Compute the decorrelation times given a timeseries

        See the ``pymbar.timeseries.statisticalInefficiency`` for full documentation
        """
        return timeseries.statisticalInefficiency(timeseries_to_analyze)

    @staticmethod
    def get_equilibration_data(timeseries_to_analyze):
        """
        Compute equilibration method given a timeseries

        See the ``pymbar.timeseries.detectEquilibration`` function for full documentation
        """
        [n_equilibration, g_t, n_effective_max] = timeseries.detectEquilibration(timeseries_to_analyze)
        return n_equilibration, g_t, n_effective_max

    @staticmethod
    def get_equilibration_data_per_sample(timeseries_to_analyze, fast=True, nskip=1):
        """
        Compute the correlation time and n_effective per sample.

        This is exactly what ``pymbar.timeseries.detectEquilibration`` does, but returns the per sample data

        See the ``pymbar.timeseries.detectEquilibration`` function for full documentation
        """
        A_t = timeseries_to_analyze
        T = A_t.size
        g_t = np.ones([T - 1], np.float32)
        Neff_t = np.ones([T - 1], np.float32)
        for t in range(0, T - 1, nskip):
            try:
                g_t[t] = timeseries.statisticalInefficiency(A_t[t:T], fast=fast)
            except:
                g_t[t] = (T - t + 1)
            Neff_t[t] = (T - t + 1) / g_t[t]
        return g_t, Neff_t

    @staticmethod
    def remove_unequilibrated_data(data, number_equilibrated, axis):
        """
        Remove the number_equilibrated samples from a dataset

        Discards number_equilibrated number of indices from given axis

        Parameters
        ----------
        data : np.array-like of any dimension length
            This is the data which will be paired down
        number_equilibrated : int
            Number of indices that will be removed from the given axis, i.e. axis will be shorter by number_equilibrated
        axis : int
            Axis index along which to remove samples from. This supports negative indexing as well

        Returns
        -------
        equilibrated_data : ndarray
            Data with the number_equilibrated number of indices removed from the beginning along axis

        """
        cast_data = np.asarray(data)
        # Define the slice along an arbitrary dimension
        slc = [slice(None)] * len(cast_data.shape)
        # Set the dimension we are truncating
        slc[axis] = slice(number_equilibrated, None)
        # Slice
        equilibrated_data = cast_data[slc]
        return equilibrated_data

    @staticmethod
    def decorrelate_data(data, subsample_rate, axis):
        """
        Generate a decorrelated version of a given input data and subsample_rate along a single axis.

        Parameters
        ----------
        data : np.array-like of any dimension length
        subsample_rate : float or int
            Rate at which to draw samples. A sample is considered decorrelated after every ceil(subsample_rate) of
            indices along data and the specified axis
        axis : int
            axis along which to apply the subsampling

        Returns
        -------
        subsampled_data : ndarray of same number of dimensions as data
            Data will be subsampled along the given axis

        """
        # TODO: find a name for the function that clarifies that decorrelation
        # TODO:             is determined exclusively by subsample_rate?
        cast_data = np.asarray(data)
        data_shape = cast_data.shape
        # Since we already have g, we can just pass any appropriate shape to the subsample function
        indices = timeseries.subsampleCorrelatedData(np.zeros(data_shape[axis]), g=subsample_rate)
        subsampled_data = np.take(cast_data, indices, axis=axis)
        return subsampled_data

    # Private Class Methods
    @abc.abstractmethod
    def _prepare_mbar_input_data(self, *args, **kwargs):
        """
        Prepare a set of data for MBAR, because each analyzer may need to do something else to prepare for MBAR, it
        should have its own function to do that with.

        This is not a public function

        Parameters
        ----------
        args : arguments needed to generate the appropriate Returns
        kwargs : keyword arguments needed to generate the appropriate Returns

        Returns
        -------
        energy_matrix : energy matrix of shape (K,L,N), indexed by k,l,n
            K is the total number of sampled states
            L is the total states we want MBAR to analyze
            N is the total number of samples
            The kth sample was drawn from state k at iteration n,
                the nth configuration of kth state is evaluated in thermodynamic state l
        samples_per_state : 1-D iterable of shape L
            The total number of samples drawn from each lth state
        """
        raise NotImplementedError()

    # Shared methods
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
        :class:`YankPhaseAnalyzers`
        """
        phases = [self]
        names = []
        signs = [self._sign]
        # Reset self._sign
        self._sign = '+'
        if self.name is None:
            names.append(generate_phase_name(self, []))
        else:
            names.append(self.name)
        if isinstance(other, MultiPhaseAnalyzer):
            new_phases = other.phases
            new_signs = other.signs
            new_names = other.names
            final_new_names = []
            for name in new_names:
                other_names = [n for n in new_names if n != name]
                final_new_names.append(generate_phase_name(name, other_names + names))
            names.extend(final_new_names)
            for new_sign in new_signs:
                if operator != '+' and new_sign == '+':
                    signs.append('-')
                else:
                    signs.append('+')
            phases.extend(new_phases)
        elif isinstance(other, YankPhaseAnalyzer):
            names.append(generate_phase_name(other.name, names))
            if operator != '+' and other._sign == '+':
                signs.append('-')
            else:
                signs.append('+')
            # Reset the other's sign if it got set to negative
            other._sign = '+'
            phases.append(other)
        else:
            baseerr = "cannot {} 'YankPhaseAnalyzer' and '{}' objects"
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
        """Internally handle the internal sign"""
        if self._sign == '+':
            self._sign = '-'
        else:
            self._sign = '+'
        return self


class RepexPhase(YankPhaseAnalyzer):

    """
    The RepexPhase is the analyzer for a simulation generated from a Replica Exchange sampler simulation, implemented
    as an instance of the :class:`YankPhaseAnalyzer`.

    See Also
    --------
    YankPhaseAnalyzer

    """

    def generate_mixing_statistics(self, number_equilibrated=0):
        """
        Generate the Transition state matrix and sorted eigenvalues

        Parameters
        ----------
        number_equilibrated : int, optional, default=0
           If specified, only samples number_equilibrated:end will be used in analysis

        Returns
        -------
        mixing_stats : np.array of shape [nstates, nstates]
            Transition matrix estimate
        mu : np.array
            Eigenvalues of the Transition matrix sorted in descending order
        """

        # Read data from disk
        states = self._reporter.read_replica_thermodynamic_states()
        n_iterations, n_states = states.shape
        n_ij = np.zeros([n_states, n_states], np.int64)

        # Compute empirical transition count matrix.
        for iteration in range(number_equilibrated, n_iterations - 1):
            for i_replica in range(n_states):
                i_state = states[iteration, i_replica]
                j_state = states[iteration + 1, i_replica]
                n_ij[i_state, j_state] += 1

        # Compute transition matrix estimate.
        # TODO: Replace with maximum likelihood reversible count estimator from msmbuilder or pyemma.
        t_ij = np.zeros([n_states, n_states], np.float64)
        for i_state in range(n_states):
            # Cast to float to ensure we dont get integer division
            denominator = float((n_ij[i_state, :].sum() + n_ij[:, i_state].sum()))
            if denominator > 0:
                for j_state in range(n_states):
                    t_ij[i_state, j_state] = (n_ij[i_state, j_state] + n_ij[j_state, i_state]) / denominator
            else:
                t_ij[i_state, i_state] = 1.0

        # Estimate eigenvalues
        mu = np.linalg.eigvals(t_ij)
        mu = -np.sort(-mu)  # Sort in descending order

        return t_ij, mu

    def show_mixing_statistics(self, cutoff=0.05, number_equilibrated=0):
        """
        Print summary of mixing statistics. Passes information off to generate_mixing_statistics then prints it out to
        the logger

        Parameters
        ----------
        cutoff : float, optional, default=0.05
           Only transition probabilities above 'cutoff' will be printed
        number_equilibrated : int, optional, default=0
           If specified, only samples number_equilibrated:end will be used in analysis

        """

        Tij, mu = self.generate_mixing_statistics(number_equilibrated=number_equilibrated)

        # Print observed transition probabilities.
        nstates = Tij.shape[1]
        logger.info("Cumulative symmetrized state mixing transition matrix:")
        str_row = "{:6s}".format("")
        for jstate in range(nstates):
            str_row += "{:6d}".format(jstate)
        logger.info(str_row)

        for istate in range(nstates):
            str_row = ""
            str_row += "{:-6d}".format(istate)
            for jstate in range(nstates):
                P = Tij[istate, jstate]
                if P >= cutoff:
                    str_row += "{:6.3f}".format(P)
                else:
                    str_row += "{:6s}".format("")
            logger.info(str_row)

        # Estimate second eigenvalue and equilibration time.
        if mu[1] >= 1:
            logger.info("Perron eigenvalue is unity; Markov chain is decomposable.")
        else:
            logger.info("Perron eigenvalue is {0:9.5f}; state equilibration timescale is ~ {1:.1f} iterations".format(
                mu[1], 1.0 / (1.0 - mu[1]))
            )

    def extract_energies(self):
        """
        Extract and decorrelate energies from the ncfile to gather energies common data for other functions

        Returns
        -------
        energy_matrix : ndarray of shape (K,K,N)
            Potential energy matrix of the sampled states
            Energy is from each sample n, drawn from state (first k), and evaluated at every sampled state (second k)
        unsampled_energy_matrix : ndarray of shape (K,L,N)
            Potential energy matrix of the unsampled states
            Energy from each sample n, drawn from sampled state k, evaluated at unsampled state l
            If no unsampled states were drawn, this will be shape (K,0,N)

        """
        logger.info("Reading energies...")
        energy_thermodynamic_states, energy_unsampled_states = self._reporter.read_energies()
        n_iterations, _, n_states = energy_thermodynamic_states.shape
        _, _, n_unsampled_states = energy_unsampled_states.shape
        energy_matrix_replica = np.zeros([n_states, n_states, n_iterations], np.float64)
        unsampled_energy_matrix_replica = np.zeros([n_states, n_unsampled_states, n_iterations], np.float64)
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
        mbar_ready_energy_matrix = sampled_energy_matrix
        if nunsampled > 0:
            fully_interacting_u_ln = unsampled_energy_matrix[:, 0, :]
            noninteracting_u_ln = unsampled_energy_matrix[:, 1, :]
            # Augment u_kln to accept the new state
            new_energy_matrix = np.zeros([nstates + 2, nstates + 2, N], np.float64)
            N_k_new = np.zeros(nstates + 2, np.int32)
            # Insert energies
            new_energy_matrix[1:-1, 0, :] = fully_interacting_u_ln
            new_energy_matrix[1:-1, -1, :] = noninteracting_u_ln
            # Fill in other energies
            new_energy_matrix[1:-1, 1:-1, :] = sampled_energy_matrix
            N_k_new[1:-1] = N_k
            # Notify users
            logger.info("Found expanded cutoff states in the energies!")
            logger.info("Free energies will be reported relative to them instead!")
            # Reset values, last step in case something went wrong so we dont overwrite u_kln on accident
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
            (Deltaf_ij, dDeltaf_ij, theta_ij) = self.mbar.getFreeEnergyDifferences()

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

    def _create_mbar_from_scratch(self):
        u_kln, unsampled_u_kln = self.extract_energies()
        u_n = self.get_timeseries(u_kln)

        # Discard equilibration samples.
        # TODO: if we include u_n[0] (the energy right after minimization) in the equilibration detection,
        # TODO:         then number_equilibrated is 0. Find a better way than just discarding first frame.
        number_equilibrated, g_t, Neff_max = self.get_equilibration_data(u_n[1:])
        self._equilibration_data = number_equilibrated, g_t, Neff_max
        u_kln = self.remove_unequilibrated_data(u_kln, number_equilibrated, -1)
        unsampled_u_kln = self.remove_unequilibrated_data(unsampled_u_kln, number_equilibrated, -1)

        # decorrelate_data subsample the energies only based on g_t so both ends up with same indices.
        u_kln = self.decorrelate_data(u_kln, g_t, -1)
        unsampled_u_kln = self.decorrelate_data(unsampled_u_kln, g_t, -1)

        mbar_ukln, mbar_N_k = self._prepare_mbar_input_data(u_kln, unsampled_u_kln)
        self._create_mbar(mbar_ukln, mbar_N_k)

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


# https://choderalab.slack.com/files/levi.naden/F4G6L9X8S/quick_diagram.png

class MultiPhaseAnalyzer(object):
    """
    Multiple Phase Analyzer creator, not to be directly called itself, but instead called by adding or subtracting
    different implemented :class:`YankPhaseAnalyzer` or other :class:`MultiPhaseAnalyzers`'s. The individual Phases of
    the :class:`MultiPhaseAnalyzer` are only references to existing Phase objects, not copies. All
    :class:`YankPhaseAnalyzer` and :class:`MultiPhaseAnalyzer` classes support ``+`` and ``-`` operations.

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
        for observable in _ObservablesRegistry.observables():
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
        """List of implemented :class:`YankPhaseAnalyzer`'s objects this :class:`MultiPhaseAnalyzer` is tied to"""
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
        :class:`YankPhaseAnalyzer` or a :class:`MultiPhaseAnalyzer`.

        If the names have collision, they are re-named with an extra digit at the end.

        Parameters
        ----------
        other : :class:`MultiPhaseAnalyzer` or :class:`YankPhaseAnalyzer`
        operator : sign of the operator connecting the two objects

        Returns
        -------
        output : :class:`MultiPhaseAnalyzer`
            New :class:`MultiPhaseAnalyzer` where the phases are the combined list of the individual phases from each
            component. Because the memory pointers to the individual phases are the same, changing any
            single :class:`YankPhaseAnalyzer`'s
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
                final_new_names.append(generate_phase_name(name, other_names + names))
            names.extend(final_new_names)
            for new_sign in new_signs:
                if (operator == '-' and new_sign == '+') or (operator == '+' and new_sign == '-'):
                    signs.append('-')
                else:
                    signs.append('+')
            signs.extend(new_signs)
            phases.extend(new_phases)
        elif isinstance(other, YankPhaseAnalyzer):
            names.append(generate_phase_name(other.name, names))
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
                if observable_name in _ObservablesRegistry.observables_with_error():
                    observable_payload = {}
                    observable_payload['value'], observable_payload['error'] = observable
                else:
                    observable_payload = observable
            else:
                raise_registry_error = False
                if observable_name in _ObservablesRegistry.observables_with_error():
                    observable_payload = {}
                    if observable_name in _ObservablesRegistry.observables_defined_by_phase():
                        observable_payload['value'], observable_payload['error'] = observable
                    elif observable_name in _ObservablesRegistry.observables_defined_by_single_state():
                        observable_payload['value'] = observable[0][single_phase.reference_states[0]]
                        observable_payload['error'] = observable[1][single_phase.reference_states[0]]
                    elif observable_name in _ObservablesRegistry.observables_defined_by_two_states():
                        observable_payload['value'] = observable[0][single_phase.reference_states[0],
                                                                    single_phase.reference_states[1]]
                        observable_payload['error'] = observable[1][single_phase.reference_states[0],
                                                                    single_phase.reference_states[1]]
                    else:
                        raise_registry_error = True
                else:  # No error
                    if observable_name in _ObservablesRegistry.observables_defined_by_phase():
                        observable_payload = observable
                    elif observable_name in _ObservablesRegistry.observables_defined_by_single_state():
                        observable_payload = observable[single_phase.reference_states[0]]
                    elif observable_name in _ObservablesRegistry.observables_defined_by_two_states():
                        observable_payload = observable[single_phase.reference_states[0],
                                                        single_phase.reference_states[1]]
                    else:
                        raise_registry_error = True
                if raise_registry_error:
                    raise RuntimeError("You have requested an observable that is improperly registered in the "
                                       "ObservablesRegistry!")
            return observable_payload

        def modify_final_output(passed_output, payload, sign):
            if observable_name in _ObservablesRegistry.observables_with_error():
                if sign == '+':
                    passed_output['value'] += payload['value']
                else:
                    passed_output['value'] -= payload['value']
                if observable_name in _ObservablesRegistry.observables_with_error_adding_linear():
                    passed_output['error'] += payload['error']
                elif observable_name in _ObservablesRegistry.observables_with_error_adding_quadrature():
                    passed_output['error'] = (passed_output['error']**2 + payload['error']**2)**0.5
            else:
                if sign == '+':
                    passed_output += payload
                else:
                    passed_output -= payload
            return passed_output

        if observable_name in _ObservablesRegistry.observables_with_error():
            final_output = {'value': 0, 'error': 0}
        else:
            final_output = 0
        for phase, phase_sign in zip(self.phases, self.signs):
            phase_observable = prepare_phase_observable(phase)
            final_output = modify_final_output(final_output, phase_observable, phase_sign)
        if observable_name in _ObservablesRegistry.observables_with_error():
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
    logger.info("")
    logger.info("Free energy{}: {:16.3f} +- {:.3f} kT ({:16.3f} +- {:.3f} kcal/mol)".format(
        calculation_type, DeltaF, dDeltaF, DeltaF * kT / units.kilocalories_per_mole,
        dDeltaF * kT / units.kilocalories_per_mole))
    logger.info("")

    for phase in phase_names:
        logger.info("DeltaG {:<25} : {:16.3f} +- {:.3f} kT".format(phase, data[phase]['DeltaF'],
                                                                   data[phase]['dDeltaF']))
        if data[phase]['DeltaF_standard_state_correction'] != 0.0:
            logger.info("DeltaG {:<25} : {:25.3f} kT".format('restraint',
                                                             data[phase]['DeltaF_standard_state_correction']))
    logger.info("")
    logger.info("Enthalpy{}: {:16.3f} +- {:.3f} kT ({:16.3f} +- {:.3f} kcal/mol)".format(
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
        u_kln_replica[:,:,n] = energies[n,:,:]
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
        u_n[iteration] = np.sum(np.diagonal(u_kln[:,:,iteration]))

    return u_n


# ==============================================================================
# Extract trajectory from NetCDF4 file
# ==============================================================================

def extract_trajectory(output_path, nc_path, nc_checkpoint_file=None, state_index=None, replica_index=None,
                       start_frame=0, end_frame=-1, skip_frame=1, keep_solvent=True,
                       discard_equilibration=False, image_molecules=False):
    """Extract phase trajectory from the NetCDF4 file.

    Parameters
    ----------
    output_path : str
        Path to the trajectory file to be created. The extension of the file
        determines the format.
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

    """
    # Check correct input
    if (state_index is None) == (replica_index is None):
        raise ValueError('One and only one between "state_index" and '
                         '"replica_index" must be specified.')
    if not os.path.isfile(nc_path):
        raise ValueError('Cannot find file {}'.format(nc_path))

    # Import simulation data
    try:
        reporter = Reporter(nc_path, open_mode='r', checkpoint_storage_file=nc_checkpoint_file)
        metadata = reporter.read_dict('metadata')
        reference_system = mmtools.utils.deserialize(metadata['reference_state']).system
        topology = mmtools.utils.deserialize(metadata['topography']).topology

        # Determine if system is periodic
        is_periodic = reference_system.usesPeriodicBoundaryConditions()
        logger.info('Detected periodic boundary conditions: {}'.format(is_periodic))

        # Get dimensions
        n_iterations = reporter._storage_checkpoint.variables['positions'].shape[0]
        n_atoms = reporter._storage_checkpoint.variables['positions'].shape[2]
        logger.info('Number of frames: {}, atoms: {}'.format(n_iterations, n_atoms))

        # Determine frames to extract
        if start_frame <= 0:
            # Discard frame 0 with minimized energy which
            # throws off automatic equilibration detection.
            start_frame = 1
        if end_frame < 0:
            end_frame = n_iterations + end_frame + 1
        frame_indices = range(start_frame, end_frame, skip_frame)
        if len(frame_indices) == 0:
            raise ValueError('No frames selected')
        logger.info('Extracting frames from {} to {} every {}'.format(
            start_frame, end_frame, skip_frame))

        # Discard equilibration samples
        if discard_equilibration:
            u_n = extract_u_n(reporter._storage_analysis)[frame_indices]
            n_equil, g, n_eff = timeseries.detectEquilibration(u_n)
            logger.info(("Discarding initial {} equilibration samples (leaving {} "
                         "effectively uncorrelated samples)...").format(n_equil, n_eff))
            frame_indices = frame_indices[n_equil:-1]

        # Extract state positions and box vectors
        positions = np.zeros((len(frame_indices), n_atoms, 3))
        if is_periodic:
            box_vectors = np.zeros((len(frame_indices), 3, 3))
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
                positions[i, :, :] = reporter._storage_checkpoint.variables['positions'][i, replica_index, :, :].astype(np.float32)
                if is_periodic:
                    box_vectors[i, :, :] = reporter._storage_checkpoint.variables['box_vectors'][i, replica_index, :, :].astype(np.float32)

        else:  # Extract replica positions and box vectors
            logger.info('Extracting positions of replica {}...'.format(replica_index))

            for i, iteration in enumerate(frame_indices):
                positions[i, :, :] = reporter._storage_checkpoint.variables['positions'][i, replica_index, :, :].astype(np.float32)
                if is_periodic:
                    box_vectors[i, :, :] = reporter._storage_checkpoint.variables['box_vectors'][i, replica_index, :, :].astype(np.float32)
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

    # Remove solvent
    if not keep_solvent:
        logger.info('Removing solvent molecules...')
        trajectory = trajectory.remove_solvent()

    # Detect format
    extension = os.path.splitext(output_path)[1][1:]  # remove dot
    try:
        save_function = getattr(trajectory, 'save_' + extension)
    except AttributeError:
        raise ValueError('Cannot detect format from extension of file {}'.format(output_path))

    # Create output directory and save trajectory
    logger.info('Creating trajectory file: {}'.format(output_path))
    output_dir = os.path.dirname(output_path)
    if output_dir != '' and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    save_function(output_path)
