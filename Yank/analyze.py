#!/usr/local/bin/env python

# =============================================================================================
# Analyze datafiles produced by YANK.
# =============================================================================================

# =============================================================================================
# REQUIREMENTS
#
# The netcdf4-python module is now used to provide netCDF v4 support:
# http://code.google.com/p/netcdf4-python/
#
# This requires NetCDF with version 4 and multithreading support, as well as HDF5.
# =============================================================================================

import os
import os.path

import yaml
import numpy as np

from .repex import Reporter
import netCDF4 as netcdf  # netcdf4-python

from pymbar import MBAR  # multistate Bennett acceptance ratio
from pymbar import timeseries  # for statistical inefficiency analysis

import mdtraj
import simtk.unit as units

import abc

from . import utils

import logging
logger = logging.getLogger(__name__)

ABC = abc.ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3

# =============================================================================================
# PARAMETERS
# =============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA


def generate_phase_name(current_name, name_list):
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
            counter +=1
            name = current_name + str(counter)
    else:
        name = current_name
    return name


class ObservablesRegitry(object):
    """
    Registry of computable observables.
    """
    @staticmethod
    def observables():
        """
        Set of observables which are derived from the subsets below
        """
        observables = set()
        for subset in (ObservablesRegitry.observables_defined_by_two_states(),
                       ObservablesRegitry.observables_defined_by_single_state(),
                       ObservablesRegitry.observables_defined_by_phase()):
            observables = observables.union(set(subset))
        return tuple(observables)

    # Non-exclusive properties
    @staticmethod
    def observables_with_error():
        observables = set()
        for subset in (ObservablesRegitry.observables_with_error_adding_quadrature(),
                       ObservablesRegitry.observables_with_error_adding_linear()):
            observables = observables.union(set(subset))
        return tuple(observables)

    @staticmethod
    def observables_with_error_adding_quadrature():
        return 'entropy', 'enthalpy', 'free_energy'

    @staticmethod
    def observables_with_error_adding_linear():
        return tuple()

    # Exclusive observables
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
        return 'restraints'


class YankPhaseAnalyzer(ABC):
    """
    Analyzer for a single phase of a YANK simulation. Uses the reporter from the simulation to determine the location
    of all variables.

    To compute a specific observable, add it to the ObservableRegistry and then implement a "compute_X" where X is the
    name of the observable you want to compute.
    """
    def __init__(self, reporter, name=None, sign=None, delta_state_i=0, delta_state_j=-1):
        """
        The reporter provides the hook into how to read the data, all other options control where differences are
        measured from and how each phase interfaces with other phases.

        Parameters
        ----------
        reporter : Reporter instance
            Reporter from Repex which ties to the simulation data on disk.
        name : str, Optional
            Unique name you want to assign this phase, this is the name that will appear in CompoundPhase's. If not
            set, it will be given the arbitrary name "phase#" where # is an integer, chosen in order that it is
            assigned to the CompoundPhase.
        sign : str, Optional, default '+'
            Representation of the operator for combining observables in the CompoundPhase. If no sign is set at the
            time a CompoundPhase is created, the sign is assumed to be positive. In the CompoundPhase, this sign is
            tracked internally so this Phase can be used in other CompoundPhases.
        delta_state_i: int, Optional Default: 0
        delta_state_j: int, Optional Default: -1
            Integers of the state that is used for reference in observables, "O". These values are only used when
            reporting single numbers or combining observables through CompoundPhase (since the number of states between
            phases can be different). Calls to functions such as `get_free_energy` in a single Phase results in the
            O being returned for all states.
            For O completely defined by the state itself (i.e. no differences between states, e.g. Temperature)"
                O[i] is returned
                O[j] is not used
            For O where differences between states are required (e.g. Free Energy):
                O[i,j] = O[j] - O[i]
        """
        if not reporter.is_open():
            reporter.open(mode='r')
        self._reporter = reporter
        observables = []
        # Auto-determine the computable observables by inspection of non-flagged methods
        # We determine valid observables by negation instead of just having each child implement the method to enforce
        # uniform function naming conventions.
        self._computed_observables = {}  # Cache of observables so the phase can be retrieved once computed
        for observable in ObservablesRegitry.observables():
            if hasattr(self, "get_" + observable):
                observables.append(observable)
                self._computed_observables[observable] = None
        # Cast observables to an immutable
        self._observables = tuple(observable)
        # Internal properties
        self._name = name
        if sign is not None:
            sign = self._check_sign(sign)
        self._sign = sign
        self._equilibration_data = None  # Internal tracker so the functions can get this data without recalculating it
        # External properties
        self.delta_state_i = delta_state_i
        self.delta_state_j = delta_state_j
        self.mbar = None

    @staticmethod
    def _check_sign(value):
        """
        Internal function which validates the sign parameter passed in.
        """
        plus_set = ['+', 'plus', 'add', 'positive']
        minus_set = ['-', 'minus', 'subtract', 'negative']
        if value in plus_set:
            output = '+'
        elif value in minus_set:
            output = '-'
        else:
            raise ValueError("Please use one of the following for 'sign': {}".format(plus_set + minus_set))
        return output

    # Property management
    @property
    def sign(self):
        return self._sign

    @sign.setter
    def sign(self, value):
        self._sign = self._check_sign(value)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def observables(self):
        return self._observables

    # Abstract methods
    @abc.abstractmethod
    def analyze_phase(self, *args, **kwargs):
        """
        Function which broadly handles "auto-analysis" for those that do not wish to call all the methods on their own.
        This should be have like the old "analyze" function from versions of YANK pre-1.0.

        Returns a dictionary of analysis objects
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def create_MBAR_from_scratch(self):
        """
        This method should automatically do everything needed to make the MBAR object from file. It should make all
        the assumptions needed to make the MBAR object.  Typically alot of these functions will be needed for the
        analyze_phase function,

        Returns nothing, but the self.mbar object should be set after this
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_u_kln(self):
        """Return the u_kln matrix of energies at sampled """

    @abc.abstractmethod
    def extract_energies(self):
        """
        Extract the deconvoluted energies from a phase. Energies from this are NOT decorrelated.

        Returns
        -------
        u_kln : Deconvoluted energy of sampled states evaluated at other sampled states.
            Has shape (K,K,N) = (number of sampled states, number of sampled states, number of iterations)
            Indexed by [k,l,n]
            where an energy drawn from sampled state [k] is evaluated in sampled state [l] at iteration [n]
        unsampled_u_kln
            Has shape (K, L, N) = (number of sampled states, number of UN-sampled states, number of iterations)
            Indexed by [k,l,n]
            where an energy drawn from sampled state [k] is evaluated in un-sampled state [l] at iteration [n]
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_timeseries(self, u_kln):
        """
        Generate the timeseries that will be

        Returns
        -------
        generated_timeseries : 1-D iterable
            timeseries which can be fed into get_decorrelation_time to get the decorrelation
        """

        raise NotImplementedError

    @abc.abstractmethod
    def prepare_MBAR_input_data(self, *args, **kwargs):
        """
        Prepare a set of data for MBAR, because each method may need to do something else to prepare for MBAR, it
        should have its own function to do that with=

        Parameters
        ----------
        args: whatever is needed to generate the appropriate outputs

        Returns
        -------
        u_kln : energy matrix of shape (K,L,N), indexed by k,l,n
            K is the total number of sampled states
            L is the total states we want MBAR to analyze
            N is the total number of samples
            The kth sample was drawn from state k at iteration n,
                the nth configuration of kth state is evaluated in thermodynamic state l
        N_k: The total number of samples drawn from each lth state
        """
        raise NotImplementedError()


    # Shared methods
    def create_MBAR(self, u_kln, N_k):
        """
        Initialize MBAR for Free Energy and Enthalpy estimates, this may take a while.

        u_kln : array of numpy.float64, optional, default=None
           Reduced potential energies of the replicas; if None, will be extracted from the ncfile
        N_k : array of ints, optional, default=None
           Number of samples drawn from each kth replica; if None, will be extracted from the ncfile

        TODO
        ----
        * Ensure that the u_kln and N_k are decorrelated if not provided in this function

        """

        # Delete observables cache since we are now resetting the estimator
        for observable in self.observables():
            self._computed_observables[observable] = None

        # Initialize MBAR (computing free energy estimates, which may take a while)
        logger.info("Computing free energy differences...")
        mbar = MBAR(u_kln, N_k)

        self.mbar = mbar

    def _combine_phases(self, other, operator='+'):
        phases = [self]
        names = []
        signs = []
        if self.name is None:
            names.append(generate_phase_name(self, []))
        if self.sign is None:
            signs.append('+')
        else:
            signs.append(self.sign)
        if isinstance(other, CompoundPhase):
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
            if other.sign is None:
                if operator == '+':
                    signs.append('+')
                else:
                    signs.append('-')
            else:
                if operator == '+':
                    signs.append(other.sign)
                elif operator == '-' and other.sign == '+':
                    signs.append('-')
                else:
                    signs.append('+')
            phases.append(other)
        else:
            baseerr = "cannot {} 'CompoundPhase' and '{}' objects"
            if operator == '+':
                err = baseerr.format('add', type(other))
            else:
                err = baseerr.format('subtract', type(other))
            raise TypeError(err)
        phase_pass = {'phases': phases, 'signs': signs, 'names': names}
        return CompoundPhase(phase_pass)

    def __add__(self, other):
        return self._combine_phases(other, operator='+')

    def __sub__(self, other):
        return self._combine_phases(other, operator='-')


    # Static methods
    @staticmethod
    def get_decorrelation_time(timeseries_to_analyze):
        return timeseries.statisticalInefficiency(timeseries_to_analyze)

    @staticmethod
    def get_equilibration_data(timeseries_to_analyze):
        [nequil, g_t, Neff_max] = timeseries.detectEquilibration(timeseries_to_analyze)
        return nequil, g_t, Neff_max

    @staticmethod
    def remove_unequilibrated_data(data, nequil, axis):
        """
        Remove the unequilbrated samples from a dataset by discarding nequil number of indices from given axis


        Parameters
        ----------
        data: np.array-like of any dimension length
        nequil: int
            Number of indices that will be removed from the given axis, i.e. axis will be shorter by nequil
        axis: int
            axis index along wich to remove samples from

        Returns
        -------
        equilibrated_data: ndarray
            Data with the nequil number of indices removed from the begining along axis

        """
        cast_data = np.asarray(data)
        # Define the sluce along an arbitrary dimension
        slc = [slice(None)] * len(cast_data.shape)
        # Set the dimension we are truncating
        slc[axis] = slice(nequil, None)
        # Slice
        equilibrated_data = cast_data[slc]
        return equilibrated_data

    @staticmethod
    def decorrelate_data(data, subsample_rate, axis):
        """
        Generate a decorrelated version of a given input data and subsample_rate along a single axis.

        Parameters
        ----------
        data: np.array-like of any dimension length
        subsample_rate : float or int
            Rate at which to draw samples. A sample is considered decorrelated after every ceil(subsample_rate) of
            indicies along data and the specified axis
        axis: int
            axis along which to apply the subsampling

        Returns
        -------
        subsampled_data : ndarray of same number of dimensions as data
            Data will be subsampled along the given axis

        """
        cast_data = np.asarray(data)
        data_shape = cast_data.shape
        # Since we already have g, we can just pass any appropriate shape to the subsample function
        indices = timeseries.subsampleCorrelatedData(np.zeros(data_shape[axis]), g=subsample_rate)
        subsampled_data = np.take(cast_data, indices, axis=axis)
        return subsampled_data




class RepexPhase(YankPhaseAnalyzer):

    def generate_mixing_statistics(self, nequil=0):
        """
        Generate the mixing statistics

        Parameters
        ----------
        nequil : int, optional, default=0
           If specified, only samples nequil:end will be used in analysis

        Returns
        -------
        mixing_stats : np.array of shape [nstates, nstates]
            Transition matrix estimate
        mu : np.array
            Eigenvalues of the Transition matrix sorted in descending order
        """

        # Get mixing stats from reporter
        n_accepted_matrix, n_proposed_matrix = self._reporter.read_mixing_statistics()
        # Add along iteration dim
        n_accepted_matrix = n_accepted_matrix[nequil:].sum(axis=0).astype(float)  # Ensure float division
        n_proposed_matrix = n_proposed_matrix[nequil:].sum(axis=0)
        # Compute empirical transition count matrix
        Tij = 1 - n_accepted_matrix/n_proposed_matrix

        # Estimate eigenvalues
        mu = np.linalg.eigvals(Tij)
        mu = -np.sort(-mu)  # Sort in descending order

        return Tij, mu

    def show_mixing_statistics(self, cutoff=0.05, nequil=0):
        """
        Print summary of mixing statistics. Passes information off to generate_mixing_statistics then prints it out to
        the logger

        Parameters
        ----------

        cutoff : float, optional, default=0.05
           Only transition probabilities above 'cutoff' will be printed
        nequil : int, optional, default=0
           If specified, only samples nequil:end will be used in analysis

        """

        Tij, mu = self.generate_mixing_statistics(nequil=nequil)

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

        return

    def extract_energies(self):
        """
        Extract and decorelate energies from the ncfile to gather energies common data for other functions=

        """
        logger.info("Reading energies...")
        energy_thermodynamic_states, energy_unsampled_states = self._reporter.read_energies()
        niterations, nstates, _ = energy_thermodynamic_states.shape
        _, n_unsampled_states, _ = energy_unsampled_states.shape
        u_kln_replica = np.zeros([nstates, nstates, niterations], np.float64)
        unsampled_u_kln_replica = np.zeros([nstates, n_unsampled_states, niterations], np.float64)
        for n in range(niterations):
            u_kln_replica[:, :, n] = energy_thermodynamic_states[n, :, :]
            unsampled_u_kln_replica[:, :, n] = energy_unsampled_states[n, :, :]
        logger.info("Done.")

        logger.info("Deconvoluting replicas...")
        u_kln = np.zeros([nstates, nstates, niterations], np.float64)
        unsampled_u_kln = np.zeros([nstates, n_unsampled_states, niterations], np.float64)
        for iteration in range(niterations):
            state_indices = self._reporter.read_replica_thermodynamic_states(iteration)
            u_kln[state_indices, :, iteration] = u_kln_replica[iteration, :, :]
            unsampled_u_kln[state_indices, :, iteration] = unsampled_u_kln_replica[iteration, :, :]
        logger.info("Done.")

        return u_kln, unsampled_u_kln

    def get_timeseries(self, u_kln):
        niterations = u_kln.shape[-1]
        u_n = np.zeros([niterations], np.float64)
        # Compute total negative log probability over all iterations.
        for iteration in range(niterations):
            u_n[iteration] = np.sum(np.diagonal(u_kln[:, :, iteration]))
        return u_n

    def prepare_MBAR_input_data(self, u_kln_sampled, u_kln_unsampled, nuse=0):
        nstates, _, niterations = u_kln_sampled.shape
        _, nunsampled, _ = u_kln_sampled.shape
        # Subsample data to obtain uncorrelated samples
        N_k = np.zeros(nstates, np.int32)
        # print(u_n) # DEBUG
        # indices = range(0,u_n.size) # DEBUG - assume samples are uncorrelated
        N = len(niterations)  # number of uncorrelated samples
        N_k[:] = N
        u_kln = u_kln_sampled
        if nunsampled > 0:
            fully_interacting_u_ln = u_kln_unsampled[:, 0, :]
            noninteracting_u_ln = u_kln_unsampled[:, 1, :]
            # Augment u_kln to accept the new state
            u_kln_new = np.zeros([nstates + 2, nstates + 2, N], np.float64)
            N_k_new = np.zeros(nstates + 2, np.int32)
            # Insert energies
            u_kln_new[1:-1, 0, :] = fully_interacting_u_ln
            u_kln_new[1:-1, -1, :] = noninteracting_u_ln
            # Fill in other energies
            u_kln_new[1:-1, 1:-1, :] = u_kln_sampled
            N_k_new[1:-1] = N_k
            # Notify users
            logger.info("Found expanded cutoff states in the energies!")
            logger.info("Free energies will be reported relative to them instead!")
            # Reset values, last step in case something went wrong so we dont overwrite u_kln on accident
            u_kln = u_kln_new
            N_k = N_k_new
        return u_kln, N_k

    def _compute_free_energy(self):
        """
        Estimate free energies of all alchemical states.

        Parameters
        ----------
        """

        # Create MBAR object if not provided
        if self.mbar is None:
            raise RuntimeError("Cannot compute free energy without MBAR object!")

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
        Return the free energy and error in free energy from the MBAR object

        Returns
        -------
        DeltaF_ij : Free energy from delta_f
        dDeltaF_ij: Error in the free energy estimate.

        """
        if self._computed_observables['free_energy'] is None:
            self._compute_free_energy()
        free_energy_dict = self._computed_observables['free_energy']
        return free_energy_dict['value'], free_energy_dict['error']

    def _compute_enthalpy_and_entropy(self):
        (f_k, df_k, H_k, dH_k, S_k, dS_k) = self.mbar.computeEntropyAndEnthalpy()
        enthalpy = {'value': H_k, 'error': dH_k}
        entropy = {'value': S_k, 'error': dS_k}
        self._computed_observables['enthalpy'] = enthalpy
        self._computed_observables['entropy'] = entropy

    def get_enthalpy(self):
        """
        Return the difference in enthalpy and error in that estimate from the MBAR object
        """
        if self._computed_observables['enthalpy'] is None:
            self._compute_enthalpy_and_entropy()
        enthalpy_dict = self._computed_observables['enthalpy']
        return enthalpy_dict['value'], enthalpy_dict['error']

    def get_entropy(self):
        """
        Return the difference in entropy and error in that estimate from the MBAR object]
        """
        if self._computed_observables['entropy'] is None:
            self._compute_enthalpy_and_entropy()
        entropy_dict = self._computed_observables['entropy']
        return entropy_dict['value'], entropy_dict['error']

    def get_restraints(self):
        """
        Compute the restraint free energy associated with the Reporter.
        This usually is just a stored variable, but it may need other calculations

        Returns
        DeltaF_restratints: Free energy contribution from the restraints
        -------

        """
        raise NotImplementedError()  #TODO: Figure out how this is brought into the new reporter with the new restraints
        if self._computed_observables['restraints'] is None:
            self._computed_observables['restraints'] = None # TODO: Do something here
        return self._computed_observables['restraints']

    def create_MBAR_from_scratch(self):
        u_kln, unsampled_u_kln = self.extract_energies()
        u_n = self.get_timeseries(u_kln)
        nequil, g_t, Neff_max = self.get_equilibration_data(u_n)
        self._equilbration_data = nequil, g_t, Neff_max
        self.show_mixing_statistics(cutoff=cutoff, nequil=nequil)
        u_kln = self.remove_unequilibrated_data(u_kln, nequil, -1)
        unsampled_u_kln = self.remove_unequilibrated_data(unsampled_u_kln, nequil, -1)
        u_kln = self.decorrelate_data(u_kln, g_t, -1)
        unsampled_u_kln = self.decorrelate_data(unsampled_u_kln, g_t, -1)
        mbar_ukln, mbar_N_k = self.prepare_MBAR_input_data(u_kln, unsampled_u_kln)
        self.create_MBAR(mbar_ukln, mbar_N_k)

    def analyze_phase(self, cutoff=0.05):
        if self.mbar is None:
            self.create_MBAR_from_scratch()
        nequil, g_t, _ = self._equilibration_data
        self.show_mixing_statistics(cutoff=cutoff, nequil=nequil)
        data = {}
        # Accumulate free energy differences
        Deltaf_ij, dDeltaf_ij = self.get_free_energy()
        DeltaH_ij, dDeltaH_ij = self.get_enthalpy()
        data['DeltaF'] = Deltaf_ij[self.delta_state_i, self.delta_state_j]
        data['dDeltaF'] = dDeltaf_ij[self.delta_state_i, self.delta_state_j]
        data['DeltaH'] = DeltaH_ij[self.delta_state_i, self.delta_state_j]
        data['dDeltaH'] = dDeltaH_ij[self.delta_state_i, self.delta_state_j]
        data['DeltaF_restraints'] = self.get_restraints()

        # Do something to get temperatures here
        return data  #, kT?


def get_analyzer(reporter):
    """
    Utility function to convert Reporter to Analyzer by reading the data on file
    """
    if reporter.storage_convention == "ReplicaExchange":
        analyzer = RepexPhase(reporter)
    else:
        raise RuntimeError("Cannot automatically determine analyzer for Reporter: {}".format(Reporter))
    return analyzer


# https://choderalab.slack.com/files/levi.naden/F4G6L9X8S/quick_diagram.png

class CompoundPhase(object):
    """
    Combined Phase creator, not to be directly called itself, but instead called by adding or subtracting different
    Phases or CompoundPhases's
    """
    def __init__(self, phases):
        """
        Create the compound phase which is any combination of phases to generate a new compound phase.
        Parameters
        ----------
        phases: dict
            has keys "phases", "names", and "signs" which control the signs
        """
        # Determine
        observables = []
        for observable in ObservablesRegitry.observables():
            shared_observable = True
            for phase in phases['phases']:
                if observable not in phase.observables:
                    shared_observable = False
                    break
            if shared_observable:
                observables.append(observable)
        self._observables = tuple(observables)
        self._phases = phases['phases']
        self._names = phases['names']
        self._signs = phases['signs']
        # Set the methods shared between both objects
        for observable in self.observables:
            setattr(self, "get_" + observable, lambda: self._compute_observable(observable))

    @property
    def observables(self):
        return self._observables

    @property
    def phases(self):
        return self._phases

    @property
    def names(self):
        return self._names

    @property
    def signs(self):
        return self._signs

    def _combine_phases(self, other, operator='+'):
        phases = []
        names = []
        signs = []
        # create copies
        for phase in self.phases:
            phases.append(phase)
        for name in self.names:
            names.append(name)
        for sign in self.signs:
            signs.append(sign)
        if isinstance(other, CompoundPhase):
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
            signs.extend(new_signs)
            phases.extend(new_phases)
        elif isinstance(other, YankPhaseAnalyzer):
            names.append(generate_phase_name(other.name, names))
            if other.sign is None:
                if operator == '+':
                    signs.append('+')
                else:
                    signs.append('-')
            else:
                if operator == '+':
                    signs.append(other.sign)
                elif operator == '-' and other.sign == '+':
                    signs.append('-')
                else:
                    signs.append('+')
            phases.append(other)
        else:
            baseerr = "cannot {} 'CompoundPhase' and '{}' objects"
            if operator == '+':
                err = baseerr.format('add', type(other))
            else:
                err = baseerr.format('subtract', type(other))
            raise TypeError(err)
        phase_pass = {'phases': phases, 'signs': signs, 'names': names}
        return CompoundPhase(phase_pass)

    def __add__(self, other):
        return self._combine_phases(other, operator='+')

    def __sub__(self, other):
        return self._combine_phases(other, operator='-')

    def _compute_observable(self, observable_name):
        """
        Helper function to compute arbitrary observable in both phases

        Parameters
        ----------
        observable_name: str
            Name of the observable as its defined in the ObservablesRegistry

        Returns
        -------
        observable_value
            The observable as its combined between the two phases

        """
        def prepare_phase_observable(single_phase):
            """Helper function to cast the observable in terms of observable's registry"""
            observable = getattr(single_phase, "get_" + observable_name)()
            if isinstance(single_phase, CompoundPhase):
                if observable_name in ObservablesRegitry.observables_with_error():
                    observable_payload = {}
                    observable_payload['value'], observable_payload['error'] = observable
                else:
                    observable_payload = observable
            else:
                raise_registry_error = False
                if observable_name in ObservablesRegitry.observables_with_error():
                    observable_payload = {}
                    if observable_name in ObservablesRegitry.observables_defined_by_phase():
                        observable_payload['value'], observable_payload['error'] = observable
                    elif observable_name in ObservablesRegitry.observables_defined_by_single_state():
                        observable_payload['value'] = observable[0][single_phase.delta_state_i]
                        observable_payload['error'] = observable[1][single_phase.delta_state_i]
                    elif observable_name in ObservablesRegitry.observables_defined_by_two_states():
                        observable_payload['value'] = observable[0][single_phase.delta_state_i,
                                                                    single_phase.delta_state_j]
                        observable_payload['error'] = observable[1][single_phase.delta_state_i,
                                                                    single_phase.delta_state_j]
                    else:
                        raise_registry_error = True
                else:  # No error
                    if observable_name in ObservablesRegitry.observables_defined_by_phase():
                        observable_payload = observable
                    elif observable_name in ObservablesRegitry.observables_defined_by_single_state():
                        observable_payload = observable[single_phase.delta_state_i]
                    elif observable_name in ObservablesRegitry.observables_defined_by_two_states():
                        observable_payload = observable[single_phase.delta_state_i, single_phase.delta_state_j]
                    else:
                        raise_registry_error = True
                if raise_registry_error:
                    raise RuntimeError("You have requested an observable that is improperly registered in the "
                                       "ObservablesRegistry!")
            return observable_payload

        def modify_final_output(passed_output, payload, sign):
            if observable_name in ObservablesRegitry.observables_with_error():
                if sign == '+':
                    passed_output['value'] += payload['value']
                else:
                    passed_output['value'] -= payload['value']
                if observable_name in ObservablesRegitry.observables_with_error_adding_linear():
                    passed_output['error'] += payload['error']
                elif observable_name in ObservablesRegitry.observables_with_error_adding_quadrature():
                    passed_output['error'] += (passed_output['error']**2 + payload['error']**2)**0.5
            else:
                if sign == '+':
                    passed_output += payload
                else:
                    passed_output -= payload
            return passed_output

        if observable_name in ObservablesRegitry.observables_with_error():
            final_output = {'value': 0, 'error': 0}
        else:
            final_output = 0
        for phase, phase_sign in zip(self.phases, self.signs):
            phase_observable = prepare_phase_observable(phase)
            final_output = modify_final_output(final_output, phase_observable, phase_sign)
        return final_output

# =============================================================================================
# SUBROUTINES
# =============================================================================================


def generate_mixing_statistics(ncfile, nequil=0):
    """
    Generate the mixing statistics

    Parameters
    ----------
    ncfile : netCDF4.Dataset
       NetCDF file
    nequil : int, optional, default=0
       If specified, only samples nequil:end will be used in analysis

    Returns
    -------
    mixing_stats : np.array of shape [nstates, nstates]
        Transition matrix estimate
    mu : np.array
        Eigenvalues of the Transition matrix sorted in descending order
    """

    # Get dimensions.
    niterations = ncfile.variables['states'].shape[0]
    nstates = ncfile.variables['states'].shape[1]

    # Compute empirical transition count matrix.
    Nij = np.zeros([nstates, nstates], np.float64)
    for iteration in range(nequil, niterations-1):
        for ireplica in range(nstates):
            istate = ncfile.variables['states'][iteration, ireplica]
            jstate = ncfile.variables['states'][iteration+1, ireplica]
            Nij[istate, jstate] += 1

    # Compute transition matrix estimate.
    # TODO: Replace with maximum likelihood reversible count estimator from msmbuilder or pyemma.
    Tij = np.zeros([nstates,nstates], np.float64)
    for istate in range(nstates):
        denom = (Nij[istate,:].sum() + Nij[:,istate].sum())
        if denom > 0:
            for jstate in range(nstates):
                Tij[istate, jstate] = (Nij[istate, jstate] + Nij[jstate, istate]) / denom
        else:
            Tij[istate, istate] = 1.0

    # Estimate eigenvalues
    mu = np.linalg.eigvals(Tij)
    mu = -np.sort(-mu)  # Sort in descending order

    return Tij, mu


def show_mixing_statistics(ncfile, cutoff=0.05, nequil=0):
    """
    Print summary of mixing statistics. Passes information off to generate_mixing_statistics then prints it out to
    the logger

    Parameters
    ----------

    ncfile : netCDF4.Dataset
       NetCDF file
    cutoff : float, optional, default=0.05
       Only transition probabilities above 'cutoff' will be printed
    nequil : int, optional, default=0
       If specified, only samples nequil:end will be used in analysis

    """

    Tij, mu = generate_mixing_statistics(ncfile, nequil=nequil)

    # Print observed transition probabilities.
    nstates = ncfile.variables['states'].shape[1]
    logger.info("Cumulative symmetrized state mixing transition matrix:")
    str_row = "%6s" % ""
    for jstate in range(nstates):
        str_row += "%6d" % jstate
    logger.info(str_row)

    for istate in range(nstates):
        str_row = ""
        str_row += "%-6d" % istate
        for jstate in range(nstates):
            P = Tij[istate,jstate]
            if P >= cutoff:
                str_row += "%6.3f" % P
            else:
                str_row += "%6s" % ""
        logger.info(str_row)

    # Estimate second eigenvalue and equilibration time.
    if mu[1] >= 1:
        logger.info("Perron eigenvalue is unity; Markov chain is decomposable.")
    else:
        logger.info("Perron eigenvalue is {0:9.5f}; state equilibration timescale is ~ {1:.1f} iterations".format(
            mu[1], 1.0 / (1.0 - mu[1]))
        )

    return


def extract_ncfile_energies(ncfile, ndiscard=0, nuse=None, g=None):
    """
    Extract and decorelate energies from the ncfile to gather common data for other functions

    Parameters
    ----------
    ncfile : NetCDF
       Input YANK netcdf file
    ndiscard : int, optional, default=0
       Number of iterations to discard to equilibration
    nuse : int, optional, default=None
       Maximum number of iterations to use (after discarding)
    g : int, optional, default=None
       Statistical inefficiency to use if desired; if None, will be computed.

    TODO
    ----
    * Automatically determine 'ndiscard'.

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
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = energies[iteration,:,:]
    logger.info("Done.")

    # Compute total negative log probability over all iterations.
    u_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        u_n[iteration] = np.sum(np.diagonal(u_kln[:,:,iteration]))

    # Discard initial data to equilibration.
    u_kln_replica = u_kln_replica[:,:,ndiscard:]
    u_kln = u_kln[:,:,ndiscard:]
    u_n = u_n[ndiscard:]

    # Truncate to number of specified conforamtions to use
    if (nuse):
        u_kln_replica = u_kln_replica[:,:,0:nuse]
        u_kln = u_kln[:,:,0:nuse]
        u_n = u_n[0:nuse]

    # Subsample data to obtain uncorrelated samples
    N_k = np.zeros(nstates, np.int32)
    indices = timeseries.subsampleCorrelatedData(u_n, g=g) # indices of uncorrelated samples
    #print(u_n) # DEBUG
    #indices = range(0,u_n.size) # DEBUG - assume samples are uncorrelated
    N = len(indices) # number of uncorrelated samples
    N_k[:] = N
    u_kln = u_kln[:,:,indices]
    logger.info("number of uncorrelated samples:")
    logger.info(N_k)
    logger.info("")

    # Check for the expanded cutoff states, and subsamble as needed
    try:
        u_ln_full_raw = ncfile.variables['fully_interacting_expanded_cutoff_energies'][:].T #Its stored as nl, need in ln
        u_ln_non_raw = ncfile.variables['noninteracting_expanded_cutoff_energies'][:].T 
        fully_interacting_u_ln = np.zeros(u_ln_full_raw.shape)
        noninteracting_u_ln = np.zeros(u_ln_non_raw.shape)
        # Deconvolute the fully interacting state
        for iteration in range(niterations):
            state_indices = ncfile.variables['states'][iteration,:]
            fully_interacting_u_ln[state_indices,iteration] = u_ln_full_raw[:,iteration]
            noninteracting_u_ln[state_indices,iteration] = u_ln_non_raw[:,iteration]
        # Discard non-equilibrated samples
        fully_interacting_u_ln = fully_interacting_u_ln[:,ndiscard:]
        fully_interacting_u_ln = fully_interacting_u_ln[:,indices]
        noninteracting_u_ln = noninteracting_u_ln[:,ndiscard:]
        noninteracting_u_ln = noninteracting_u_ln[:,indices]
        # Augment u_kln to accept the new state
        u_kln_new = np.zeros([nstates + 2, nstates + 2, N], np.float64)
        N_k_new = np.zeros(nstates + 2, np.int32)
        # Insert energies
        u_kln_new[1:-1,0,:] = fully_interacting_u_ln
        u_kln_new[1:-1,-1,:] = noninteracting_u_ln
        # Fill in other energies
        u_kln_new[1:-1,1:-1,:] = u_kln 
        N_k_new[1:-1] = N_k
        # Notify users
        logger.info("Found expanded cutoff states in the energies!")
        logger.info("Free energies will be reported relative to them instead!")
        # Reset values, last step in case something went wrong so we dont overwrite u_kln on accident
        u_kln = u_kln_new
        N_k = N_k_new
    except:
        pass

    return u_kln, N_k, u_n


def initialize_MBAR(ncfile, u_kln=None, N_k=None):
    """
    Initialize MBAR for Free Energy and Enthalpy estimates, this may take a while.

    ncfile : NetCDF
       Input YANK netcdf file
    u_kln : array of numpy.float64, optional, default=None
       Reduced potential energies of the replicas; if None, will be extracted from the ncfile
    N_k : array of ints, optional, default=None
       Number of samples drawn from each kth replica; if None, will be extracted from the ncfile

    TODO
    ----
    * Ensure that the u_kln and N_k are decorrelated if not provided in this function

    """

    if u_kln is None or N_k is None:
        (u_kln, N_k, u_n) = extract_ncfile_energies(ncfile)

    # Initialize MBAR (computing free energy estimates, which may take a while)
    logger.info("Computing free energy differences...")
    mbar = MBAR(u_kln, N_k)
    
    return mbar
    

def estimate_free_energies(ncfile, mbar=None):
    """
    Estimate free energies of all alchemical states.

    Parameters
    ----------
    ncfile : NetCDF
       Input YANK netcdf file
    mbar : pymbar MBAR object, optional, default=None
       Initilized MBAR object from simulations; if None, it will be generated from the ncfile
    
    """

    # Create MBAR object if not provided
    if mbar is None:
        mbar = initialize_MBAR(ncfile)

    nstates = mbar.N_k.size

    # Get matrix of dimensionless free energy differences and uncertainty estimate.
    logger.info("Computing covariance matrix...")

    try:
        # pymbar 2
        (Deltaf_ij, dDeltaf_ij) = mbar.getFreeEnergyDifferences()
    except ValueError:
        # pymbar 3
        (Deltaf_ij, dDeltaf_ij, theta_ij) = mbar.getFreeEnergyDifferences()

    # Matrix of free energy differences
    logger.info("Deltaf_ij:")
    for i in range(nstates):
        str_row = ""
        for j in range(nstates):
            str_row += "%8.3f" % Deltaf_ij[i, j]
        logger.info(str_row)

    # Matrix of uncertainties in free energy difference (expectations standard
    # deviations of the estimator about the true free energy)
    logger.info("dDeltaf_ij:")
    for i in range(nstates):
        str_row = ""
        for j in range(nstates):
            str_row += "%8.3f" % dDeltaf_ij[i, j]
        logger.info(str_row)

    # Return free energy differences and an estimate of the covariance.
    return Deltaf_ij, dDeltaf_ij


def estimate_enthalpies(ncfile, mbar=None):
    """
    Estimate enthalpies of all alchemical states.

    Parameters
    ----------
    ncfile : NetCDF
       Input YANK netcdf file
    mbar : pymbar MBAR object, optional, default=None
       Initilized MBAR object from simulations; if None, it will be generated from the ncfile

    TODO
    ----
    * Check if there is an output/function name difference between pymbar 2 and 3
    """

    # Create MBAR object if not provided
    if mbar is None:
        mbar = initialize_MBAR(ncfile)

    nstates = mbar.N_k.size

    # Compute average enthalpies
    (f_k, df_k, H_k, dH_k, S_k, dS_k) = mbar.computeEntropyAndEnthalpy()

    return H_k, dH_k


def extract_u_n(ncfile):
    """
    Extract timeseries of u_n = - log q(X_n) from store file

    where q(X_n) = \pi_{k=1}^K u_{s_{nk}}(x_{nk})

    with X_n = [x_{n1}, ..., x_{nK}] is the current collection of replica configurations
    s_{nk} is the current state of replica k at iteration n
    u_k(x) is the kth reduced potential

    Parameters
    ----------
    ncfile : str
       The filename of the repex NetCDF file.

    Returns
    -------
    u_n : numpy array of numpy.float64
       u_n[n] is -log q(X_n)

    TODO
    ----
    Move this to repex.

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
        state_indices = ncfile.variables['states'][iteration,:]
        u_kln[state_indices,:,iteration] = energies[iteration,:,:]
    logger.info("Done.")

    # Compute total negative log probability over all iterations.
    u_n = np.zeros([niterations], np.float64)
    for iteration in range(niterations):
        u_n[iteration] = np.sum(np.diagonal(u_kln[:,:,iteration]))

    return u_n

# =============================================================================================
# SHOW STATUS OF STORE FILES
# =============================================================================================


def print_status(store_directory):
    """
    Print a quick summary of simulation progress.

    Parameters
    ----------
    store_directory : string
       The location of the NetCDF simulation output files.

    Returns
    -------
    success : bool
       True is returned on success; False if some files could not be read.

    """
    # Get NetCDF files
    phases = utils.find_phases_in_store_directory(store_directory)

    # Process each netcdf file.
    for phase, fullpath in phases.items():

        # Check that the file exists.
        if not os.path.exists(fullpath):
            # Report failure.
            logger.info("File %s not found." % fullpath)
            logger.info("Check to make sure the right directory was specified, and 'yank setup' has been run.")
            return False

        # Open NetCDF file for reading.
        logger.debug("Opening NetCDF trajectory file '%(fullpath)s' for reading..." % vars())
        ncfile = netcdf.Dataset(fullpath, 'r')

        # Read dimensions.
        niterations = ncfile.variables['positions'].shape[0]
        nstates = ncfile.variables['positions'].shape[1]
        natoms = ncfile.variables['positions'].shape[2]

        # Print summary.
        logger.info("%s" % phase)
        logger.info("  %8d iterations completed" % niterations)
        logger.info("  %8d alchemical states" % nstates)
        logger.info("  %8d atoms" % natoms)

        # TODO: Print average ns/day and estimated completion time.

        # Close file.
        ncfile.close()

    return True

# =============================================================================================
# ANALYZE STORE FILES
# =============================================================================================


def analyze(source_directory):
    """
    Analyze contents of store files to compute free energy differences.

    Parameters
    ----------
    source_directory : string
       The location of the NetCDF simulation storage files.

    """
    analysis_script_path = os.path.join(source_directory, 'analysis.yaml')
    if not os.path.isfile(analysis_script_path):
        err_msg = 'Cannot find analysis.yaml script in {}'.format(source_directory)
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    with open(analysis_script_path, 'r') as f:
        analysis = yaml.load(f)
    phases = [phase_name for phase_name, sign in analysis]

    # Storage for different phases.
    data = dict()

    # Process each netcdf file.
    for phase in phases:
        ncfile_path = os.path.join(source_directory, phase + '.nc')

        # Open NetCDF file for reading.
        logger.info("Opening NetCDF trajectory file %(ncfile_path)s for reading..." % vars())
        try:
            ncfile = netcdf.Dataset(ncfile_path, 'r')

            logger.debug("dimensions:")
            for dimension_name in ncfile.dimensions.keys():
                logger.debug("%16s %8d" % (dimension_name, len(ncfile.dimensions[dimension_name])))

            # Read dimensions.
            niterations = ncfile.variables['positions'].shape[0]
            nstates = ncfile.variables['positions'].shape[1]
            logger.info("Read %(niterations)d iterations, %(nstates)d states" % vars())

            DeltaF_restraints = 0.0
            if 'metadata' in ncfile.groups:
                # Read phase direction and standard state correction free energy.
                # Yank sets correction to 0 if there are no restraints
                DeltaF_restraints = ncfile.groups['metadata'].variables['standard_state_correction'][0]

            # Choose number of samples to discard to equilibration
            MIN_ITERATIONS = 10 # minimum number of iterations to use automatic detection
            if niterations > MIN_ITERATIONS:
                from pymbar import timeseries
                u_n = extract_u_n(ncfile)
                u_n = u_n[1:] # discard initial frame of zero energies TODO: Get rid of initial frame of zero energies
                [nequil, g_t, Neff_max] = timeseries.detectEquilibration(u_n)
                nequil += 1 # account for initial frame of zero energies
                logger.info([nequil, Neff_max])
            else:
                nequil = 1  # discard first frame
                g_t = 1
                Neff_max = niterations

            # Examine acceptance probabilities.
            show_mixing_statistics(ncfile, cutoff=0.05, nequil=nequil)

            # Extract equilibrated, decorrelated energies, check for fully interacting state
            (u_kln, N_k, u_n) = extract_ncfile_energies(ncfile, ndiscard=nequil, g=g_t)

            # Create MBAR object to use for free energy and entropy states
            mbar = initialize_MBAR(ncfile, u_kln=u_kln, N_k=N_k)

            # Estimate free energies, use fully interacting state if present
            (Deltaf_ij, dDeltaf_ij) = estimate_free_energies(ncfile, mbar=mbar)

            # Estimate average enthalpies
            (DeltaH_i, dDeltaH_i) = estimate_enthalpies(ncfile, mbar=mbar)

            # Accumulate free energy differences
            entry = dict()
            entry['DeltaF'] = Deltaf_ij[0, -1]
            entry['dDeltaF'] = dDeltaf_ij[0, -1]
            entry['DeltaH'] = DeltaH_i[0, -1]
            entry['dDeltaH'] = dDeltaH_i[0, -1]
            entry['DeltaF_restraints'] = DeltaF_restraints
            data[phase] = entry

            # Get temperatures.
            ncvar = ncfile.groups['thermodynamic_states'].variables['temperatures']
            temperature = ncvar[0] * units.kelvin
            kT = kB * temperature

        finally:
            ncfile.close()

    # Compute free energy and enthalpy
    DeltaF = 0.0
    dDeltaF = 0.0
    DeltaH = 0.0
    dDeltaH = 0.0
    for phase, sign in analysis:
        DeltaF -= sign * (data[phase]['DeltaF'] + data[phase]['DeltaF_restraints'])
        dDeltaF += data[phase]['dDeltaF']**2
        DeltaH -= sign * (data[phase]['DeltaH'] + data[phase]['DeltaF_restraints'])
        dDeltaH += data[phase]['dDeltaH']**2
    dDeltaF = np.sqrt(dDeltaF)
    dDeltaH = np.sqrt(dDeltaH)

    # Attempt to guess type of calculation
    calculation_type = ''
    for phase in phases:
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

    for phase in phases:
        logger.info("DeltaG {:<25} : {:16.3f} +- {:.3f} kT".format(phase, data[phase]['DeltaF'],
                                                                   data[phase]['dDeltaF']))
        if data[phase]['DeltaF_restraints'] != 0.0:
            logger.info("DeltaG {:<25} : {:25.3f} kT".format('restraint',
                                                             data[phase]['DeltaF_restraints']))
    logger.info("")
    logger.info("Enthalpy{}: {:16.3f} +- {:.3f} kT ({:16.3f} +- {:.3f} kcal/mol)".format(
        calculation_type, DeltaH, dDeltaH, DeltaH * kT / units.kilocalories_per_mole,
        dDeltaH * kT / units.kilocalories_per_mole))


# ==============================================================================
# Extract trajectory from NetCDF4 file
# ==============================================================================

def extract_trajectory(output_path, nc_path, state_index=None, replica_index=None,
                       start_frame=0, end_frame=-1, skip_frame=1, keep_solvent=True,
                       discard_equilibration=False, image_molecules=False):
    """Extract phase trajectory from the NetCDF4 file.

    Parameters
    ----------
    output_path : str
        Path to the trajectory file to be created. The extension of the file
        determines the format.
    nc_path : str
        Path to the NetCDF4 file containing the trajectory.
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
        nc_file = netcdf.Dataset(nc_path, 'r')

        # Extract topology and system serialization
        serialized_system = nc_file.groups['metadata'].variables['reference_system'][0]
        serialized_topology = nc_file.groups['metadata'].variables['topology'][0]

        # Determine if system is periodic
        from simtk import openmm
        reference_system = openmm.XmlSerializer.deserialize(str(serialized_system))
        is_periodic = reference_system.usesPeriodicBoundaryConditions()
        logger.info('Detected periodic boundary conditions: {}'.format(is_periodic))

        # Get dimensions
        n_iterations = nc_file.variables['positions'].shape[0]
        n_atoms = nc_file.variables['positions'].shape[2]
        logger.info('Number of iterations: {}, atoms: {}'.format(n_iterations, n_atoms))

        # Determine frames to extract
        if start_frame <= 0:
            # TODO yank saves first frame with 0 energy!
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
            u_n = extract_u_n(nc_file)[frame_indices]
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
                replica_indices = nc_file.variables['states'][iteration, :]
                state_indices[i] = np.where(replica_indices == state_index)[0][0]

            # Extract state positions and box vectors
            for i, iteration in enumerate(frame_indices):
                replica_index = state_indices[i]
                positions[i, :, :] = nc_file.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                if is_periodic:
                    box_vectors[i, :, :] = nc_file.variables['box_vectors'][iteration, replica_index, :, :].astype(np.float32)

        else:  # Extract replica positions and box vectors
            logger.info('Extracting positions of replica {}...'.format(replica_index))

            for i, iteration in enumerate(frame_indices):
                positions[i, :, :] = nc_file.variables['positions'][iteration, replica_index, :, :].astype(np.float32)
                if is_periodic:
                    box_vectors[i, :, :] = nc_file.variables['box_vectors'][iteration, replica_index, :, :].astype(np.float32)
    finally:
        nc_file.close()

    # Create trajectory object
    logger.info('Creating trajectory object...')
    topology = utils.deserialize_topology(serialized_topology)
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
