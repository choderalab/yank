#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
Analyzer Utilities
===================

Utilities and common functions for MultiStateSampler analyzers.


"""

# =============================================================================================
# MODULE IMPORTS
# =============================================================================================

import re
import logging
import numpy as np

from pymbar import timeseries  # for statistical inefficiency analysis

from typing import Union, Optional

logger = logging.getLogger(__name__)

__all__ = [
    'generate_phase_name',
    'get_decorrelation_time',
    'get_equilibration_data',
    'get_equilibration_data_per_sample',
    'remove_unequilibrated_data',
    'subsample_data_along_axis',
    'ObservablesRegistry'
]


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


def get_decorrelation_time(timeseries_to_analyze):
    """
    Compute the decorrelation times given a timeseries.

    See the ``pymbar.timeseries.statisticalInefficiency`` for full documentation
    """
    return timeseries.statisticalInefficiency(timeseries_to_analyze)


def get_equilibration_data(timeseries_to_analyze):
    """
    Compute equilibration method given a timeseries

    See the ``pymbar.timeseries.detectEquilibration`` function for full documentation
    """
    [n_equilibration, g_t, n_effective_max] = timeseries.detectEquilibration(timeseries_to_analyze)
    return n_equilibration, g_t, n_effective_max


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


def subsample_data_along_axis(data, subsample_rate, axis):
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


# =============================================================================================
# MODULE CLASSES
# =============================================================================================

class ObservablesRegistry(object):
    """
    Registry of computable observables.

    This is a hidden class accessed by the :class:`PhaseAnalyzer` objects to check
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
        error_class: "quad", "linear", or None
            How the error of the observable is computed when added with other errors from the same observable.

            * "quad": Adds in the quadrature, Observable C = A + B, Error eC = sqrt(eA**2 + eB**2)

            * "linear": Adds linearly,  Observable C = A + B, Error eC = eA + eB

            * None: Does not carry error

        re_register: bool, optional, Default: False
            Re-register an existing observable
        """

        self._register_observable(name, "phase", error_class, re_register=re_register)

    ########################
    # Define the observables
    ########################
    @property
    def observables(self) -> tuple:
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
    def observables_defined_by_two_states(self) -> tuple:
        """
        Observables that require an i and a j state to define the observable accurately between phases
        """
        return self._get_observables('two_state')

    @property
    def observables_defined_by_single_state(self) -> tuple:
        """
        Defined observables which are fully defined by a single state, and not by multiple states such as differences
        """
        return self._get_observables('one_state')

    @property
    def observables_defined_by_phase(self) -> tuple:
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
    def observables_with_error(self) -> tuple:
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
    def observables_with_error_adding_quadrature(self) -> tuple:
        """Observable C = A + B, Error eC = sqrt(eA**2 + eB**2)"""
        return self._get_errors('quad')

    @property
    def observables_with_error_adding_linear(self) -> tuple:
        """Observable C = A + B, Error eC = eA + eB"""
        return self._get_errors('linear')

    @property
    def observables_without_error(self) -> tuple:
        return self._get_errors(None)

    # ------------------
    # Internal functions
    # ------------------

    def _get_observables(self, key) -> tuple:
        return tuple(self._observables[key])

    def _get_errors(self, key) -> tuple:
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
ObservablesRegistry = ObservablesRegistry()
ObservablesRegistry.register_two_state_observable('free_energy', error_class='quad')
ObservablesRegistry.register_two_state_observable('entropy', error_class='quad')
ObservablesRegistry.register_two_state_observable('enthalpy', error_class='quad')
ObservablesRegistry.register_phase_observable('standard_state_correction')
