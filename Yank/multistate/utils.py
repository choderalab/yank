#!/usr/local/bin/env python

# ==============================================================================
# MODULE DOCSTRING
# ==============================================================================

"""
Multistate Utilities
====================

Sampling Utilities for the YANK Multistate Package. A collection of functions and small classes
which are common to help the samplers and analyzers and other public hooks.

COPYRIGHT

Current version by Andrea Rizzi <andrea.rizzi@choderalab.org>, Levi N. Naden <levi.naden@choderalab.org> and
John D. Chodera <john.chodera@choderalab.org> while at Memorial Sloan Kettering Cancer Center.

Original version by John D. Chodera <jchodera@gmail.com> while at the University of
California Berkeley.

LICENSE

This code is licensed under the latest available version of the MIT License.

"""
import logging
import numpy as np

from pymbar import timeseries  # for statistical inefficiency analysis

logger = logging.getLogger(__name__)

__all__ = [
    'generate_phase_name',
    'get_decorrelation_time',
    'get_equilibration_data',
    'get_equilibration_data_per_sample',
    'remove_unequilibrated_data',
    'subsample_data_along_axis',
    'SimulationNaNError'
]


# =============================================================================================
# Sampling Exceptions
# =============================================================================================

class SimulationNaNError(Exception):
    """Error when a simulation goes to NaN"""
    pass


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
