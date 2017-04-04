#!/usr/bin/env/python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Module which houses all the handling instructions for reading and writing to netCDF files for a given type.

This exists as its own module to keep the main storage module file smaller since any number of types may need to be
saved which special instructions for each.

====== WARNING ========
THIS IS VERY MUCH A WORK IN PROGRESS AND WILL PROBABLY MOSTLY BE SCRAPPED ON THE WAY

"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import abc
import copy
import netCDF4 as nc

import numpy as np
from simtk import openmm, unit

from . import utils

ABC = abc.ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3

# =============================================================================
# MODULE VARIABLES
# =============================================================================

all_handlers = []
known_types = {handler.type_string:handler for handler in all_handlers}

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

# =============================================================================
# ABSTRACT TYPE HANDLER
# =============================================================================


class NCVariableTypeHandler(ABC):
    """
    Pointer class which provides instructions on how to handle a given nc_variable
    """
    def __init__(self, nc_variable):
        self._nc_variable = nc_variable

    @abc.abstractproperty  # TODO: Depreciate when we move to Python 3 fully with @abs.abstractmethod + @property
    def _dtype(self):
        """
        Define the Python data type for this variable

        Returns
        -------
        type

        """
        return None

    @property
    def dtype(self):
        """
        Create the property which calls the protected property

        Returns
        -------
        self._dtype : type

        """
        return self._dtype

    @abc.abstractproperty
    def _dtype_type_string(self):
        """
        Short name of variable for strings and errors

        Returns
        -------
        string

        """
        return "None"

    def type_sting(self):
        """
        Read the specified string name of the nc_variable type

        Returns
        -------
        type_string : string

        """
        return self._dtype_type_string

    @abc.abstractproperty
    def value(self):
        """
        Return the property read from the ncfile

        Returns
        -------
        Given property read from the nc file and cast into the correct Python data type
        """

        raise NotImplementedError("Extracting nc_variable into `value` has not been implemented!")

    @value.setter
    def value(self, data):
        """
        Write the data to the nc_variable after ensuring that the input type matches the output type

        """

        if type(data) is not self.dtype:
            raise TypeError("Cannot set value! Expected data type is {}".format(self.type_string))
        self._cast_data_to_nc(data)
        return

    @abc.abstractmethod
    def _cast_data_to_nc(self, data):
        """
        Write to the nc_variable object after converting to the correct type.
        This may just be as simple as nc_var = data, or it may be more complicated

        """
        raise NotImplementedError("Converting input data to a netCDF writable data has not been implemented!")

# =============================================================================
# REAL TYPE HANDLERS
# =============================================================================


class NCString(NCVariableTypeHandler):

    @property
    def _dtype(self):
        return str

    @property
    def _dtype_type_string(self):
        return "str"

    def value(self):
        return str(self._nc_variable[0])

    def _cast_data_to_nc(self, data):
        self._nc_variable[0] = data


class NCInt(NCVariableTypeHandler):

    @property
    def _dtype(self):
        return int

    @property
    def _dtype_type_string(self):
        return "int"

    def value(self):
        return self._nc_variable[0]

    def _cast_data_to_nc(self, data):
        self._nc_variable[0] = data


class OptionallyQuantitiedArray(NCVariableTypeHandler):

    """
    Handle numpy arrays and possible simtk.Unit.Quantity arrays

    """
    def __init__(self, nc_variable):
        super(NCVariableTypeHandler, self).__init__(nc_variable)
        self._check_units()

    def _check_units(self):
        try:
            unit_string = self._nc_variable.units
        except AttributeError:
            unit_string = 'none'
        if unit_string == 'none':
            self.has_units = False
            self.units = None
        else:
            self.has_units = True


    @property
    def _dtype(self):
        return unit.Quantity

    @property
    def _dtype_type_string(self):
        return "Quantity"

    def value(self):
        return self._nc_variable[:]

    def _cast_data_to_nc(self, data):
        self._nc_variable = data

    @value.__getitem___
    def value(self, key):
        return self.value[key]

    @value.__setitem__
    def value(self):
        pass
