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
import collections

from simtk import openmm, unit

from . import utils
from .utils import typename, quantity_from_string

# TODO: Use the `with_metaclass` from yank.utils when we merge it in
ABC = abc.ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3


# =============================================================================
# MODULE VARIABLES
# =============================================================================

all_handlers = []
known_types = {handler.type_string: handler for handler in all_handlers}

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def decompose_path(path):
    """

    Parameters
    ----------
    path : string
        Path to variable on the

    Returns
    -------
    structure : tuple
        Tuple of split apart path
    """
    return tuple((path_entry for path_entry in path.split('/') if path_entry != ''))

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


# =============================================================================
# ABSTRACT DRIVER HANDLER
# =============================================================================

class StorageIODriver(ABC):
    """
    Abstract class to define the basic functions any storage driver needs to read/write to the disk.
    The specific driver for a type of storage should be a subclass of this with its own
    encoders and decoders for specific file types.
    """
    def __init__(self, file_name, access_mode='w'):
        """

        Parameters
        ----------
        file_name : string
            Name of the file to read/write to of a given storage type
        access_mode : string, Default 'w', accepts 'w', 'r', 'a'
            Define how to access the file in either write, read, or append mode
        """
        # Internal map from Python Type <-> De/Encoder which handles the actual encoding and decoding of the data
        self._type_maps = {}
        self._variables = {}

    def add_deencoder(self, type_key, de_encoder):
        """
        Add new De/Encode to the specific driver class. This coder must know how to read/write and append to disk.

        Parameters
        ----------
        type_key : Unique immutable object
            Unique key that will be added to identify this de_encoder as part of the class
        de_encoder : Specific DeEnCoder class
            Class to handle all of the encoding of decoding of the variables

        """
        self._type_maps[type_key] = de_encoder

    @abc.abstractmethod
    def create_variable(self, path, type_key):
        """
        Create a new variable on the disk and at the path location and store it as the given type.

        Parameters
        ----------
        path : string
            The way to identify the variable on the storage system. This can be either a variable name or a full path
            (such as in NetCDF files)
        type_key : Immutable object
            Type specifies the key identifier in the _type_maps added by the add_deencoder function. If type is not in
            _type_maps variable, an error is raised.

        Returns
        -------
        bound_de_encoder : DeEnCoder which is linked to a specific reference on the disk.

        """
        raise NotImplementedError("create_variable has not been implemented!")


# =============================================================================
# NetCDF IO Driver
# =============================================================================


class NetCDFIODriver(StorageIODriver):
    """
    Driver to handle all NetCDF IO operations, variable creation, and other operations.
    Can be extended to add new or modified type handlers
    """
    def get_netcdf_group(self, path):
        """
        Get the top level group on the NetCDF file, create the full path if not present

        Parameters
        ----------
        path : string
            Path to group on the disk

        Returns
        -------
        group : NetCDF Group
            Group object requested from file. All subsequent groups are created on the way down and can be accessed
            the same way.
        """
        try:
            group = self._groups[path]
        except KeyError:
            group = self._bind_group(path)
        finally:
            return group

    def get_variable_handler(self, path):
        """
        Get a variable IO object from disk at path. Raises an error if no storage object exists at that level

        Parameters
        ----------
        path : string
            Path to the variable/storage object on disk

        Returns
        -------
        handler : Subclass of NCVariableTypeHandler
            The handler tied to a specific variable and bound to it on the disk

        """
        try:
            handler = self._variables[path]
        except KeyError:
            try:
                # Attempt to read the disk and bind to that variable
                head_group = self.ncfile
                split_path = decompose_path(path)
                for header in split_path[:-1]:
                    head_group = head_group.groups[header]
                # Check if this is a group type
                is_group = False
                if split_path[-1] in head_group.groups:
                    try:
                        obj = head_group.groups[split_path[-1]]
                        store_type = obj.getncattr('IODriver_Storage_Type')
                        if store_type == 'group':
                            variable = obj
                            is_group = True
                    except AttributeError:
                        pass
                if not is_group:
                    variable = head_group.variables[split_path[-1]]
            except KeyError as e:
                raise KeyError("No variable found at {} on file!".format(path))
            try:
                data_type = variable.getncattr('IODriver_Type')
                head_path = '/' + '/'.join(split_path[:-1])
                target_name = split_path[-1]
                group = self._bind_group(head_path)
                handler_unset = self._IOMetaDataReaders[data_type]
                self._variables[path] = handler_unset(self, target_name, storage_object=group)
                handler = self._variables[path]
                # Bind the variable since we know its real
                handler._bind_read()
            except AttributeError:
                raise AttributeError("Cannot auto-detect variable type, ensure that 'IODriver_Type' is a set ncattr")
            except KeyError:
                raise KeyError("No mapped type handler known for 'IODriver_Type' = '{}'".format(data_type))
        return handler

    def create_variable(self, path, type_key):
        try:
            handler = self._type_maps[type_key]
        except KeyError:
            raise KeyError("No known Type Handler for given type!")
        split_path = decompose_path(path)
        # Bind groups as needed, splitting off the last entry
        # Ensure the head path has at least a '/' at the start
        head_path = '/' + '/'.join(split_path[:-1])
        target_name = split_path[-1]
        group = self._bind_group(head_path)
        self._variables[path] = handler(self, target_name, storage_object=group)
        return self._variables[path]

    @staticmethod
    def check_scalar_dimension(ncfile):
        """
        Check that the `scalar` dimension exists on file and create it if not

        """
        if 'scalar' not in ncfile.dimensions:
            ncfile.createDimension('scalar', 1)  # scalar dimension

    @staticmethod
    def check_infinite_dimension(ncfile, name='iteration'):
        """
        Check that the arbitrary infinite dimension exists on file and create it if not.

        Parameters
        ----------
        ncfile : NetCDF File
        name : string, optional, Default = 'iteration'
            Name of the dimension

        """
        if name not in ncfile.dimensions:
            ncfile.createDimension(name, 0)

    def add_metadata(self, name, value):
        """
        Add metadata to self on disk, extra bits of information that can be used for flags or other variables

        Parameters
        ----------
        name : string
            Name of the attribute you wish to assign
        value : any, but prefered string
            Extra meta data to add to the variable
        """
        self.ncfile.setncattr(name, value)

    def _bind_group(self, path):
        """
        Bind a group to a particular path on the nc file. Note that this method creates the cascade of groups all the
        way to the final object if it can.

        Parameters
        ----------
        path : string
            Absolute path to the group as it appears on the NetCDF file.

        Returns
        -------
        group : NetCDF Group
            The group that path points to. Can be accessed by path through the ._groups dictionary after binding

        """
        # NetCDF4 creates the cascade of groups automatically or returns the group if already present
        # To simplify code, the cascade of groups is not stored in this class until called
        self._groups[path] = self.ncfile.createGroup(path)
        return self._groups[path]

    def __init__(self, file_name, access_mode='w'):
        super(NetCDFIODriver, self).__init__(file_name, access_mode=access_mode)
        # Bind to file
        self.ncfile = nc.Dataset(file_name, access_mode)
        self._groups = {}
        # Bind all of the Type Handlers
        self.add_deencoder(str, NCString)  # String
        # Int
        self.add_deencoder(dict, NCDict)  # Dict
        # List
        # Array
        # Quantity
        # Units?
        # OpenMM System
        # Bind the metadata reader types based on the dtype string of each class
        self._IOMetaDataReaders = {self._type_maps[key]._dtype_type_string(): self._type_maps[key] for key in self._type_maps}


# =============================================================================
# ABSTRACT TYPE HANDLER
# =============================================================================


class NCVariableTypeHandler(ABC):
    """
    Pointer class which provides instructions on how to handle a given nc_variable
    """
    def __init__(self, parent_handler, target, storage_object=None):
        """
        Bind to a given nc_storage_object on ncfile with given final_target_name,
        If no nc_storage_object is None, it defaults to the top level ncfile
        Parameters
        ----------
        parent_handler : Parent NetCDF handler
            Class which can manipulate the NetCDF file at the top level for dimension creation and meta handling
        target : string
            String of the name of the object. Not explicitly a variable nor a group since the object could be either
        storage_object : NetCDF file or NetCDF group, optional, Default to ncfile on parent_handler
            Object the variable/object will be written onto

        """
        self._target = target
        # Eventual NetCDF object this class will be bound to
        self._bound_target = None
        self._parent_handler = parent_handler
        if storage_object is None:
            storage_object = self._parent_handler.ncfile
        self._storage_object = storage_object
        self._metadata_buffer = {}
        self._can_append = False
        self._can_write = False

    @abc.abstractproperty  # TODO: Depreciate when we move to Python 3 fully with @abc.abstractmethod + @property
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

    # @abc.abstractproperty
    @staticmethod
    def _dtype_type_string():
        """
        Short name of variable for strings and errors

        Returns
        -------
        string

        """
        # TODO: Replace with @abstractstaticmethod when on Python 3
        raise NotImplementedError("_dtype_type_string has not been implemented in this subclass yet!")

    @property
    def type_string(self):
        """
        Read the specified string name of the nc_variable type

        Returns
        -------
        type_string : string

        """
        return self._dtype_type_string()

    @abc.abstractmethod
    def _bind_read(self):
        """
        A one time event that binds this class to the object on disk. This method should set self._bound_target
        This function is unique to the read() function in that no data is attempted to write to the disk.
        Should raise error if the object is not found on disk (i.e. no data has been written to this location yet)

        Returns
        -------
        None, but should set self._bound_target
        """

    @abc.abstractmethod
    def _bind_write(self, None):
        """
        A one time event that binds this class to the object on disk. This method should set self._bound_target
        This function is unique to the write() function in that the data passed in should overwrite what is at
        the location, or should create the object, then write the data.

        Parameters
        ----------
        data : any data to save, optional, Default None

        Returns
        -------
        None, but should set self._bound_target
        """

    @abc.abstractmethod
    def _bind_append(self, None):
        """
        A one time event that binds this class to the object on disk. This method should set self._bound_target
        This function is unique to the append() function in that the data passed in should append what is at
        the location, or should create the object, then write the data with the first dimension infinite in size

        Parameters
        ----------
        data : any data to save, optional, Default None

        Returns
        -------
        None, but should set self._bound_target
        """

    @abc.abstractmethod
    def read(self):
        """
        Return the property read from the ncfile

        Returns
        -------
        Given property read from the nc file and cast into the correct Python data type
        """

        raise NotImplementedError("Extracting stored NetCDF data into Python data has not been implemented!")

    @abc.abstractmethod
    def write(self, data):
        """
        Tell this writer how to write to the NetCDF file given the final object that it is bound to

        Parameters
        ----------
        data

        Returns
        -------

        """
        raise NotImplementedError("Writing Python data to NetCDF data has not been implemented!")

    @abc.abstractmethod
    def append(self, data):
        """
        Tell this writer how to write to the NetCDF file given the final object that it is bound to

        Parameters
        ----------
        data

        Returns
        -------

        """
        raise NotImplementedError("Writing Python data to NetCDF data has not been implemented!")

    def add_metadata(self, name, value):
        """
        Add metadata to self on disk, extra bits of information that can be used for flags or other variables
        This is NOT a staticmethod of the top dataset since you can buffer this before binding

        Parameters
        ----------
        name : string
            Name of the attribute you wish to assign
        value : any, but prefered string
            Extra meta data to add to the variable
        """
        if not self._bound_target:
            self._metadata_buffer[name] = value
        else:
            self._bound_target.setncattr(name,value)

    def _dump_metadata_buffer(self):
        """
        Dump the metadata buffer to file
        """
        if not self._bound_target:
            raise UnboundLocalError("Cannot dump the metadata buffer to target since no target exists!")
        self._bound_target.setncatts(self._metadata_buffer)
        self._metadata_buffer = {}

    @staticmethod
    def _convert_netcdf_store_type(stored_type):
        """
        Convert the stored NetCDF datatype from string to type without relying on unsafe eval() function

        Parameters
        ----------
        stored_type : string
            Read from ncfile.Variable.type

        Returns
        -------
        proper_type : type
            Python or module type

        """
        import importlib
        try:
            # Check if it's a builtin type
            try:  # Python 2
                module = importlib.import_module('__builtin__')
            except:  # Python 3
                module = importlib.import_module('builtins')
            proper_type = getattr(module, stored_type)
        except AttributeError:
            # if not, separate module and class
            module, stored_type = stored_type.rsplit(".", 1)
            module = importlib.import_module(module)
            proper_type = getattr(module, stored_type)
        return proper_type

# =============================================================================
# REAL TYPE HANDLERS
# =============================================================================

# Int
# List
# Array
# Quantity
# Units?
# OpenMM System


class NCString(NCVariableTypeHandler):
    """
    NetCDF handling strings
    I don't expect the unit to be affixed to a string, so there is no processing for it
    """
    @property
    def _dtype(self):
        return str

    @staticmethod
    def _dtype_type_string():
        return "str"

    def read(self):
        if self._bound_target is None:
            self._bind_read()

    def write(self, data):
        # Check type
        if type(data) is not self._dtype:
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        # Bind
        if self._bound_target is None:
            self._bind_write(data)
        # Check writeable
        if not self._can_append:
            raise TypeError("{} at {} was saved as appendable data! Cannot overwrite, must use append()".format(
                self.type_string, self._target)
            )
        # Save data
        packed_data = np.empty(1, 'O')
        packed_data[0] = data
        self._bound_target[:] = packed_data
        return

    def append(self, data):
        # Check type
        if type(data) is not self._dtype:
            raise TypeError("Invalid data type on variable {}.".format(self._target))
        # Bind
        if self._bound_target is None:
            self._bind_append(data)
        # Can append
        if not self._can_append:
            raise TypeError("{} at {} was saved as static, non-appendable data! Cannot append, must use write()".format(
                self.type_string, self._target)
            )
        # Determine current current length and therefore the last index
        length = self._bound_target.shape[0]
        # Save data
        packed_data = np.empty(1, 'O')
        packed_data[0] = data
        self._bound_target[length, :] = packed_data
        return

    def _bind_write(self, data):
        try:
            self._bind_read()
        except KeyError:
            NetCDFIODriver.check_scalar_dimension(self._parent_handler.ncfile)
            self._bound_target = self._storage_object.createVariable(self._target, str, dimensions='scalar', chunksize=(1,))
            # Specify a way for the IO to handle the
            self.add_metadata('IODriver_Type', self.type_string)
            # Specify the type of storage object this should tie to
            self.add_metadata('IODriver_Storage_Type', 'variable')
            self.add_metadata('IODriver_Appendable', 0)
        self._dump_metadata_buffer()
        if not self._bound_target.getncattr('IODriver_Appendable'):  # Returns integer form of bool
            # TODO: Make this an overwriteable warning
            raise TypeError("{} at {} was saved as appendable data! Cannot overwrite, must use append()".format(
                self.type_string, self._target)
            )
        else:
            self._can_write = True
        return

    def _bind_read(self):
        try:
            self._bound_target = self._storage_object[self._target]
        except KeyError as e:
            raise e

    def _bind_append(self, data, infinite_dimension='iteration'):
        try:
            self._bind_read()
        except KeyError:
            NetCDFIODriver.check_scalar_dimension(self._parent_handler.ncfile)
            NetCDFIODriver.check_infinite_dimension(self._parent_handler.ncfile)
            self._bound_target = self._storage_object.createVariable(
                self._target,
                str,
                dimensions=(infinite_dimension, 'scalar'),
                chunksize=(1, 1))
            # Specify a way for the IO to handle the
            self.add_metadata('IODriver_Type', self.type_string)
            # Specify the type of storage object this should tie to
            self.add_metadata('IODriver_Storage_Type', 'variable')
            self.add_metadata('IODriver_Appendable', 1)
        self._dump_metadata_buffer()
        if self._bound_target.getncattr('IODriver_Appendable'):  # Returns integer form of bool
            # TODO: Make this an overwriteable warning
            raise TypeError("{} at {} was saved as static, non-appendable data! Cannot append, must use write()".format(
                self.type_string, self._target)
            )
        else:
            self._can_append = True
        return


class NCDict(NCVariableTypeHandler):
    """
    NetCDF handling of dictionaries
    This is by in-large the most complicated object to store since its the combination of all types
    """
    @property
    def _dtype(self):
        return dict

    @staticmethod
    def _dtype_type_string():
        return "dict"

    def read(self):
        if self._bound_target is None:
            self._bind_read()
        return self._decode_dict()

    def write(self, data):
        if self._bound_target is None:
            self._bind_write(data)
        self._encode_dict(data)

    def append(self, data):
        self._bind_append(data)

    def _bind_write(self, data):
        # Because the _bound_target in this case is a NetCDF group, no initial data is writen.
        # The write() function handles that though
        try:
            self._bind_read()
        except KeyError:
            self._bound_target = self._storage_object.createGroup(self._target)
        # Specify a way for the IO to handle the
        self.add_metadata('IODriver_Type', self.type_string)
        # Specify the type of storage object this should tie to
        self.add_metadata('IODriver_Storage_Type', 'group')
        self.add_metadata('IODriver_Appendable', 0)
        self._dump_metadata_buffer()

    def _bind_read(self):
        try:
            self._bound_target = self._storage_object[self._target]
        except KeyError as e:
            raise e

    def _bind_append(self, data):
        # TODO: Determine how to do this eventually
        raise NotImplementedError("Dictionaries cannot be appended to!")

    def _decode_dict(self):
        """

        Returns
        -------
        output_dict : dict
            The restored dictionary as a dict.

        """
        output_dict = dict()
        for output_name in self._bound_target.variables.keys():
            # Get NetCDF variable.
            output_ncvar = self._bound_target.variables[output_name]
            type_name = getattr(output_ncvar, 'type')
            # TODO: Remove the if/elseif structure into one handy function
            # Get output value.
            if type_name == 'NoneType':
                output_value = None
            else:  # Handle all Types not None
                output_type = NCVariableTypeHandler._convert_netcdf_store_type(type_name)
                if output_ncvar.shape == ():
                    # Handle Standard Types
                    output_value = output_type(output_ncvar.getValue())
                elif output_ncvar.shape[0] >= 0:
                    # Handle array types
                    output_value = np.array(output_ncvar[:], output_type)
                    # TODO: Deal with values that are actually scalar constants.
                    # TODO: Cast to appropriate type
                else:
                    # Handle iterable types?
                    # TODO: Figure out what is actually cast here
                    output_value = output_type(output_ncvar[0])
            # If Quantity, assign unit.
            if hasattr(output_ncvar, 'units'):
                output_unit_name = getattr(output_ncvar, 'units')
                if output_unit_name[0] == '/':
                    output_value = str(output_value) + output_unit_name
                else:
                    output_value = str(output_value) + '*' + output_unit_name
                output_value = quantity_from_string(output_value)
            # Store output.
            output_dict[output_name] = output_value

        return output_dict

    def _encode_dict(self, data):
        """
        Store the contents of a dict in a NetCDF file.

        Parameters
        ----------
        data : dict
            The dict to store.

        """
        NetCDFIODriver.check_scalar_dimension(self._parent_handler.ncfile)
        for datum_name in data.keys():
            # Get entry value.
            datum_value = data[datum_name]
            # If Quantity, strip off units first.
            datum_unit = None
            if type(datum_value) == unit.Quantity:
                datum_unit = datum_value.unit
                datum_value = datum_value / datum_unit
            # Store the Python type.
            datum_type = type(datum_value)
            datum_type_name = typename(datum_type)
            # Handle booleans
            if type(datum_value) == bool:
                datum_value = int(datum_value)
            # Store the variable.
            if type(datum_value) == str:
                ncvar = self._bound_target.createVariable(datum_name, type(datum_value), 'scalar')
                packed_data = np.empty(1, 'O')
                packed_data[0] = datum_value
                ncvar[:] = packed_data
                ncvar.setncattr('type', datum_type_name)
            elif isinstance(datum_value, collections.Iterable):
                nelements = len(datum_value)
                element_type = type(datum_value[0])
                element_type_name = typename(element_type)
                self._bound_target.createDimension(datum_name, nelements) # unlimited number of iterations
                ncvar = self._bound_target.createVariable(datum_name, element_type, (datum_name,))
                for (i, element) in enumerate(datum_value):
                    ncvar[i] = element
                ncvar.setncattr('type', element_type_name)
            elif datum_value is None:
                ncvar = self._bound_target.createVariable(datum_name, int)
                ncvar.assignValue(0)
                ncvar.setncattr('type', datum_type_name)
            else:
                ncvar = self._bound_target.createVariable(datum_name, type(datum_value))
                ncvar.assignValue(datum_value)
                ncvar.setncattr('type', datum_type_name)
            if datum_unit:
                ncvar.setncattr('units', str(datum_unit))
        return


# =============================================================================
# LOGIC TIME!
# =============================================================================

"""
Storage Interface (SI) passes down a name of file and instantiates a Storage Handler (SH) instance
SI Directory/Variable (SIDV) objects send and request data down
SH Binds to a file (if present) and waits for next instructions, all logic follows instructions from SI => SH

From the SIDV:
SIDV.{write/append/fetch}
    If not VARIABLE:
        SH.bind_variable(PATH, MODE?)
    VARIABLE.{write/fetch/append}


SI.write_metadata(NAME, DATA):
    Add NAME as a metadata entry with DATA attached:
        *def write_metadata(NAME, DATA)*
SIDV.{write/append/fetch} initiate a BIND event, cascading up
    PATH will be generated from the SIDV and passed down
    Convert PATH to groups and variable
    Bind .{write/append}
        Check if PATH in FILE
            Yes?
                Fetch STORAGEOBJECT (SO), TYPE
                Bind (D)E(N)CODER (denC) based on TYPE
                Bind UNIT to denC
                Bind SO to denC
                Return denC to SIDV
            No?



SIDV.{fetch}
    PATH passed down by the SIDV
    If PATH not in BOUNDSET:
        If not on_file(PATH):
            Raise Error
        else:
            Fetch STORAGEOBJECT, TYPE, UNIT from FILE at PATH
            BOUNDSET[PATH] = denC(STORAGEOBJECT, TYPE, UNIT)
    denC = BOUNDSET[PATH]
    denC.



BOUNDSET is a dict of objects which point to different denC's with keywords as fed PATHs from SIDV
Needed Functions of SH:
    write_metadata(NAME, DATA)
        Returns None
    on_file(PATH)
        Returns Bool
    read_file(PATH)     Might combine with on_file
        Returns STORAGEOBJECT, TYPE, UNIT
Needed Functions of denC:
    unit
        Property, returns units of the bound STORAGEOBJECT









Write Bound Present
Write Unbound Present
Append Bound Present
append Unbound Present
Fetch Bound Present
Fetch Unbound Present
Write Bound NotPresent
Write Unbound NotPresent
Append Bound NotPresent
append Unbound NotPresent

{Fetch Bound NotPresent
Fetch Unbound NotPresent}
    Raise Error
"""























