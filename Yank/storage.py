#!/usr/bin/env/python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Classes that store arbitrary NetCDF (or other) options and describe how to handle them

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
from .storage_handlers import *

ABC = abc.ABCMeta('ABC', (object,), {})  # compatible with Python 2 *and* 3

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

# =============================================================================
# GENERIC SAVABLE DATA
# =============================================================================


class StorageInterfaceDirVar(object):
    """
    Storage Interface Directory/Variable (SIDV) class is a versatile, dynamic class which gives structure to the
    variables stored on disk by representing them as methods of the StorageInterface and other instances of itself.
    New variables and folders are created by simply trying to use them as properties of the class itself.
    The data stored and read is not kept in memory, the SIDV passes the data onto the writer or tells the fetcher to
    read from disk on demand.

    The API only shows the protected and internal methods of the SIDV class which cannot be used for variable and
    directory names.

    This class is never meant to be used as its own object and should never be invoked by itself. The class is currently
    housed outside of the main StorageInterface class to show the API to users.

    TODO: Move this class as an internal class of StorageInterface (NOT a subclass)

    == Examples ==

    Create a basic storage system and write new DATA to disk
    >>> my_storage = StorageInterface('my_store.nc')
    >>> my_storage.my_variable.write(DATA)

    Create a folder called "vardir" with two variables in it, "var1" and "var2" holding DATA1 and DATA2 respectively
    >>> my_storage = StorageInterface('my_store.nc')
    >>> my_storage.vardir.var1.write(DATA1)
    >>> my_storage.vardir.var2.write(DATA2)

    Fetch data from the same folders and variables as above
    >>> my_storage = StorageInterface('my_store.nc')
    >>> var1 = my_storage.vardir.var1.fetch()
    >>> var2 = my_storage.vardir.var2.fetch()

    Run some_function() to generate data over an iteration, store each value in a dynamically sized var called "looper"
    >>> my_storage = StorageInterface('my_store.nc')
    >>> for i in range(10):
    >>>     mydata = some_function()
    >>>     my_storage.looper.append(mydata)

    """

    def __init__(self, file_name, storage_system='netcdf'):
        """
        Parameters
        ----------
        file_name : string
            Specify what file name to read and write to on the disk.
        storage_system : string
            Specify what type of storage system module to load for interfacing with the file_name
        """

    """
    USER FUNCTIONS
    Methods the user calls on SIDV variables to actually store and read data from disk. These cannot be names of child
    SIDV instances and will raise an error if you try to use them as such.
    """

    def write(self, data, protected_write=True):
        """
        Write data to a variable which cannot be appended to, nor overwritten (unless specified). This method should
        be called when you do not want the value stored to change, typically written only once. The protected_write
        variable will let you overwrite this behavior if you want to use something on disk that is a toggleable flag,
        but does not change in shape.

        Metadata is added to this variable if possible to indicate this is a write protectable variable.

        This method raises an error if this instance of SIDV is DIRECTORY

        Parameters
        ----------
        data
            This is what will be written to disk, the data will be processed by the STORAGESYSTEM as to how to actually
            handle the data once the units have been stripped out and assigned to this instance of SIDV.
        protected_write : boolean, default True
            Decide whether to check if the variable is already on file if present.
            If True, no overwrite is allowed and an error is raised if the variable already exists.
            If False, overwriting is allowed but the variable will still not be able to be appended to.

        """

    def append(self, data):
        """
        Write data to a variable who's size changes every time this function is called. The first dimension of the
        variable on disk will be appended with data after checking the size. If no variable exists, one is created with
        a dynamic variable size for its first dimension, then the other dimensions are inferred from the data.

        This method raises an error if this instance of SIDV is DIRECTORY

        Parameters
        ----------
        data
            This is what will be written to disk, the data will be processed by the STORAGESYSTEM as to how to actually
            handle the data once the units have been stripped out and assigned to this instance of SIDV.

        """

    def fetch(self):
        """
        Read the variable and its data from disk.

        If this instance is unbound to an object in the storage_handler or is DIRECTORY, then this method raises an
        error.

        Returns
        -------
        data
            Data stored on VARIABLE read from disk and processed through the STORAGESYSTEM back into a Python type
            and possibly through the UNIT logic to recast into Quantity before being handed to the user.
        """

    def add_metadata(self, name, data):
        """
        Attempt to add metadata to the variable/directory. Usually a string input to include additional information
        about the variable.

        Because this can act on DIRECTORY or VARIABLE types, all metadata is buffered until this instance of SIDV is
        bound to an object on the file itself.

        This does not check if the metadata already exists and simply overwrites it if present.


        Parameters
        ----------
        name : string
            Name of the metadata variable. Since this is data we are directly attaching to the variable or directory
            itself, we do not make this another SIDV instance.
        data: What data you wish to attach to the `name`d metadata pointer.

        """

    """
    PROTECTED FLAGS
    Used for functionality checks that will likely never be called by the user, but exposed to show
    what names can NOT be used in the dynamic directory/variable namespace a user will generate.

    All capital names to distinguish them as protected types and avoid clashes with built-in functions.
    """
    @property
    def VARIABLE(self):
        """
        Checks if the object can be used in the .write, .append, .fetch functions can be used. Once this is set, this
        instance cannot be converted to a directory type.

        Returns
        -------
        variable_pointer : None or storage_system specific unit of storage
            Returns None if this instance is a directory, or if its functionality has not been determined yet (this
                includes the variable not being assigned yet)
            Returns the storage_system specific variable that this instance is bound to once assigned

        """

    @property
    def DIRECTORY(self):
        """
        Checks if the object can be used as a directory for accepting other SIDV objects. Once this is set, the
        .write, .append, and .fetch functions are locked out.

        Returns
        -------
        directory_pointer : None or storage_system specific directory of storage
            Returns None if this instance is a variable, or if its functionality has not been determined yet (this
                includes the directory not being assigned yet)
            Returns the storage_system specific directory that this instance is bound to once assigned

        """

    @property
    def PATH(self):
        """
        Generate the complete path of this instance by getting its predecessor's path + itself. This is a cascading
        function up the stack of SIDV's until the top level attached to the main SI instance is reached, then
        reassembled on the way down.

        Returns
        -------
        full_path : string
            The complete path of this variable as it is seen on the storage file_name, returned as / separated values.

        """

    @property
    def PREDECESSOR(self):
        """
        Give the parent SIDV to construct the full path on the way down

        Returns
        -------
        predecessor : None or StorageInterfaceDirVar instance
            Returns this instance's parent SIDV instance or None if it is the top level SIDV.
        """

    @property
    def TYPE(self):
        """
        Return the underlying Python type of this variable. This serves as metadata for reading/writing to disk, and
        as a check if new data will be compatible with existing data for `append`

        Returns
        -------
        my_type : type
            Python type of the variable that this instance is bound to. Returns NoneType if variable is actually a
            None or if it is a Directory
        """

    @property
    def UNIT(self):
        """
        Returns the optional units attached to VARIABLE as metadata if present. Any data read into the VARIABLE must
        have the same UNIT. e.g. Kelvin data cannot be stored to Kilojoues data. Limitation e.g. Data in angstroms cannot
        be stored with data in nanometers, they must all be of the same UNIT.

        Returns
        -------
        attached_unit : None or simtk.unit.Unit
            None if there are no units, or the appropriate Unit from simtk.unit module
        """

    @property
    def STORAGESYSTEM(self):
        """
        Pointer to the object which actually handles read/write operations
        Returns
        -------
        storage_system :
            Instance of the module which handles IO actions to specific storage type requested by storage_system
            string at initialization.

        """

    @property
    def METADATABUFFER(self):
        """
        Buffer variable that stores metadata until this instance is bound to either a directory or variable object.

        Returns
        -------
        Dictionary of metadata that will be written to directory/variable once bound. Variable is deleted but still
        protected once bound.

        """

# =============================================================================
# STORAGE INTERFACE
# =============================================================================


class StorageInterface(object):
    def __init__(self, file_name, storage_system='netcdf'):
        """
        Initialize the class, set the file save location and what type of output is expected.
        The class itself simply holds what folders and variables are known to the file on the disk, or dynamically
        creates them on the fly. See StorageInterfaceVarDir for how this works.

        Parameters
        ----------
        file_name : string
            Full path to the location of the save file on the operating system. A new file will be generated if one
            cannot be found, or this class will load and read the file it does find in that location.
        storage_system : string, default "netcdf"
            Tell the system what type of storage to use. Currently only works for for NetCDF but could be extended to
            use more.

        """

    def write_metadata(self, name, data):
        """
        Write additional meta data to attach to the storage_system file itself.
        Parameters
        ----------
        name
        data

        Returns
        -------

        """

    @property
    def file_name(self):
        """
        Returns the protected _file_name variable, mentioned as an explicit method as it is one of the protected names
        that cannot be a directory or variable pointer

        Returns
        -------
        file_name : string
            Name of the file on the disk

        """

    @property
    def storage_system_name(self):
        """
        Returns the protected _storage_system_name variable, mentioned as an explicit method as it is one of the protected
        names that cannot be a directory or variable pointer

        Returns
        -------
        storage_system_name :
            Name of the storage system which is reading/writing to disk

        """

    @property
    def STORAGESYSTEM(self):
        """
        Pointer to the object which actually handles read/write operations
        Returns
        -------
        storage_system :
            Instance of the module which handles IO actions to specific storage type requested by storage_system
            string at initialization.

        """

# =============================================================================
# LOGIC TIME!
# =============================================================================
"""
Storage Interface (SI) and Storage Interface Directory Variable (SIDV) have their own methods, albeit very few. These
are protected and cannot be overwritten.

SI.var
    instantiate .var as storage object with unknown TYPE, unknown UNIT, basic PATH as self, unlocked WRITE/APPEND/FETCH/DIRECTORIZE, no PREDECESSOR
SI.var.{write/append/get}
    Is .var DIRECTORIZED?
        If yes:
            Raise Error
    One time initialization actions:
        Fetch full PATH from predecessors
        Does var exists on storage medium?
            Yes?
                Assign TYPE to .var
                Assign UNIT to .var
                Set EXISTS to True
            No?
                Set EXISTS to False
        lock ability DIRECTORIZE
    .{write/append}(DATA):
        .write unique action:
            If EXISTS and overwrite_protection = True:
                Raise protection warning, fail
        Strip UNIT
        Read DATA type
        If not TYPE:
            Assign TYPE = type(DATA)
        else:
            Assert TYPE == type(DATA)
        .write action:
            Store DATA to disk, overwriting if there
        .append action:
            If hasattr PROTECTED:
                Fail as this is not dynamic data
            If EXISTS:
                Append DATA to first dim
            Else:
                Write DATA to disk,
                Set EXISTS to True
    .fetch():
        If not EXISTS:
            Fail
        Else:
            Read Disk
SI.folder.var
    Effectively: __getattr__(name) where name is not a known method
    If already DIRECTORY:
        Carry on
    Elif can DIRECTORIZE:
        lock out ability to WRITE/APPEND/FETCH/DIRECTORIZE
        Assign is DIRECTORY
    else:
        Raise cannot because already VARIABLE
    Instantiate .var
        Assign .var PREDECESSOR = self


Notes:
    UNIT can be None (no unit)
    overwrite_protection is a kwarg of .write which defaults to True
"""
































