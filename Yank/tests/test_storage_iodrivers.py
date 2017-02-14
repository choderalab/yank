#!/usr/local/bin/env python

"""
Test iodrivers.py facility.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import numpy as np
from simtk import unit
import openmoltools as omt

from nose import tools

from yank.storage import NetCDFIODriver

# =============================================================================================
# NETCDFIODRIVER TESTING FUNCTIONS
# =============================================================================================


def test_netcdf_driver_group_manipulation():
    """Test that the NetCDFIODriver can create groups, rebind to groups, and that they are on the file"""
    with omt.utils.temporary_directory() as tmp_dir:
        nc_io_driver = NetCDFIODriver(tmp_dir + 'test.nc')
        group2 = nc_io_driver.get_directory('group1/group2')
        group1 = nc_io_driver.get_directory('group1')
        ncfile = nc_io_driver.ncfile
        ncgroup1 = ncfile.groups['group1']
        ncgroup2 = ncfile.groups['group1'].groups['group2']
        assert group1 is ncgroup1
        assert group2 is ncgroup2


def test_netcdf_driver_dimension_manipulation():
    """Test that the NetCDFIODriver can check and create dimensions"""
    with omt.utils.temporary_directory() as tmp_dir:
        nc_io_driver = NetCDFIODriver(tmp_dir + '/test.nc')
        NetCDFIODriver.check_scalar_dimension(nc_io_driver)
        NetCDFIODriver.check_iterable_dimension(nc_io_driver, length=4)
        NetCDFIODriver.check_infinite_dimension(nc_io_driver)
        ncfile = nc_io_driver.ncfile
        dims = ncfile.dimensions
        assert 'scalar' in dims
        assert 'iterable4' in dims
        assert 'iteration' in dims


def test_netcdf_driver_metadata_creation():
    """Test that the NetCDFIODriver can create metadata on different objects"""
    with omt.utils.temporary_directory() as tmp_dir:
        nc_io_driver = NetCDFIODriver(tmp_dir + '/test.nc')
        group1 = nc_io_driver.get_directory('group1')
        nc_io_driver.add_metadata('root_metadata', 'IAm(G)Root!')
        nc_io_driver.add_metadata('group_metadata', 'group1_metadata', path='/group1')
        ncfile = nc_io_driver.ncfile
        nc_metadata = ncfile.getncattr('root_metadata')
        group_metadata = group1.getncattr('group_metadata')
        assert nc_metadata == 'IAm(G)Root!'
        assert group_metadata == 'group1_metadata'


# =============================================================================================
# NETCDF TYPE HANDLER TESTING FUNCTIONS
# =============================================================================================


def generic_type_handler_check(input_data, with_append=True):
    """Generic type handler test to ensure all callable functions are working"""
    with omt.utils.temporary_directory() as tmp_dir:
        file_path = tmp_dir + '/test.nc'
        nc_io_driver = NetCDFIODriver(file_path)
        input_type = type(input_data)
        # Create a write and an append of the data
        write_path = 'data_write'
        data_write = nc_io_driver.create_storage_variable(write_path, input_type)
        if with_append:
            append_path = 'group1/data_append'
            data_append = nc_io_driver.create_storage_variable(append_path, input_type)
        # Store initial data (unbound write/append)
        data_write.write(input_data)
        if with_append:
            data_append.append(input_data)
        # Test that we can act on them again (bound write/append)
        data_write.write(input_data)
        if with_append:
            data_append.append(input_data)
        # Test bound read
        data_write_out = data_write.read()
        if with_append:
            data_append_out = data_append.read()
        try:  # Compound dictionary processing
            for key in data_write_out.keys():
                assert np.all(data_write_out[key] == input_data[key])
        except AttributeError:
            assert np.all(data_write_out == input_data)
        if with_append:
            try:
                for key in data_write_out.keys():
                    assert np.all(data_append_out[0][key] == input_data[key])
                    assert np.all(data_append_out[1][key] == input_data[key])
            except AttributeError:
                assert np.all(data_append_out[0] == input_data)
                assert np.all(data_append_out[1] == input_data)
        # Delete the IO driver (and close the ncfile in the process)
        nc_io_driver.close_down()
        del data_write, data_write_out
        if with_append:
            del data_append, data_append_out
        # Reopen and test reading actions
        nc_io_driver = NetCDFIODriver(file_path, access_mode='r')
        data_write = nc_io_driver.get_variable_handler(write_path)
        if with_append:
            data_append = nc_io_driver.get_variable_handler(append_path)
        # Test unbound read
        data_write_out = data_write.read()
        if with_append:
            data_append_out = data_append.read()
        try:  # Compound dictionary processing
            for key in data_write_out.keys():
                assert np.all(data_write_out[key] == input_data[key])
        except AttributeError:
            assert np.all(data_write_out == input_data)
        if with_append:
            try:
                for key in data_write_out.keys():
                    assert np.all(data_append_out[0][key] == input_data[key])
                    assert np.all(data_append_out[1][key] == input_data[key])
            except AttributeError:
                assert np.all(data_append_out[0] == input_data)
                assert np.all(data_append_out[1] == input_data)


def test_netcdf_int_type_handler():
    """Test that the Int type handler can read/write/append"""
    input_data = 4
    generic_type_handler_check(input_data)


def test_netcdf_float_type_handler():
    """Test that the Float type handler can read/write/append"""
    input_data = 4.0
    generic_type_handler_check(input_data)


def test_netcdf_string_type_handler():
    """Test that the String type handler can read/write/append"""
    input_data = 'four point oh'
    generic_type_handler_check(input_data)


def test_netcdf_list_type_handler():
    """Test that the List type handler can read/write/append"""
    input_data = [4, 4, 4]
    generic_type_handler_check(input_data)


def test_netcdf_tuple_type_handler():
    """Test that the tuple type handler can read/write/append"""
    input_data = (4, 4, 4)
    generic_type_handler_check(input_data)


def test_netcdf_array_type_handler():
    """Test that the ndarray type handler can read/write/append"""
    input_data = np.array([4, 4.0, 4])
    generic_type_handler_check(input_data)


def test_netcdf_quantity_type_handler():
    """Test that the simtk.unit.Quantity type handler can read/write/append with various unit and _value types"""
    input_data = 4 * unit.kelvin
    generic_type_handler_check(input_data)
    input_data = [4, 4, 4] * unit.kilojoules_per_mole
    generic_type_handler_check(input_data)
    input_data = np.array([4, 4, 4]) / unit.nanosecond
    generic_type_handler_check(input_data)


def test_netcdf_dictionary_type_handler():
    """Test that the simtk.unit.Quantity type handler can read/write/append with various unit and _value types"""
    input_data = {
        'count': 4,
        'ratio': 0.4,
        'name': 'four',
        'repeated': [4,4,4],
        'temperature': 4 * unit.kelvin,
        'box_vectors': (np.eye(3) * 4.0) * unit.nanometer
    }
    generic_type_handler_check(input_data, with_append=False)
