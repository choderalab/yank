#!/usr/local/bin/env python

"""
Test storage.py facility.

The tests are written around the netcdf storage handler for its asserts (its default)
Testing the storage handlers themselves should be left to the test_storage_handlers.py file

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import numpy as np
import openmoltools as omt

from nose import tools

from yank.storage import *

# =============================================================================================
# NETCDFIODRIVER TESTING FUNCTIONS
# =============================================================================================


def test_storage_interface_creation():
    """Test that the storage interface can create a top level file and read from it"""
    with omt.utils.temporary_directory() as tmp_dir:
        si = StorageInterface(tmp_dir + '/teststore.nc')
        si.add_metadata('name', 'data')
        assert si._storage_system.ncfile.getncattr('name') == 'data'


@tools.raises(Exception)
def test_read_trap():
    """Test that attempting to read a non-existent file fails"""
    with omt.utils.temporary_directory() as tmp_dir:
        si = StorageInterface(tmp_dir + '/teststore.nc')
        si._instance_read()


def test_variable_write_read():
    """Test that a variable can be create and written to file"""
    with omt.utils.temporary_directory() as tmp_dir:
        si = StorageInterface(tmp_dir + '/teststore.nc')
        input_data = 4
        si.four.write(input_data)
        output_data = si.four.read()
        assert output_data == input_data


def test_variable_append_read():
    """Test that a variable can be create and written to file"""
    with omt.utils.temporary_directory() as tmp_dir:
        si = StorageInterface(tmp_dir + '/teststore.nc')
        input_data = np.eye(3) * 4.0
        si.four.append(input_data)
        si.four.append(input_data)
        output_data = si.four.read()
        assert np.all(output_data[0] == input_data)
        assert np.all(output_data[1] == input_data)


def test_unbound_read():
    """Test that a variable can read from the file without previous binding"""
    with omt.utils.temporary_directory() as tmp_dir:
        si = StorageInterface(tmp_dir + '/teststore.nc')
        input_data = 4*unit.kelvin
        si.four.write(input_data)
        si.storage_system.close_down()
        del si
        si = StorageInterface(tmp_dir + '/teststore.nc')
        output_data = si.four.read()
        assert input_data == output_data


def test_directory_creation():
    """Test that automatic directory-like objects are created on the fly"""
    with omt.utils.temporary_directory() as tmp_dir:
        si = StorageInterface(tmp_dir + '/teststore.nc')
        input_data = 'four'
        si.dir0.dir1.dir2.var.write(input_data)
        ncfile = si.storage_system.ncfile
        target = ncfile
        for i in range(3):
            my_dir = 'dir{}'.format(i)
            assert my_dir in target.groups
            target = target.groups[my_dir]
        si.storage_system.close_down()
        del si
        si = StorageInterface(tmp_dir + '/teststore.nc')
        target = si
        for i in range(3):
            my_dir = 'dir{}'.format(i)
            target = getattr(target, my_dir)
        assert target.var.read() == input_data


def test_multi_variable_creation():
    """Test that multiple variables can be created in a single directory structure"""
    with omt.utils.temporary_directory() as tmp_dir:
        si = StorageInterface(tmp_dir + '/teststore.nc')
        input_data = [4.0, 4.0, 4.0]
        si.dir0.var0.write(input_data)
        si.dir0.var1.append(input_data)
        si.dir0.var1.append(input_data)
        si.storage_system.close_down()
        del si
        si = StorageInterface(tmp_dir + '/teststore.nc')
        assert si.dir0.var0.read() == input_data
        app_data = si.dir0.var1.read()
        assert app_data[0] == input_data
        assert app_data[1] == input_data
