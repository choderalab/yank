#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test examples.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os, os.path
import subprocess

from openmmtools import testsystems

from nose.plugins.skip import Skip, SkipTest

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def run_example(path, example):
    # Change to example directory.
    cwd = os.getcwd()
    os.chdir(os.path.join(path, example))

    # Execute one iteration of the example.
    import subprocess
    returncode = subprocess.call('NITERATIONS=5 ./run.sh', shell=True, executable='/bin/bash')

    # Restore working directory.
    os.chdir(cwd)

    if returncode:
        raise Exception('Example %s returned exit code %d' % (example, returncode))

    return

def get_immediate_subdirectories(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and os.path.exists(os.path.join(path, name, 'run.sh'))]

def test_examples():
    # Get example path.
    from pkg_resources import resource_filename
    path = resource_filename('yank', '../examples')

    # Get list of examples.
    directories = get_immediate_subdirectories(path)

    # Test examples
    for directory in directories:
        run_example(path, directory)

