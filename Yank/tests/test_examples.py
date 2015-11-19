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

import os

from yank import utils
from nose.plugins.attrib import attr

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def run_example(path, example):
    example_dir = os.path.join(path, example)

    # Skip folders that do not contain working examples (e.g. yank-yaml-cookbook)
    if not os.path.exists(os.path.join(example_dir, 'run.sh')):
        return

    # Change to example directory.
    with utils.temporary_cd(example_dir):
        # Execute one iteration of the example.
        import subprocess
        returncode = subprocess.call('NITERATIONS=1 ./run.sh', shell=True, executable='/bin/bash')

    if returncode:
        raise Exception('Example %s returned exit code %d' % (example, returncode))

    return

def get_immediate_subdirectories(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and os.path.exists(os.path.join(path, name, 'run.sh'))]

@attr('slow') # Skip on Travis-CI
def test_examples():
    # Get example path.
    from pkg_resources import resource_filename
    path = resource_filename('yank', '../examples')

    # Get list of examples.
    directories = get_immediate_subdirectories(path)

    # Test examples
    for directory in directories:
        run_example(path, directory)

