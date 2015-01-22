#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test 'yank prepare'.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import tempfile

from openmmtools import testsystems

from nose.plugins.skip import Skip, SkipTest

from docopt import docopt
from yank.cli import usage
from yank import version, utils

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def test_setup_amber_implicit(verbose=False):
    """
    Test 'yank prepare binding amber' for implicit solvent system.
    """
    store_directory = tempfile.mkdtemp()
    examples_path = utils.get_data_filename("../examples/benzene-toluene-implicit/setup/")  # Could only figure out how to install things like yank.egg/examples/, rather than yank.egg/yank/examples/
    command = 'yank prepare binding amber --setupdir=%(examples_path)s --ligname=BEN --store=%(store_directory)s --iterations=1 --restraints=harmonic --gbsa=OBC2 --temperature=300*kelvin' % vars()
    if verbose: command += ' --verbose'
    argv = command.split()
    args = docopt(usage, version=version.version, argv=argv[1:])
    from yank.commands import prepare
    setup.dispatch(args)

def test_setup_amber_explicit(verbose=False):
    """
    Test 'yank prepare binding amber' for explicit solvent system.
    """
    store_directory = tempfile.mkdtemp()
    examples_path = utils.get_data_filename("../examples/benzene-toluene-explicit/setup/")  # Could only figure out how to install things like yank.egg/examples/, rather than yank.egg/yank/examples/
    command = 'yank prepare binding amber --setupdir=%(examples_path)s --ligname=BEN --store=%(store_directory)s --iterations=1 --nbmethod=CutoffPeriodic --temperature=300*kelvin --pressure=1*atmospheres' % vars()
    if verbose: command += ' --verbose'
    argv = command.split()
    args = docopt(usage, version=version.version, argv=argv[1:])
    from yank.commands import prepare
    setup.dispatch(args)

#=============================================================================================
# MAIN
#=============================================================================================

if __name__ == '__main__':
    test_setup_amber_implicit(verbose=True)
    test_setup_amber_explicit(verbose=True)
