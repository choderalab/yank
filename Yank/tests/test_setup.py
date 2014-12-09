#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test 'yank setup'.

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
    Test 'yank setup binding amber' for implicit solvent system.
    """
    store_directory = tempfile.mkdtemp()
    examples_path = utils.get_data_filename("examples/")
    command = 'yank setup binding amber --setupdir=%(examples_path)/benzene-toluene-implicit/setup --ligname=BEN --store=%(store_directory)s --iterations=1 --restraints=harmonic --gbsa=OBC2 --temperature=300*kelvin' % vars()
    if verbose: command += ' --verbose'
    argv = command.split()
    args = docopt(usage, version=version.version, argv=argv[1:])
    from yank.commands import setup
    setup.dispatch(args)

def test_setup_amber_explicit(verbose=False):
    """
    Test 'yank setup binding amber' for explicit solvent system.
    """
    store_directory = tempfile.mkdtemp()
    examples_path = utils.get_data_filename("examples/")
    command = 'yank setup binding amber --setupdir=%(examples_path)/examples/benzene-toluene-explicit/setup --ligname=BEN --store=%(store_directory)s --iterations=1 --nbmethod=CutoffPeriodic --temperature=300*kelvin --pressure=1*atmospheres' % vars()
    if verbose: command += ' --verbose'
    argv = command.split()
    args = docopt(usage, version=version.version, argv=argv[1:])
    from yank.commands import setup
    setup.dispatch(args)

#=============================================================================================
# MAIN
#=============================================================================================

if __name__ == '__main__':
    test_setup_amber_implicit(verbose=True)
    test_setup_amber_explicit(verbose=True)
