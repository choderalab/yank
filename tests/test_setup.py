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
from yank import version

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def test_setup_amber():
    store_directory = tempfile.mkdtemp()
    command = 'yank setup binding amber --setupdir=examples/benzene-toluene/setup --ligname=BEN --store=%(store_directory)s --iterations=1 --restraints=harmonic --gbsa=OBC2 --temperature=300*kelvin --verbose' % vars()
    argv = command.split()
    args = docopt(usage, version=version.version, argv=argv[1:])
    from yank.commands import setup
    setup.dispatch(args)

