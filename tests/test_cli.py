#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test command-line interface.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import tempfile

from openmmtools import testsystems

from yank import cli

from nose.plugins.skip import Skip, SkipTest

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def run_cli(command):
    cli.main(argv=command.split())

def test_help():
    run_cli('--help')

def test_selftest():
    run_cli('selftest')

def test_setup_binding():
    dirname = testsystems.get_data_filename("data/T4-lysozyme-L99A-implicit")
    storedir = tempfile.mkdtemp()
    run_cli('setup binding amber --setupdir=%(dirname)s --ligname MOL --store %(storedir)s' % vars())
    # TODO: Clean up directory.
