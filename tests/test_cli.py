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
import commands

from openmmtools import testsystems

from yank import cli

from nose.plugins.skip import Skip, SkipTest

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def run_cli(arguments, expected_output=None):
    #cli.main(argv=arguments.split())
    [status, output] = commands.getstatusoutput('yank ' + arguments)

    if status:
        message  = "An error return value (%s) was obtained:\n" % str(status)
        message += "\n"
        message += output
        message += "\n"
        raise Exception(message)

    if expected_output:
        if output != expected_output:
            message  = "Output differs from expected output.\n"
            message += "\n"
            message += "Expected output:\n"
            message += expected_output
            message += "\n"
            message += "Actual output:\n"
            message += output
            message += "\n"
            raise Exception(message)

def test_help():
    run_cli('--help')

def test_selftest():
    run_cli('selftest')

def test_setup_binding():
    dirname = testsystems.get_data_filename("data/T4-lysozyme-L99A-implicit")
    storedir = tempfile.mkdtemp()
    run_cli('setup binding amber --setupdir=%(dirname)s --ligname MOL --store %(storedir)s' % vars())
    # TODO: Clean up directory.
