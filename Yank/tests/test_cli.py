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

from yank import cli, utils

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

def test_prepare_binding():
    # NOTE: switched to yank p-xylene from openmmtools T4-lysozyme because of yank bugs.
    dirname = utils.get_data_filename("../examples/p-xylene-implicit/setup/")  # Could only figure out how to install things like yank.egg/examples/, rather than yank.egg/yank/examples/
    storedir = tempfile.mkdtemp()
    run_cli('prepare binding amber --setupdir=%(dirname)s --ligand="resname MOL" --store %(storedir)s --gbsa OBC1' % vars())
    # TODO: Clean up directory.
