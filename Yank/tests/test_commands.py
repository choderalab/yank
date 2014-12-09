#!/usr/bin/python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Test command-line tools.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import tempfile

from openmmtools import testsystems

from nose.plugins.skip import Skip, SkipTest

#=============================================================================================
# UNIT TESTS
#=============================================================================================

def test_cite():
    from yank.commands import cite
    args = []
    cite.dispatch(args)

def test_help():
    from yank.commands import help
    args = []
    help.dispatch(args)
