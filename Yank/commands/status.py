#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Query output files for quick status.

"""

#=============================================================================================
# MODULE IMPORTS
#=============================================================================================

from .. import utils

#=============================================================================================
# COMMAND-LINE INTERFACE
#=============================================================================================

usage = """
YANK status

Usage:
  yank status (-s=STORE | --store=STORE) [-v | --verbose]

Description:
  Get the current status of a running or completed simulation

Required Arguments:
  -s=STORE, --store=STORE       Storage directory for NetCDF data files.

General Options:
  -v, --verbose                 Print verbose output

"""

#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    from .. import analyze
    utils.config_root_logger(args['--verbose'])
    success = analyze.print_status(args['--store'])
    return success
