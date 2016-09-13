#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Clean up files produced by a YANK calculation.

"""

#=============================================================================================
# MODULE IMPORTS
#=============================================================================================

import os, os.path
import glob
import sys

#=============================================================================================
# COMMAND-LINE INTERFACE
#=============================================================================================

usage = """
YANK cleanup

Usage:
  yank cleanup (-s=STORE | --store=STORE) [-v | --verbose]

Description:
  Clean up (delete) the run files.

Required Arguments:
  -s=STORE, --store=STORE       Storage directory for NetCDF data files.

General Options:
  -v, --verbose                 Print verbose output

"""

#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    verbose = args['--verbose']

    # Remove NetCDF files in the destination directory.
    for filename in glob.glob(os.path.join(args['--store'], '*.nc')):
        if verbose: print "Removing file %s" % filename
        os.remove(filename)

    return True
