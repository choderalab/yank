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
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    # Remove NetCDF files in the destination directory.
    for filename in glob.glob(os.path.join(args['--store'], '*.nc')):
        print "Removing file %s" % filename
        os.remove(filename)

    return True
