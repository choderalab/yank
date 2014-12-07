#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Print available OpenMM platforms.

"""

#=============================================================================================
# MODULE IMPORTS
#=============================================================================================

#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    from simtk import openmm
    print "Available OpenMM platforms:"
    for platform_index in range(openmm.Platform.getNumPlatforms()):
        print "%5d %s" % (platform_index, openmm.Platform.getPlatform(platform_index).getName())
    print ""

    return True

