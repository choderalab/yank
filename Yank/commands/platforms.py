#!/usr/local/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Print available OpenMM platforms.

"""

# =============================================================================================
# MODULE IMPORTS
# =============================================================================================

# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================

usage = """
YANK platforms

Usage:
  yank platforms

Description:
  List available OpenMM platforms

"""

# =============================================================================================
# COMMAND DISPATCH
# =============================================================================================


def dispatch(args):
    from simtk import openmm
    print("Available OpenMM platforms:")
    for platform_index in range(openmm.Platform.getNumPlatforms()):
        print("{0:5d} {1:s}".format(platform_index, openmm.Platform.getPlatform(platform_index).getName()))
    print("")

    return True

