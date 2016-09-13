#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Analyze YANK output file.

"""

#=============================================================================================
# MODULE IMPORTS
#=============================================================================================

from yank import utils, analyze

#=============================================================================================
# COMMAND-LINE INTERFACE
#=============================================================================================

usage = """
YANK analyze

Usage:
  yank analyze (-s STORE | --store=STORE) [-v | --verbose]
  yank analyze extract-trajectory --netcdf=FILEPATH (--state=STATE | --replica=REPLICA) --trajectory=FILEPATH [--start=START_FRAME] [--skip=SKIP_FRAME] [--end=END_FRAME] [--nosolvent] [--discardequil] [-v | --verbose]

Description:
  Analyze the data to compute Free Energies OR extract the trajectory from the NetCDF file into a common fortmat.

Free Energy Required Arguments:
  -s=STORE, --store=STORE       Storage directory for NetCDF data files.

Extract Trajectory Required Arguments:
  --netcdf=FILEPATH             Path to the NetCDF file.
  --state=STATE_IDX             Index of the alchemical state for which to extract the trajectory
  --replica=REPLICA_IDX         Index of the replica for which to extract the trajectory
  --trajectory=FILEPATH         Path to the trajectory file to create (extension determines the format)

Extract Trajectory Options:
  --start=START_FRAME           Index of the first frame to keep
  --end=END_FRAME               Index of the last frame to keep
  --skip=SKIP_FRAME             Extract one frame every SKIP_FRAME
  --nosolvent                   Do not extract solvent
  --discardequil                Detect and discard equilibration frames

General Options:
  -v, --verbose                 Print verbose output

"""

#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    utils.config_root_logger(args['--verbose'])

    if args['extract-trajectory']:
        return dispatch_extract_trajectory(args)

    analyze.analyze(args['--store'])
    return True


def dispatch_extract_trajectory(args):
    # Paths
    output_path = args['--trajectory']
    nc_path = args['--netcdf']

    # Get keyword arguments to pass to extract_trajectory()
    kwargs = {}

    if args['--state']:
        kwargs['state_index'] = int(args['--state'])
    else:
        kwargs['replica_index'] = int(args['--replica'])

    if args['--start']:
        kwargs['start_frame'] = int(args['--start'])
    if args['--skip']:
        kwargs['skip_frame'] = int(args['--skip'])
    if args['--end']:
        kwargs['end_frame'] = int(args['--end'])

    if args['--nosolvent']:
        kwargs['keep_solvent'] = False
    if args['--discardequil']:
        kwargs['discard_equilibration'] = True

    # Extract trajectory
    analyze.extract_trajectory(output_path, nc_path, **kwargs)

    return True
