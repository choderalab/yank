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
