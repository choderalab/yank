#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Run a YANK calculation.

"""

#=============================================================================================
# MODULE IMPORTS
#=============================================================================================

import os

#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    # Create YANK object associated with data storage directory.
    from yank.yank import Yank
    yank = Yank(output_directory=args['--store'], verbose=args['--verbose'])

    # Configure YANK object with command-line parameter overrides.
    if args['--niterations']:
        yank.niterations = int(args['--niterations'])
    if args['--verbose']:
        yank.verbose = True
    if args['--online-analysis']:
        yank.online_analysis = True
    if args['--restraints']:
        yank.restraint_type = args['--restraints']
    if args['--randomize-ligand']:
        yank.randomize_ligand = True
    if args['--platform']:
        from simtk import openmm
        yank.platform = openmm.Platform.getPlatformByName(args['--platform'])


    # Run calculation.
    if args['--mpi']:
        # Initialize MPI.
        from mpi4py import MPI
        hostname = os.uname()[1]
        if not MPI.COMM_WORLD.rank == 0:
            yank.verbose = False
        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0: print "Initialized MPI on %d processes." % (MPI.COMM_WORLD.size)

        # Run MPI version.
        yank.run_mpi(MPI)
    else:
        # Run serial version.
        yank.run()
