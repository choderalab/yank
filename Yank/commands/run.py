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
import logging
logger = logging.getLogger(__name__)

from simtk import openmm
from simtk import unit
from simtk.openmm import app

from yank import utils

#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    from yank.yank import Yank # TODO: Fix this awkward import syntax.

    # Create YANK object associated with data storage directory.
    store_directory = args['--store']
    yank = Yank(store_directory)

    # Set override options.
    options = dict()
    if args['--iterations']:
        options['number_of_iterations'] = int(args['--iterations'])
    if args['--verbose']:
        options['verbose'] = True
    if args['--online-analysis']:
        options['online_analysis'] = True
    if args['--platform'] not in [None, 'None']:
        options['platform'] = openmm.Platform.getPlatformByName(args['--platform'])

    # Configure logger
    utils.config_root_logger(options['verbose'],
                             log_file_path=os.path.join(store_directory, 'run.log'))

    # Set YANK to resume from the store file.
    phases = None # By default, resume from all phases found in store_directory
    if args['--phase']: phases=[args['--phase']]
    yank.resume(phases=phases)

    # Configure MPI, if requested.
    mpicomm = None
    if args['--mpi']:
        # Initialize MPI.
        from mpi4py import MPI
        hostname = os.uname()[1]
        MPI.COMM_WORLD.barrier()
        logger.info("Initialized MPI on %d processes." % (MPI.COMM_WORLD.size))
        mpicomm = MPI.COMM_WORLD

    # Run simulation.
    yank.run(mpicomm=mpicomm, options=options)

    return True
