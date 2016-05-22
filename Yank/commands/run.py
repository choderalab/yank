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

    store_directory = args['--store']

    # Set override options.
    options = dict()

    # Configure MPI, if requested.
    mpicomm = None
    if args['--mpi']:
        # Initialize MPI.
        from mpi4py import MPI
        hostname = os.uname()[1]
        MPI.COMM_WORLD.barrier()
        logger.info("Initialized MPI on %d processes." % (MPI.COMM_WORLD.size))
        mpicomm = MPI.COMM_WORLD
        # Set all nodes to abort on exception.
        # TODO: This is just a band-aid, and may result in file corruption.
        # TODO: Instead, look into having all running code exit gracefully using context managers.
        # TODO: Add tests for signal aborts.
        import signal, os
        def cleanup(*args):
            print('Caught signal.')
            print(args)
            print('Exiting via call to MPI.Abort()')
            MPI.COMM_WORLD.Abort(1)
            os._exit(1)
        for sig in (signal.SIGABRT, signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, cleanup)
        if MPI.COMM_WORLD.rank == 0:
            logger.info("Configured all MPI processes to call MPI.Abort() on exception.")

    # Configure logger
    utils.config_root_logger(args['--verbose'], mpicomm=mpicomm,
                             log_file_path=os.path.join(store_directory, 'run.log'))

    if args['--iterations']:
        options['number_of_iterations'] = int(args['--iterations'])
    if args['--online-analysis']:
        options['online_analysis'] = True
    if args['--platform'] not in [None, 'None']:
        options['platform'] = openmm.Platform.getPlatformByName(args['--platform'])
    if args['--precision']:
        # We need to modify the Platform object.
        if args['--platform'] is None:
            raise Exception("The --platform argument must be specified in order to specify platform precision.")

        # Set platform precision.
        precision = args['--precision']
        platform_name = args['--platform']
        logger.info("Setting %s platform to use precision model '%s'." % (platform_name, precision))
        if precision is not None:
            if platform_name == 'CUDA':
                options['platform'].setPropertyDefaultValue('CudaPrecision', precision)
            elif platform_name == 'OpenCL':
                options['platform'].setPropertyDefaultValue('OpenCLPrecision', precision)
            elif platform_name == 'CPU':
                if precision != 'mixed':
                    raise Exception("CPU platform does not support precision model '%s'; only 'mixed' is supported." % precision)
            elif platform_name == 'Reference':
                if precision != 'double':
                    raise Exception("Reference platform does not support precision model '%s'; only 'double' is supported." % precision)
            else:
                raise Exception("Platform selection logic is outdated and needs to be updated to add platform '%s'." % platform_name)

    # Create YANK object associated with data storage directory.
    yank = Yank(store_directory, mpicomm=mpicomm, **options)

    # Set YANK to resume from the store file.
    phases = None # By default, resume from all phases found in store_directory
    if args['--phase']: phases=[args['--phase']]
    yank.resume(phases=phases)

    # Run simulation.
    yank.run()

    return True
