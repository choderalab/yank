#!/usr/local/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Analyze YANK output file.

"""

# =============================================================================================
# MODULE IMPORTS
# =============================================================================================

from .. import utils, analyze
import re
import pkg_resources

# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================

usage = """
YANK analyze

Usage:
  yank analyze (-s STORE | --store=STORE) [-v | --verbose]
  yank analyze report (-s STORE | --store=STORE) (-o REPORT | --output=REPORT)
  yank analyze extract-trajectory --netcdf=FILEPATH [--checkpoint=FILEPATH ] (--state=STATE | --replica=REPLICA) --trajectory=FILEPATH [--start=START_FRAME] [--skip=SKIP_FRAME] [--end=END_FRAME] [--nosolvent] [--discardequil] [--imagemol] [-v | --verbose]

Description:
  Analyze the data to compute Free Energies OR extract the trajectory from the NetCDF file into a common fortmat.
  yank analyze report generates a Jupyter (Ipython) notebook of the report instead of writing it to standard output

Free Energy Required Arguments:
  -s=STORE, --store=STORE       Storage directory for NetCDF data files.

YANK Health Report Arguments:
  -o=REPORT, --output=REPORT    Name of the health report Jupyter notebook, can use a path + name as well

Extract Trajectory Required Arguments:
  --netcdf=FILEPATH             Path to the NetCDF file.
  --checkpoint=FILEPATH         Path to the NetCDF checkpoint file if not the default name inferned from "netcdf" option
  --state=STATE_IDX             Index of the alchemical state for which to extract the trajectory
  --replica=REPLICA_IDX         Index of the replica for which to extract the trajectory
  --trajectory=FILEPATH         Path to the trajectory file to create (extension determines the format)

Extract Trajectory Options:
  --start=START_FRAME           Index of the first frame to keep
  --end=END_FRAME               Index of the last frame to keep
  --skip=SKIP_FRAME             Extract one frame every SKIP_FRAME
  --nosolvent                   Do not extract solvent
  --discardequil                Detect and discard equilibration frames
  --imagemol                    Reprocess trajectory to enforce periodic boundary conditions to molecules positions

General Options:
  -v, --verbose                 Print verbose output

"""

# =============================================================================================
# COMMAND DISPATCH
# =============================================================================================


def dispatch(args):
    utils.config_root_logger(args['--verbose'])

    if args['report']:
        return dispatch_report(args)

    if args['extract-trajectory']:
        return dispatch_extract_trajectory(args)

    analyze.analyze_directory(args['--store'])
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
    if args['--imagemol']:
        kwargs['image_molecules'] = True
    if args['--checkpoint']:
        kwargs["nc_checkpoint_file"] = args['--checkpoint']

    # Extract trajectory
    analyze.extract_trajectory(output_path, nc_path, **kwargs)

    return True


def dispatch_report(args):
    # Check modules for render
    try:
        import matplotlib
    except ImportError:
        print("Rendering this notebook requires the following packages:\n"
              " - matplotlib\n"
              "These are not required to generate the notebook however")
    store = args['--store']
    output = args['--output']
    template_path = pkg_resources.resource_filename('yank', 'reports/YANK_Health_Report_Template.ipynb')
    with open(template_path, 'r') as template:
        notebook_text = re.sub('STOREDIRBLANK', store, template.read())
    with open(output, 'w') as notebook:
        notebook.write(notebook_text)

    return True
