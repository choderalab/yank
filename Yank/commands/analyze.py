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

import io
import re
import os

from simtk import unit

import pkg_resources
from .. import utils, analyze

# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================

usage = """
YANK analyze

Usage:
  yank analyze (-s STORE | --store=STORE) [--skipunbiasing] [--distcutoff=DISTANCE] [--energycutoff=ENERGY] [-v | --verbose]
  yank analyze report (-s STORE | --store=STORE) [--skipunbiasing] [--distcutoff=DISTANCE] [--energycutoff=ENERGY] (-o REPORT | --output=REPORT)
  yank analyze extract-trajectory --netcdf=FILEPATH [--checkpoint=FILEPATH ] (--state=STATE | --replica=REPLICA) --trajectory=FILEPATH [--start=START_FRAME] [--skip=SKIP_FRAME] [--end=END_FRAME] [--nosolvent] [--discardequil] [--imagemol] [-v | --verbose]

Description:
  Analyze the data to compute Free Energies OR extract the trajectory from the NetCDF file into a common fortmat.
  yank analyze report generates a Jupyter (Ipython) notebook of the report instead of writing it to standard output

Free Energy Required Arguments:
  -s=STORE, --store=STORE       Storage directory for NetCDF data files.

Free Energy Optional Arguments:
  --skipunbiasing               Skip the radially-symmetric restraint unbiasing. This can be an expensive step.
  --distcutoff=DISTANCE         The restraint distance cutoff (in angstroms) to be used to unbias the restraint.
  --energycutoff=ENERGY         The restraint unitless potential energy cutoff (i.e. in kT) to be used to unbias the
                                restraint.

YANK Health Report Arguments:
  -o=REPORT, --output=REPORT    Name of the health report Jupyter notebook or static file, can use a path + name as well
                                If the filename ends in .pdf or .html, the notebook is auto run and converted to a
                                static PDF or HTML file respectively
                                PDF requires xelatex binary in OS path, often provided by LaTeX packages
                                Static generation may be slow

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

    # Configure analyzer keyword arguments.
    analyzer_kwargs = extract_analyzer_kwargs(args)
    analyze.analyze_directory(args['--store'], **analyzer_kwargs)
    return True


def extract_analyzer_kwargs(args, quantities_as_strings=False):
    """Return a dictionary with the keyword arguments to pass to the analyzer."""
    analyzer_kwargs = {}
    if args['skipunbiasing']:
        analyzer_kwargs['unbias_restraint'] = False
    if args['energycutoff']:
        analyzer_kwargs['restraint_energy_cutoff'] = float(args['energycutoff'])
    if args['distcutoff']:
        if quantities_as_strings:
            distcutoff = args['distcutoff'] + '*angstroms'
        else:
            distcutoff = float(args['distcutoff']) * unit.angstroms
        analyzer_kwargs['restraint_distance_cutoff'] = distcutoff
    return analyzer_kwargs


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
    trajectory = analyze.extract_trajectory(nc_path, **kwargs)

    # Detect output format.
    extension = os.path.splitext(output_path)[1][1:]  # remove dot
    try:
        save_function = getattr(trajectory, 'save_' + extension)
    except AttributeError:
        raise ValueError('Cannot detect format from extension of file {}'.format(output_path))

    # Create output directory and save trajectory
    output_dir = os.path.dirname(output_path)
    if output_dir != '' and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    save_function(output_path)

    return True


def dispatch_report(args):
    # Check modules for render
    store = args['--store']
    output = args['--output']
    analyzer_kwargs = extract_analyzer_kwargs(args, quantities_as_strings=True)
    file_full_path, file_extension = os.path.splitext(output)
    _, file_base_name = os.path.split(file_full_path)

    # If we need to pre-render the notebook, check if we have the necessary libraries.
    # PDF requires xelatex binary in the OS (provided by LaTeX such as TeXLive and MiKTeX)
    requires_prerendering = file_extension.lower() in {'.pdf', '.html'}
    try:
        import matplotlib
        import jupyter
    except ImportError:
        error_msg = ("Rendering this notebook requires the following packages:\n"
                     " - matplotlib\n"
                     " - jupyter\n"
                     "These are not required to generate the notebook however")
        if requires_prerendering:
            error_msg += "\nRendering as static {} is not possible without the packages!".format(file_extension)
            raise ImportError(error_msg)
        else:
            print(error_msg)

    # Read template notebook and inject variables.
    template_path = pkg_resources.resource_filename('yank', 'reports/YANK_Health_Report_Template.ipynb')
    with open(template_path, 'r') as template:
        notebook_text = re.sub('STOREDIRBLANK', store, template.read())
        notebook_text = re.sub('ANALYZERKWARGSBLANK', str(analyzer_kwargs), notebook_text)

    # Determine whether to pre-render the notebook or not.
    if not requires_prerendering:
        # No pre-rendering, no need to process anything
        with open(output, 'w') as notebook:
            notebook.write(notebook_text)
    else:
        # Cast to static output
        print("Rendering notebook as static file...")
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        import nbconvert.exporters
        # Categorize exporters based on extension, requires exporter object and data type output
        # 'b' = byte types output, e.g. PDF
        # 't' = text based output, e.g. HTML or even raw notebook, human-readable-like
        exporters = {
            ".pdf": {'exporter': nbconvert.exporters.PDFExporter, 'write_type': 'b'},
            ".html": {'exporter': nbconvert.exporters.HTMLExporter, 'write_type': 't'}
        }

        # Load the notebook through Jupyter.
        loaded_notebook = nbformat.read(io.StringIO(notebook_text), as_version=4)
        # Process the notebook.
        ep = ExecutePreprocessor(timeout=None)
        # Set the title name, does not appear in all exporters
        resource_data = {'metadata': {'name': 'YANK Simulation Report: {}'.format(file_base_name)}}
        print("Processing notebook now, this may take a while...")
        processed_notebook, resources = ep.preprocess(loaded_notebook, resource_data)

        # Retrieve exporter
        exporter_data = exporters[file_extension.lower()]
        # Determine exporter and data output type
        exporter = exporter_data['exporter']
        write_type = exporter_data['write_type']
        with open(output, 'w{}'.format(write_type)) as notebook:
            exported_notebook, _ = nbconvert.exporters.export(exporter, processed_notebook, resources=resources)
            notebook.write(exported_notebook)

    return True
