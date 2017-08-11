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
import os
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
    file_full_path, file_extension = os.path.splitext(output)
    _, file_base_name = os.path.split(file_full_path)
    # PDF requires xelatex binary in the OS (provided by LaTeX such as TeXLive and MiKTeX)
    static_extensions = [".pdf", ".html"]
    try:
        import matplotlib
        import jupyter
    except ImportError:
        error_msg = ("Rendering this notebook requires the following packages:\n"
                     " - matplotlib\n"
                     " - jupyter\n"
                     "These are not required to generate the notebook however")
        if file_extension.lower() in static_extensions:
            error_msg += "\nRendering as static {} is not possible without the packages!".format(file_extension)
            raise ImportError(error_msg)
        else:
            print(error_msg)
    template_path = pkg_resources.resource_filename('yank', 'reports/YANK_Health_Report_Template.ipynb')
    with open(template_path, 'r') as template:
        notebook_text = re.sub('STOREDIRBLANK', store, template.read())
    if file_extension.lower() in static_extensions:
        # Cast to static output
        print("Rendering notebook as static file...")
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        import nbconvert.exporters
        # Categorize exporters based on extension, requires exporter object and data type output
        # 'b' = byte types output, e.g. PDF
        # 't' = text based output, e.g. HTML or even raw notebook, human-readable-like
        exporters = {".pdf": {'exporter': nbconvert.exporters.PDFExporter, 'write_type': 'b'},
                     ".html": {'exporter': nbconvert.exporters.HTMLExporter, 'write_type': 't'}
                     }
        temporary_directory = analyze.mmtools.utils.temporary_directory
        with temporary_directory() as tmp_dir_path:
            temp_notebook_name = "tmp_notebook.ipynb"
            tmp_nb_path = os.path.join(tmp_dir_path, temp_notebook_name)
            # Write out temporary notebook to raw text
            with open(tmp_nb_path, 'w') as tmp_notebook:
                tmp_notebook.write(notebook_text)
            # Read the temp notebook into a notebook_node object nbconvert can work with
            with open(tmp_nb_path, 'r') as tmp_notebook:
                loaded_notebook = nbformat.read(tmp_notebook, as_version=4)
            ep = ExecutePreprocessor(timeout=None)
            # Set the title name, does not appear in all exporters
            resource_data = {'metadata': {'name': 'YANK Simulation Report: {}'.format(file_base_name)}}
            # Process notebook
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
    else:
        # No pre-rendering, no need to process anything
        with open(output, 'w') as notebook:
            notebook.write(notebook_text)

    return True
