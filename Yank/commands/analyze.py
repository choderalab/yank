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

# Module imports handled in individual functions since CLI should be faster to boot up

# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================

usage = """
YANK analyze

Usage: 
  yank analyze ((-s STORE | --store=STORE) | (-y YAML | --yaml=YAML)) [-e SERIAL | --serial=SERIAL] [--skipunbiasing] [--distcutoff=DISTANCE] [--energycutoff=ENERGY] [-v | --verbose] [--fulltraj]
  yank analyze report ((-s STORE | --store=STORE) | (-y YAML | --yaml=YAML)) (-o OUTPUT | --output=OUTPUT) [--format=FORMAT] [-e SERIAL | --serial=SERIAL] [--skipunbiasing] [--distcutoff=DISTANCE] [--energycutoff=ENERGY] [-v | --verbose] [--fulltraj]
  yank analyze extract-trajectory --netcdf=FILEPATH [--checkpoint=FILEPATH ] (--state=STATE | --replica=REPLICA) --trajectory=FILEPATH [--start=START_FRAME] [--skip=SKIP_FRAME] [--end=END_FRAME] [--nosolvent] [--discardequil] [--imagemol] [-v | --verbose]

Description:
  Analyze the data to compute Free Energies OR extract the trajectory from the NetCDF file into a common format.
  yank analyze report generates a Jupyter (Ipython) notebook of the report instead of writing it to standard output

Free Energy Required Arguments:
  -s STORE, --store=STORE       Storage directory for NetCDF data files. 
                                EXCLUSIVE with -y and --yaml
  -y YAML, --yaml=YAML          Target YAML file which setup and ran the experiment(s) being analyzed. 
                                This slightly changes the optional -o|--output flag.
                                EXCLUSIVE with -s and --store
  
YANK Analysis Output Arguments:
  -e SERIAL, --serial=SERIAL    Save data in Pickle serialized output. This behaves differently in report mode. 
                                In normal mode, this is a SINGULAR output file in Pickle format
                                In report mode, this is the base name of the individual serial files. If not provided, 
                                then the name is inferred from the storage (-s) or the yaml (-y) file
  report                        Toggles output to be of the Jupyter Notebook analysis as a rendered notebook or as 
                                a static file. Can use a path + name as well. File format is set by the --format flag
                                
  -o=REPORT, --output=REPORT    Name of the health report Jupyter notebook or static file, can use a path + name as well
                                If the filename ends in .pdf or .html, the notebook is auto run and converted to a
                                static PDF or HTML file respectively
                                PDF requires xelatex binary in OS path, often provided by LaTeX packages
                                MODIFIED BY -y|--yaml: This becomes the DIRECTORY of the output. The names are inferred 
                                from the input YAML file
  --format=FORMAT               File format of the notebook. If the filename ends in .pdf or .html, the notebook is run 
                                and converted to a static PDF or HTML file respectively. If --format is NOT set, it 
                                defaults to '.ipynb'                                

Free Energy Optional Arguments:
  --skipunbiasing               Skip the radially-symmetric restraint unbiasing. This can be an expensive step.
                                If this flag is not specified, and no cutoff is given, a distance cutoff is
                                automatically determined as the 99.9-percentile of the restraint distance distribution
                                in the bound state.
  --distcutoff=DISTANCE         The restraint distance cutoff (in angstroms) to be used to unbias the restraint.
                                When the restraint is unbiased, the analyzer discards all the samples for which the
                                distance between the restrained atoms is above this cutoff. Effectively, this is
                                equivalent to placing a hard wall potential at a restraint distance "distcutoff".
  --energycutoff=ENERGY         The restraint unitless potential energy cutoff (i.e. in kT) to be used to unbias the
                                restraint. When the restraint is unbiased, the analyzer discards all the samples for
                                which the restrain potential energy (in kT) is above this cutoff. Effectively, this is
                                equivalent to placing a hard wall potential at a restraint distance such that the
                                restraint potential energy is equal to "energycutoff".

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
  --fulltraj                    Force ALL analysis run from this command to rely on the full trajectory and not do any 
                                automatic equilibration detection or decorrelation subsampling. Although the 
                                equilibration and correlation times will still be computed, no calculation depending on 
                                them will use this information.

"""

# =============================================================================================
# COMMAND DISPATCH
# =============================================================================================


def dispatch(args):

    import os
    import pickle
    from .. import utils, analyze, mpi

    utils.config_root_logger(args['--verbose'])

    if args['report']:
        if not args['--format']:
            args['--format'] = '.ipynb'
        elif args['--format'][0] != '.':
            # Ensure format is not double dotted
            args['--format'] = '.' + args['--format']
        if args['--yaml'] is not None and args['--output']:
            # Ensure the last output is treated as a directory in all cases
            os.makedirs(args['--output'], exist_ok=True)
            base, last_item = os.path.split(args['--output'])
            if last_item != '':
                args['--output'] = os.path.join(base, last_item, '')
        return dispatch_report(args)

    if args['extract-trajectory']:
        return dispatch_extract_trajectory(args)

    # Configure analyzer keyword arguments.
    analyzer_kwargs = extract_analyzer_kwargs(args)
    do_serialize = True if args['--serial'] is not None else False
    if args['--yaml']:
        multi_analyzer = analyze.MultiExperimentAnalyzer(args['--yaml'])
        output = multi_analyzer.run_all_analysis(serial_data_path=args['--serial'], serialize_data=do_serialize,
                                                 **analyzer_kwargs)
        for exp_name, data in output.items():
            analyze.print_analysis_data(data, header="######## EXPERIMENT: {} ########".format(exp_name))

    else:
        @mpi.on_single_node(0)
        def single_run():
            # Helper to ensure case someone does MPI on a single diretory
            output = analyze.analyze_directory(args['--store'], **analyzer_kwargs)
            if do_serialize:
                with open(args['--serial'], 'wb') as f:
                    pickle.dump(output, f)
                print("Results have been serialized to {}".format(args['--serial']))
        single_run()
    return True


def extract_analyzer_kwargs(args, quantities_as_strings=False):

    import simtk.unit as unit

    """Return a dictionary with the keyword arguments to pass to the analyzer."""
    analyzer_kwargs = {}
    if args['--skipunbiasing']:
        analyzer_kwargs['unbias_restraint'] = False
    if args['--energycutoff']:
        analyzer_kwargs['restraint_energy_cutoff'] = float(args['--energycutoff'])
    if args['--distcutoff']:
        if quantities_as_strings:
            distcutoff = args['--distcutoff'] + '*angstroms'
        else:
            distcutoff = float(args['--distcutoff']) * unit.angstroms
        analyzer_kwargs['restraint_distance_cutoff'] = distcutoff
    if args['--fulltraj']:
        analyzer_kwargs['use_full_trajectory'] = True
    return analyzer_kwargs


def dispatch_extract_trajectory(args):

    import os
    from .. import analyze

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

    # Create output directory and save trajectory
    output_dir = os.path.dirname(output_path)
    if output_dir != '' and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    trajectory.save(output_path)

    return True


def dispatch_report(args):

    import io
    import os
    import re
    import pkg_resources
    from .. import analyze

    # Check modules for render
    store = args['--store']
    yaml_input = args['--yaml']
    output = args['--output']
    do_serialize = True if args['--serial'] is not None else False
    analyzer_kwargs = extract_analyzer_kwargs(args, quantities_as_strings=True)
    file_extension = args['--format']
    # requires_prerendering = file_extension.lower() in {'.pdf', '.html', '.ipynb'}

    try:
        import seaborn
        import matplotlib
        import jupyter
    except ImportError:
        error_msg = ("Rendering this notebook requires the following packages:\n"
                     " - seaborn\n"
                     " - matplotlib\n"
                     " - jupyter\n"
                     "Rendering as {} is not possible without the packages!".format(file_extension))
        raise ImportError(error_msg)

    def run_notebook(source_path, output_file, serial_file, **analyzer_kwargs):
        template_path = pkg_resources.resource_filename('yank', 'reports/YANK_Health_Report_Template.ipynb')
        with open(template_path, 'r') as template:
            notebook_text = re.sub('STOREDIRBLANK', source_path, template.read())
            notebook_text = re.sub('ANALYZERKWARGSBLANK', str(analyzer_kwargs), notebook_text)
            if serial_file is not None:
                # Uncomment the line. Traps '#' and the rest, reports only the rest
                notebook_text = re.sub(r"(#)(report\.dump_serial_data\('SERIALOUTPUT'\))", r'\2', notebook_text)
                notebook_text = re.sub('SERIALOUTPUT', serial_file, notebook_text)

        # Cast to static output
        print("Rendering notebook as a {} file...".format(file_extension))
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        import nbconvert.exporters
        # Categorize exporters based on extension, requires exporter object and data type output
        # 'b' = byte types output, e.g. PDF
        # 't' = text based output, e.g. HTML or even raw notebook, human-readable-like
        exporters = {
            ".pdf": {'exporter': nbconvert.exporters.PDFExporter, 'write_type': 'b'},
            ".html": {'exporter': nbconvert.exporters.HTMLExporter, 'write_type': 't'},
            ".ipynb": {'exporter': nbconvert.exporters.NotebookExporter, 'write_type': 't'}
        }

        # Load the notebook through Jupyter.
        loaded_notebook = nbformat.read(io.StringIO(notebook_text), as_version=4)
        # Process the notebook.
        ep = ExecutePreprocessor(timeout=None)
        # Sometimes the default startup timeout exceed the default of 60 seconds.
        ep.startup_timeout = 180
        # Set the title name, does not appear in all exporters
        _, output_file_name = os.path.split(output_file)
        resource_data = {'metadata': {'name': 'YANK Simulation Report: {}'.format(output_file_name)}}
        print("Processing notebook now, this may take a while...")
        processed_notebook, resources = ep.preprocess(loaded_notebook, resource_data)

        # Retrieve exporter
        exporter_data = exporters[file_extension.lower()]
        # Determine exporter and data output type
        exporter = exporter_data['exporter']
        write_type = exporter_data['write_type']
        with open(output_file, 'w{}'.format(write_type)) as notebook:
            exported_notebook, _ = nbconvert.exporters.export(exporter, processed_notebook, resources=resources)
            notebook.write(exported_notebook)

    def cast_notebook_serial_path(relative_notebook_path):
        if args['--serial'] is None:
            serial_file = None
        else:
            serial_file = os.path.splitext(relative_notebook_path)[0] + '_' + args['--serial']
        return serial_file

    class NotebookMultiExperimentAnalyzer(analyze.MultiExperimentAnalyzer):
        """Custom Multi Experiment Analyzer for notebooks"""

        @staticmethod
        def _run_specific_analysis(path, **analyzer_kwargs):
            _, exp_name = os.path.split(path)
            single_output_file = os.path.join(output, exp_name + args['--format'])
            single_serial_file = cast_notebook_serial_path(single_output_file)
            run_notebook(path, single_output_file, single_serial_file, **analyzer_kwargs)
            return

        @staticmethod
        def _serialize(serial_path, payload):
            """The notebooks do not have a general serial dump"""
            pass

    if yaml_input is not None:
        multi_notebook = NotebookMultiExperimentAnalyzer(yaml_input)
        _ = multi_notebook.run_all_analysis(serialize_data=do_serialize,
                                            serial_data_path=args['--serial'],
                                            **analyzer_kwargs)
    else:
        notebook_serial_file = cast_notebook_serial_path(output)
        run_notebook(store, output, notebook_serial_file, **analyzer_kwargs)

    return True
