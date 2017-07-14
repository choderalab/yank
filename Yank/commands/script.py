#!/usr/local/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Set up and run YANK calculation from script.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import os
from ..experiment import ExperimentBuilder


# =============================================================================================
# COMMAND-LINE INTERFACE
# =============================================================================================

usage = """
YANK script

Usage:
  yank script (-y FILEPATH | --yaml=FILEPATH) [-o OVERRIDE]...

Description:
  Set up and run free energy calculations from a YAML script. All options can be specified in the YAML script.

Required Arguments:
  -y, --yaml=FILEPATH           Path to the YAML script specifying options and/or how to set up and run the experiment.

Optional Arguments:

  -o, --override=OVERRIDE       Override a single option in the script file. May be specified multiple times.
                                Specified as a nested dictionary of the form:
                                top_option:sub_option:value
                                of any depth level where the last entry is always the value, do not use quotes.
                                Please see script file documentation for valid options.
                                Specifying the same option multiple times results in an error.
                                This method is not recommended for complex options such as lists or combinations
"""

# =============================================================================================
# COMMAND DISPATCH
# =============================================================================================


def dispatch(args):
    """
    Set up and run YANK calculation from a script.

    Parameters
    ----------
    args : dict
       Command-line arguments from docopt.

    """
    override = None
    if args['--override']:  # Is False for None and [] (empty list)
        over_opts = args['--override']
        # Check for duplicates
        if len(over_opts) != len(set(over_opts)):
            raise ValueError("There were duplicate override options, result will be ambiguous!")
        # Create a dict of strings
        override_dict = {}
        for opt in over_opts:
            split_opt = opt.split(':')
            try:
                key, value = split_opt[-2:]
            except IndexError:
                raise IndexError("Override option {} is not of the form top_dict:key:value. Minimal accepted case is "
                                 "key:value".format(opt))
            top_dict = override_dict
            for opt_level in split_opt[:-2]:
                try:
                    # Dict already exists
                    top_dict[opt_level]
                except KeyError:
                    top_dict[opt_level] = {}
                finally:
                    top_dict = top_dict[opt_level]
            top_dict[key] = value
        # Create a string that looks like a dictionary, removing quotes
        # This is done to avoid input type ambiguity and instead let the parser handle it as though it were a file
        override = str(override_dict).replace("'", "").replace('"', '')

    if args['--yaml']:
        yaml_path = args['--yaml']

        if not os.path.isfile(yaml_path):
            raise ValueError('Cannot find YAML script "{}"'.format(yaml_path))

        yaml_builder = ExperimentBuilder(yaml_source=yaml_path)
        if override:  # Parse the string present.
            yaml_builder.update_yaml(override)
        yaml_builder.run_experiments()
        return True

    return False
