#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Set up and run YANK calculation from script.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
from ..yamlbuild import YamlBuilder


#=============================================================================================
# COMMAND-LINE INTERFACE
#=============================================================================================

usage = """
YANK script

Usage:
  yank script (-y=FILEPATH | --yaml=FILEPATH) [--platform=PLATFORM]

Description:
  Set up and run free energy calculations from a YAML scrpit. All options can be specified in the YAML script. 

Required Arguments:
  -y, --yaml=FILEPATH           Path to the YAML script specifying options and/or how to set up and run the experiment.

"""

#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    """
    Set up and run YANK calculation from a script.

    Parameters
    ----------
    args : dict
       Command-line arguments from docopt.

    """
    if args['--yaml']:
        yaml_path = args['--yaml']

        if not os.path.isfile(yaml_path):
            raise ValueError('Cannot find YAML script "{}"'.format(yaml_path))

        yaml_builder = YamlBuilder(yaml_source=yaml_path)
        yaml_builder.build_experiments()
        return True

    return False
