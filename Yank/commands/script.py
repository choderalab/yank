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
from yank.yamlbuild import YamlBuilder


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
