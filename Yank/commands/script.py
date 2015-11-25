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
        yaml_builder = YamlBuilder(yaml_source=args['--yaml'])
        yaml_builder.build_experiment()
        return True

    return False
