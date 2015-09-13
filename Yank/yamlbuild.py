#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Tools to build Yank experiments from a YAML configuration file.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import yaml
import logging
logger = logging.getLogger(__name__)


#=============================================================================================
# BUILDER CLASS
#=============================================================================================

class YamlParseError(Exception):
    """Represent errors occurring during parsing of Yank YAML file."""
    pass

class YamlBuilder:
    """Parse YAML configuration file and build the experiment.

    Properties
    ----------
    options : dict
        The options specified in the parsed YAML file.

    """

    _accepted_options = frozenset(['timestep', 'nsteps_per_iteration'])

    @property
    def options(self):
        return self._options

    def __init__(self, yaml_file):
        """Parse the given YAML configuration file.

        This does not build the actual experiment but simply checks that the syntax
        is correct and loads the configuration into memory.

        Parameters
        ----------
        yaml_file : str
            A relative or absolute path to the YAML configuration file.

        """

        # TODO check version of yank-yaml language
        # TODO what if there are multiple streams in the YAML file?
        with open(yaml_file, 'r') as f:
            yaml_config = yaml.load(f)

        if yaml_config is None:
            error_msg = 'The YAML file is empty!'
            logger.error(error_msg)
            raise YamlParseError(error_msg)

        # Set options only accepted options
        try:
            opts = yaml_config['options']
            self._options = {x: opts[x] for x in opts if x in YamlBuilder._accepted_options}
            if len(self._options) != len(yaml_config['options']):
                unknown_opts = {x for x in opts if x not in YamlBuilder._accepted_options}
                error_msg = 'YAML configuration contains unidentifiable options: '
                error_msg += ', '.join(unknown_opts)
                logger.error(error_msg)
                raise YamlParseError(error_msg)
        except KeyError:
            self._options = {}
            logger.warning('No YAML options found.')

    def build_experiment(self):
        """Build the Yank experiment (TO BE IMPLEMENTED)."""
        raise NotImplemented
