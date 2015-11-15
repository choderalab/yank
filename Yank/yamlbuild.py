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

from simtk import unit

import utils
from yank import Yank
from repex import ReplicaExchange


#=============================================================================================
# UTILITY FUNCTIONS
#=============================================================================================

def process_second_compatible_quantity(quantity_str):
    """
    Shortcut to process a string containing a quantity compatible with seconds.

    Parameters
    ----------
    quantity_str : str
         A string containing a value with a unit of measure for time.

    Returns
    -------
    quantity : simtk.unit.Quantity
       The specified string, returned as a Quantity.

    See Also
    --------
    utils.process_unit_bearing_str : the function used for the actual conversion.

    """
    return utils.process_unit_bearing_str(quantity_str, unit.seconds)

def process_bool(bool_val):
    """Raise ValueError if this an ambiguous representation of a bool.

    PyYAML load a boolean the following words: true, True, yes, Yes, false, False,
    no and No, but in Python strings and numbers can be used as booleans and create
    subtle errors. This function ensure that the value is a true boolean.

    Returns
    -------
    bool
        bool_val only if it is a boolean.

    Raises
    ------
    ValueError
        If bool_var is not a boolean.

    """
    if isinstance(bool_val, bool):
        return bool_val
    else:
        raise ValueError('The value must be true, yes, false, or no.')

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

    default_options = {
        'verbose': False,
        'mpi': False,
        'platform': None,
        'precision': None,
        'resume': False,
        'output_directory': 'output/'
    }

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

        # Find and merge options and metadata
        try:
            opts = yaml_config['options']
        except KeyError:
            opts = {}
            logger.warning('No YAML options found.')
        try:
            opts.update(yaml_config['metadata'])
        except KeyError:
            pass

        # Store YAML builder options
        self._options = {par: opts.pop(par, default)
                         for par, default in self.default_options.items()}

        # Store yank and repex options
        template_options = Yank.default_parameters.copy()
        template_options.update(ReplicaExchange.default_parameters)
        try:
            opts = utils.validate_parameters(opts, template_options, check_unknown=True,
                                             process_units_str=True, float_to_int=True)
        except (TypeError, ValueError) as e:
            logger.error(str(e))
            raise YamlParseError(str(e))
        self._options.update(opts)

    def build_experiment(self):
        """Build the Yank experiment (TO BE IMPLEMENTED)."""
        raise NotImplemented

if __name__ == "__main__":
    import doctest
    doctest.testmod()
