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

    _accepted_options = frozenset(['title',
                                   'timestep', 'nsteps_per_iteration', 'number_of_iterations',
                                   'minimize','equilibrate', 'equilibration_timestep',
                                   'number_of_equilibration_iterations'])
    _expected_options_types = (('timestep', process_second_compatible_quantity),
                               ('nsteps_per_iteration', int),
                               ('number_of_iterations', int),
                               ('minimize', process_bool),
                               ('equilibrate', process_bool),
                               ('equilibration_timestep', process_second_compatible_quantity),
                               ('number_of_equilibration_iterations', int))

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

        # Set only accepted options
        self._options = {x: opts[x] for x in opts if x in YamlBuilder._accepted_options}
        if len(self._options) != len(opts):
            unknown_opts = {x for x in opts if x not in YamlBuilder._accepted_options}
            error_msg = 'YAML configuration contains unidentifiable options: '
            error_msg += ', '.join(unknown_opts)
            logger.error(error_msg)
            raise YamlParseError(error_msg)

        # Enforce types that are not automatically recognized by yaml
        for special_opt, casting_func in YamlBuilder._expected_options_types:
            if special_opt in self._options:
                try:
                    self._options[special_opt] = casting_func(self._options[special_opt])
                except (TypeError, ValueError) as e:
                    error_msg = 'YAML option %s: %s' % (special_opt, str(e))
                    logger.error(error_msg)
                    raise YamlParseError(error_msg)

    def build_experiment(self):
        """Build the Yank experiment (TO BE IMPLEMENTED)."""
        raise NotImplemented

if __name__ == "__main__":
    import doctest
    doctest.testmod()
