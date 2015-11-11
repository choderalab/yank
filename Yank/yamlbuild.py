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

import os
import yaml
import logging
logger = logging.getLogger(__name__)

import openmoltools
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

    SETUP_DIR = 'setup'
    SETUP_MOLECULES_DIR = os.path.join(SETUP_DIR, 'molecules')

    DEFAULT_OPTIONS = {
        'verbose': False,
        'mpi': False,
        'platform': None,
        'precision': None,
        'resume': False,
        'output_dir': 'output/'
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
            opts = yaml_config.pop('options')
        except KeyError:
            opts = {}
            logger.warning('No YAML options found.')
        opts.update(yaml_config.pop('metadata', {}))

        # Store YAML builder options
        for opt, default in self.DEFAULT_OPTIONS.items():
            setattr(self, '_' + opt, opts.pop(opt, default))

        # Validate and store yank and repex options
        template_options = Yank.default_parameters.copy()
        template_options.update(ReplicaExchange.default_parameters)
        try:
            self._options = utils.validate_parameters(opts, template_options, check_unknown=True,
                                                      process_units_str=True, float_to_int=True)
        except (TypeError, ValueError) as e:
            logger.error(str(e))
            raise YamlParseError(str(e))

        # Store other fields, we don't raise an error if we cannot find any
        # since the YAML file could be used only to specify the options
        self._molecules = yaml_config.pop('molecules', {})
        # TODO - verify that filepath and name are not specified simultaneously

    def build_experiment(self):
        """Build the Yank experiment (TO BE IMPLEMENTED)."""
        raise NotImplemented

    def _setup_molecule(self, molecule_id):
        mol_descr = self._molecules[molecule_id]

        # Create output directory
        output_mol_dir = os.path.join(self._output_dir, self.SETUP_MOLECULES_DIR,
                                      molecule_id)
        if not os.path.isdir(output_mol_dir):
            os.makedirs(output_mol_dir)

        # Check molecule source
        if 'filepath' in mol_descr:
            input_mol_path = os.path.abspath(mol_descr['filepath'])
        elif 'name' in mol_descr:
            input_mol_path = os.path.abspath(os.path.join(output_mol_dir,
                                                          molecule_id + '.mol2'))

            # Generate molecule from name with OpenEye
            try:
                molecule = openmoltools.openeye.iupac_to_oemol(mol_descr['name'])
                molecule = openmoltools.openeye.get_charges(molecule, keep_confs=1)
                openmoltools.openeye.molecule_to_mol2(molecule, input_mol_path)
            except ImportError as e:
                error_msg = ('requested molecule generation from name but '
                             'could not find OpenEye toolkit: ' + str(e))
                raise YamlParseError(error_msg)

        with utils.temporary_cd(output_mol_dir):
            openmoltools.amber.run_antechamber(molecule_id, input_mol_path)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
