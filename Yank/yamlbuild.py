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
    SETUP_SYSTEMS_DIR = os.path.join(SETUP_DIR, 'systems')
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
        # TODO - verify that filepath and name/smiles are not specified simultaneously

        self._solvents = yaml_config.get('solvents', {})

        # Check if there is a sequence of experiments or a single one
        self._experiments = yaml_config.get('experiments', None)
        if self._experiments is None:
            self._experiments = [utils.CombinatorialTree(yaml_config.get('experiment', {}))]
        else:
            for i, exp_name in enumerate(self._experiments):
                self._experiments[i] = utils.CombinatorialTree(yaml_config[exp_name])

    def build_experiment(self):
        """Build the Yank experiment (TO BE IMPLEMENTED)."""
        raise NotImplemented

    def _expand_experiments(self):
        """Iterate over all possible combinations of experiments"""
        for exp in self._experiments:
            for combination in exp:
                yield combination

    def _setup_molecule(self, molecule_id):
        mol_descr = self._molecules[molecule_id]

        # Create output directory
        output_mol_dir = os.path.join(self._output_dir, self.SETUP_MOLECULES_DIR,
                                      molecule_id)
        if not os.path.isdir(output_mol_dir):
            os.makedirs(output_mol_dir)

        # Check if we have to generate the molecule with OpenEye
        if 'filepath' not in mol_descr:
            try:
                if 'name' in mol_descr:
                    molecule = openmoltools.openeye.iupac_to_oemol(mol_descr['name'])
                elif 'smiles' in mol_descr:
                    molecule = openmoltools.openeye.smiles_to_oemol(mol_descr['smiles'])
                molecule = openmoltools.openeye.get_charges(molecule, keep_confs=1)
                mol_descr['filepath'] = os.path.join(output_mol_dir, molecule_id + '.mol2')
                openmoltools.openeye.molecule_to_mol2(molecule, mol_descr['filepath'])
            except ImportError as e:
                error_msg = ('requested molecule generation from name but '
                             'could not find OpenEye toolkit: ' + str(e))
                raise YamlParseError(error_msg)

        # Check if we need to parametrize the molecule
        if mol_descr['parameters'] == 'antechamber':

            # Generate parameters
            input_mol_path = os.path.abspath(mol_descr['filepath'])
            with utils.temporary_cd(output_mol_dir):
                openmoltools.amber.run_antechamber(molecule_id, input_mol_path)

            # Save new parameters paths
            mol_descr['filepath'] = os.path.join(output_mol_dir, molecule_id + '.gaff.mol2')
            mol_descr['parameters'] = os.path.join(output_mol_dir, molecule_id + '.frcmod')

    def _setup_system(self, components, output_dir):
        # Create output directory
        output_sys_dir = os.path.join(self._output_dir, self.SETUP_SYSTEMS_DIR,
                                      output_dir)
        if not os.path.isdir(output_sys_dir):
            os.makedirs(output_sys_dir)

        # Create tleap script
        tleap = utils.TLeap()
        tleap.new_section('Load GAFF parameters')
        tleap.load_parameters('leaprc.gaff')

        # Implicit solvent
        solvent = self._solvents[components['solvent']]
        if solvent['nonbondedMethod'] == 'NoCutoff' and 'gbsamodel' in solvent:
            tleap.new_section('Set GB radii to recommended values for OBC')
            tleap.add_commands('set default PBRadii mbondi2')

        # Load receptor and ligand
        for group_name in ['receptor', 'ligand']:
            group = self._molecules[components[group_name]]
            tleap.new_section('Load ' + group_name)
            tleap.load_parameters(group['parameters'])
            tleap.load_group(name=group_name, file_path=group['filepath'])

        # Create complex
        tleap.new_section('Create complex')
        tleap.combine('complex', 'receptor', 'ligand')
        tleap.add_commands('check complex', 'charge complex')

        # Save prmtop and inpcrd files
        tleap.new_section('Save prmtop and inpcrd files')
        tleap.save_group('complex', os.path.join(output_sys_dir, 'complex.prmtop'))
        tleap.save_group('ligand', os.path.join(output_sys_dir, 'solvent.prmtop'))

        # Save tleap script for reference
        tleap.export_script(os.path.join(output_sys_dir, 'leap.in'))

        # Run tleap!
        tleap.run()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
