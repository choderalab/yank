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
import re
import yaml
import logging
logger = logging.getLogger(__name__)

import numpy as np
import openmoltools
from simtk import unit

import utils
from yank import Yank
from repex import ReplicaExchange
from sampling import ModifiedHamiltonianExchange


#=============================================================================================
# UTILITY FUNCTIONS
#=============================================================================================

def compute_min_dist(mol_positions, *args):
    for pos1 in args:
        # Compute squared distances
        # Each row is an array of distances from a mol2 atom to all mol1 atoms
        distances2 = np.array([((pos1 - pos2)**2).sum(1) for pos2 in mol_positions])

        # Find closest atoms and their distance
        min_idx = np.unravel_index(distances2.argmin(), distances2.shape)
        try:
            min_dist = min(min_dist, np.sqrt(distances2[min_idx]))
        except UnboundLocalError:
            min_dist = np.sqrt(distances2[min_idx])
    return min_dist

def remove_overlap(mol_positions, *args, **kwargs):
    x = np.copy(mol_positions)
    sigma = kwargs.get('sigma', 1.0)
    min_distance = kwargs.get('min_distance', 1.0)

    # Try until we have a non-overlapping conformation w.r.t. all fixed molecules
    while compute_min_dist(x, *args) <= min_distance:
        # Compute center of geometry
        x0 = x.mean(0)

        # Randomize orientation of ligand.
        q = ModifiedHamiltonianExchange._generate_uniform_quaternion()
        Rq = ModifiedHamiltonianExchange._rotation_matrix_from_quaternion(q)
        x = ((Rq * np.matrix(x - x0).T).T + x0).A

        # Choose a random displacement vector and translate
        x += sigma * np.random.randn(3)

    return x

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
        opts.update(utils.validate_parameters(opts, self.DEFAULT_OPTIONS, check_unknown=False,
                                              process_units_str=True, float_to_int=True))
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

    def _generate_molecule(self, molecule_id):
        """Generate molecule and save it to mol2 in molecule['filepath']."""
        mol_descr = self._molecules[molecule_id]
        try:
            if 'name' in mol_descr:
                molecule = openmoltools.openeye.iupac_to_oemol(mol_descr['name'])
            elif 'smiles' in mol_descr:
                molecule = openmoltools.openeye.smiles_to_oemol(mol_descr['smiles'])
            molecule = openmoltools.openeye.get_charges(molecule, keep_confs=1)
        except ImportError as e:
            error_msg = ('requested molecule generation from name or smiles but '
                         'could not find OpenEye toolkit: ' + str(e))
            raise YamlParseError(error_msg)

        return molecule

    def _setup_molecules(self, *args):
        file_mol_ids = {mol_id for mol_id in args if 'filepath' in self._molecules[mol_id]}

        # Generate missing molecules with OpenEye
        oe_molecules = {mol_id: self._generate_molecule(mol_id)
                        for mol_id in args if mol_id not in file_mol_ids}

        # Check that non-generated molecules don't have overlapping atoms
        # TODO this check should be available even without OpenEye
        # TODO also there should be an option to solve the overlap in this case
        fixed_pos = {}  # positions of molecules from files
        if utils.is_openeye_installed():
            mol_id_list = list(file_mol_ids)
            positions = [utils.get_oe_mol_positions(utils.read_oe_molecule(
                         self._molecules[mol_id]['filepath'])) for mol_id in mol_id_list]
            for i in range(len(positions) - 1):
                posi = positions[i]
                if compute_min_dist(posi, *positions[i+1:]) < 0.1:
                    raise YamlParseError('The given molecules have overlapping atoms!')

            # Convert positions list to dictionary
            fixed_pos = {mol_id_list[i]: positions[i] for i in range(len(mol_id_list))}

        # Find and solve overlapping atoms in OpenEye generated molecules
        for mol_id, molecule in oe_molecules.items():
            molecule_pos = utils.get_oe_mol_positions(molecule)
            if fixed_pos:
                molecule_pos = remove_overlap(molecule_pos, *(fixed_pos.values()),
                                              min_distance=1.0, sigma=1.0)
                utils.set_oe_mol_positions(molecule, molecule_pos)

            # Update fixed positions for next round
            fixed_pos[mol_id] = molecule_pos

        # Save parametrized molecules
        for mol_id in args:
            mol_descr = self._molecules[mol_id]

            # Create output directory
            output_mol_dir = os.path.join(self._output_dir, self.SETUP_MOLECULES_DIR,
                                          mol_id)
            if not os.path.isdir(output_mol_dir):
                os.makedirs(output_mol_dir)

            # Write OpenEye generated molecules in mol2 files
            if mol_id in oe_molecules:
                # We update the 'filepath' key in the molecule description
                mol_descr['filepath'] = os.path.join(output_mol_dir, mol_id + '.mol2')

                # We set the residue name as the first three uppercase letters
                residue_name = re.sub('[^A-Za-z]+', '', mol_id.upper())
                openmoltools.openeye.molecule_to_mol2(molecule, mol_descr['filepath'],
                                                      residue_name=residue_name)

            # Parametrize the molecule with antechamber
            if mol_descr['parameters'] == 'antechamber':
                # Generate parameters
                input_mol_path = os.path.abspath(mol_descr['filepath'])
                with utils.temporary_cd(output_mol_dir):
                    openmoltools.amber.run_antechamber(mol_id, input_mol_path)

                # Save new parameters paths
                mol_descr['filepath'] = os.path.join(output_mol_dir, mol_id + '.gaff.mol2')
                mol_descr['parameters'] = os.path.join(output_mol_dir, mol_id + '.frcmod')

    def _setup_system(self, output_dir, components):
        # Create output directory
        output_sys_dir = os.path.join(self._output_dir, self.SETUP_SYSTEMS_DIR,
                                      output_dir)
        if not os.path.isdir(output_sys_dir):
            os.makedirs(output_sys_dir)

        # Setup molecules
        self._setup_molecules(components['receptor'], components['ligand'])

        # Identify components
        receptor = self._molecules[components['receptor']]
        ligand = self._molecules[components['ligand']]
        solvent = self._solvents[components['solvent']]

        # Create tleap script
        tleap = utils.TLeap()
        tleap.new_section('Load GAFF parameters')
        tleap.load_parameters('leaprc.gaff')

        # Check that AMBER force field is specified
        if not ('leaprc.' in receptor['parameters'] or 'leaprc.' in ligand['parameters']):
            tleap.load_parameters('leaprc.ff14SB')

        # Load receptor and ligand
        for group_name in ['receptor', 'ligand']:
            group = self._molecules[components[group_name]]
            tleap.new_section('Load ' + group_name)
            tleap.load_parameters(group['parameters'])
            tleap.load_group(name=group_name, file_path=group['filepath'])

        # Create complex
        tleap.new_section('Create complex')
        tleap.combine('complex', 'receptor', 'ligand')

        # Configure solvent
        if solvent['nonbondedMethod'] == 'NoCutoff':
            if 'gbsamodel' in solvent:  # GBSA implicit solvent
                tleap.new_section('Set GB radii to recommended values for OBC')
                tleap.add_commands('set default PBRadii mbondi2')
        else:  # explicit solvent
            tleap.new_section('Solvate systems')
            clearance = utils.process_unit_bearing_str(solvent['clearance'], unit.angstroms)
            clearance = float(clearance.value_in_unit(unit.angstroms))
            tleap.solvate(group='complex', water_model='TIP3PBOX', clearance=clearance)
            tleap.solvate(group='ligand', water_model='TIP3PBOX', clearance=clearance)

        # Check charge
        tleap.new_section('Check charge')
        tleap.add_commands('check complex', 'charge complex')

        # Save prmtop and inpcrd files
        tleap.new_section('Save prmtop and inpcrd files')
        tleap.save_group('complex', os.path.join(output_sys_dir, 'complex.prmtop'))
        tleap.save_group('complex', os.path.join(output_sys_dir, 'complex.pdb'))
        tleap.save_group('ligand', os.path.join(output_sys_dir, 'solvent.prmtop'))
        tleap.save_group('ligand', os.path.join(output_sys_dir, 'solvent.pdb'))

        # Save tleap script for reference
        tleap.export_script(os.path.join(output_sys_dir, 'leap.in'))

        # Run tleap!
        tleap.run()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
