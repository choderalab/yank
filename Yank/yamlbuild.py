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
import collections
import logging
logger = logging.getLogger(__name__)

import numpy as np
import openmoltools
from simtk import unit, openmm

import utils
from yank import Yank
from repex import ReplicaExchange, ThermodynamicState
from sampling import ModifiedHamiltonianExchange
from pipeline import find_components


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

def to_openmm_app(str):
    return getattr(openmm.app, str)

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
    EXPERIMENTS_DIR = 'experiments'

    DEFAULT_OPTIONS = {
        'verbose': False,
        'mpi': False,
        'resume': False,
        'output_dir': 'output/',
        'temperature': 298 * unit.kelvin,
        'pressure': 1 * unit.atmosphere,
        'constraints': openmm.app.HBonds,
        'hydrogenMass': 1 * unit.amu
    }

    SPECIAL_CONVERSIONS = {
        'constraints': to_openmm_app
    }

    CREATE_SYSTEM_FILTER = set(('nonbondedMethod', 'nonbondedCutoff', 'implicitSolvent',
                                'constraints', 'hydrogenMass'))

    @property
    def yank_options(self):
        return {opt: val for opt, val in self.options.items()
                if opt not in self.DEFAULT_OPTIONS}

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

        # Merge options and metadata
        temp_options = yaml_config.pop('options', {})
        temp_options.update(yaml_config.pop('metadata', {}))

        # Validate options and fill in defaults
        self.options = self.DEFAULT_OPTIONS.copy()
        self.options.update(self._validate_options(temp_options))

        # Store other fields, we don't raise an error if we cannot find any
        # since the YAML file could be used only to specify the options
        self._molecules = yaml_config.pop('molecules', {})
        # TODO - verify that filepath and name/smiles are not specified simultaneously

        self._solvents = yaml_config.get('solvents', {})
        # TODO verify that implicitSolvent and cutoff are specified at the same time

        # Check if there is a sequence of experiments or a single one
        try:
            self._experiments = {exp_name: utils.CombinatorialTree(yaml_config[exp_name])
                                 for exp_name in yaml_config['experiments']}
        except KeyError:
            single_exp = yaml_config.get('experiment', {})
            self._experiments = {'experiment': utils.CombinatorialTree(single_exp)}

    def build_experiment(self):
        """Build the Yank experiment."""
        output_dir = ''
        for exp_name, experiment in self._experiments.items():
            # If there is a sequence of experiments, create a subfolder for each one
            if len(self._experiments) > 1:
                output_dir = exp_name

            # Loop over all combinations
            for name, combination in experiment.named_combinations(separator='_', max_name_length=30):
                self._run_experiment(combination, os.path.join(output_dir, name))

    def _validate_options(self, options):
        """Return a dictionary with YamlBuilder and Yank options validated."""
        template_options = self.DEFAULT_OPTIONS.copy()
        template_options.update(Yank.default_parameters)
        template_options.update(ReplicaExchange.default_parameters)
        try:
            valid = utils.validate_parameters(options, template_options, check_unknown=True,
                                              process_units_str=True, float_to_int=True,
                                              special_conversions=self.SPECIAL_CONVERSIONS)
        except (TypeError, ValueError) as e:
            logger.error(str(e))
            raise YamlParseError(str(e))
        return valid

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

    def _setup_molecules(self, output_dir, *args):
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

            # Create output directory only if it's needed
            if not (mol_id in oe_molecules or mol_descr['parameters'] == 'antechamber'):
                continue
            mol_dir = os.path.join(output_dir, self.SETUP_MOLECULES_DIR, mol_id)
            if not os.path.isdir(mol_dir):
                os.makedirs(mol_dir)

            # Write OpenEye generated molecules in mol2 files
            if mol_id in oe_molecules:
                # We update the 'filepath' key in the molecule description
                mol_descr['filepath'] = os.path.join(mol_dir, mol_id + '.mol2')

                # We set the residue name as the first three uppercase letters
                residue_name = re.sub('[^A-Za-z]+', '', mol_id.upper())
                openmoltools.openeye.molecule_to_mol2(molecule, mol_descr['filepath'],
                                                      residue_name=residue_name)

            # Parametrize the molecule with antechamber
            if mol_descr['parameters'] == 'antechamber':
                # Generate parameters
                input_mol_path = os.path.abspath(mol_descr['filepath'])
                with utils.temporary_cd(mol_dir):
                    openmoltools.amber.run_antechamber(mol_id, input_mol_path)

                # Save new parameters paths
                mol_descr['filepath'] = os.path.join(mol_dir, mol_id + '.gaff.mol2')
                mol_descr['parameters'] = os.path.join(mol_dir, mol_id + '.frcmod')

    def _setup_system(self, components, output_dir):
        # Identify components
        receptor_id = components['receptor']
        ligand_id = components['ligand']
        solvent_id = components['solvent']
        receptor = self._molecules[receptor_id]
        ligand = self._molecules[ligand_id]
        solvent = self._solvents[solvent_id]

        # Setup molecules
        self._setup_molecules(output_dir, receptor_id, ligand_id)

        # Create output directory
        system_dir = '_'.join((receptor_id, ligand_id, solvent_id))
        system_dir = os.path.join(output_dir, self.SETUP_SYSTEMS_DIR, system_dir)
        if not os.path.isdir(system_dir):
            os.makedirs(system_dir)

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
            if 'implicitSolvent' in solvent:  # GBSA implicit solvent
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
        tleap.save_group('complex', os.path.join(system_dir, 'complex.prmtop'))
        tleap.save_group('complex', os.path.join(system_dir, 'complex.pdb'))
        tleap.save_group('ligand', os.path.join(system_dir, 'solvent.prmtop'))
        tleap.save_group('ligand', os.path.join(system_dir, 'solvent.pdb'))

        # Save tleap script for reference
        tleap.export_script(os.path.join(system_dir, 'leap.in'))

        # Run tleap!
        tleap.run()

        return system_dir

    def _run_experiment(self, experiment, experiment_dir):
        components = experiment['components']
        systems = {}  # systems[phase] is the System object associated with phase 'phase'
        positions = {}  # positions[phase] is a list of coordinates associated with phase 'phase'
        atom_indices = {}  # ligand_atoms[phase] is a list of ligand atom indices associated with phase 'phase'

        # Get and validate experiment options
        exp_opts = self.options.copy()
        exp_opts.update(self._validate_options(experiment.get('options', {})))

        # Store output paths
        results_dir = os.path.join(exp_opts['output_dir'], self.EXPERIMENTS_DIR, experiment_dir)
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        # Configure logger
        utils.config_root_logger(exp_opts['verbose'], os.path.join(results_dir, 'yaml.log'))

        # Setup complex and solvent systems
        systems_dir = self._setup_system(components, exp_opts['output_dir'])

        # Get ligand resname for alchemical atom selection
        ligand_res = utils.get_mol2_resname(self._molecules[components['ligand']]['filepath'])
        if ligand_res is None:
            ligand_res = 'MOL'

        # Solvent configuration
        solvent = self._solvents[components['solvent']]
        system_opts = {opt: to_openmm_app(solvent[opt])
                       for opt in self.CREATE_SYSTEM_FILTER if opt in solvent}
        system_opts.update({opt: exp_opts[opt] for opt in self.CREATE_SYSTEM_FILTER if opt in exp_opts})

        # Prepare phases of calculation.
        for phase_prefix in ['complex', 'solvent']:
            # Read Amber prmtop and create System object.
            prmtop_filename = os.path.join(systems_dir, '%s.prmtop' % phase_prefix)
            prmtop = openmm.app.AmberPrmtopFile(prmtop_filename)

            # Read Amber inpcrd and load positions.
            inpcrd_filename = os.path.join(systems_dir, '%s.inpcrd' % phase_prefix)
            inpcrd = openmm.app.AmberInpcrdFile(inpcrd_filename)

            # Determine if this will be an explicit or implicit solvent simulation.
            if inpcrd.boxVectors is not None:
                is_periodic = True
                phase_suffix = 'explicit'
            else:
                is_periodic = False
                phase_suffix = 'implicit'
            phase = '{}-{}'.format(phase_prefix, phase_suffix)

            # Check for solvent configuration inconsistencies
            # TODO: Check to make sure both prmtop and inpcrd agree on explicit/implicit.
            err_msg = ''
            if is_periodic:
                if 'implicitSolvent' in system_opts:
                    err_msg = 'Found periodic box in inpcrd file and implicitSolvent specified.'
                if system_opts['nonbondedMethod'] == openmm.app.NoCutoff:
                    err_msg = 'Found periodic box in inpcrd file but nonbondedMethod is NoCutoff'
            else:
                if system_opts['nonbondedMethod'] != openmm.app.NoCutoff:
                    err_msg = 'nonbondedMethod is NoCutoff but could not find periodic box in inpcrd.'
            if len(err_msg) != 0:
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            # Create system and update box vectors (if needed)
            systems[phase] = prmtop.createSystem(removeCMMotion=False, **system_opts)
            if is_periodic:
                systems[phase].setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)

            # Store numpy positions
            positions[phase] = inpcrd.getPositions(asNumpy=True)

            # Check to make sure number of atoms match between prmtop and inpcrd.
            prmtop_natoms = systems[phase].getNumParticles()
            inpcrd_natoms = positions[phase].shape[0]
            if prmtop_natoms != inpcrd_natoms:
                err_msg = "Atom number mismatch: prmtop {} has {} atoms; inpcrd {} has {} atoms.".format(
                    prmtop_filename, prmtop_natoms, inpcrd_filename, inpcrd_natoms)
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            # Find ligand atoms and receptor atoms
            ligand_dsl = 'resname ' + ligand_res  # MDTraj DSL that specifies ligand atoms
            atom_indices[phase] = find_components(prmtop.topology, ligand_dsl)

        # Create thermodynamic state
        thermodynamic_state = ThermodynamicState(temperature=exp_opts['temperature'],
                                                 pressure=exp_opts['pressure'])

        # Configure MPI, if requested
        if exp_opts['mpi']:
            from mpi4py import MPI
            MPI.COMM_WORLD.barrier()
            mpicomm = MPI.COMM_WORLD
        else:
            mpicomm = None

        # TODO configure platform and precision when they are fixed in Yank

        # Create and run simulation
        phases = systems.keys()
        yank = Yank(results_dir, mpicomm=mpicomm, **self.yank_options)
        yank.create(phases, systems, positions, atom_indices, thermodynamic_state)
        yank.run()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
