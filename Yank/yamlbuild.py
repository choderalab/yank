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
import sys
import glob
import copy
import yaml
import logging
logger = logging.getLogger(__name__)

import numpy as np
import openmoltools as omt
from simtk import unit, openmm
from simtk.openmm.app import PDBFile
from alchemy import AlchemicalState, AbsoluteAlchemicalFactory

import utils
import pipeline
from yank import Yank
from repex import ReplicaExchange, ThermodynamicState
from sampling import ModifiedHamiltonianExchange


#=============================================================================================
# UTILITY FUNCTIONS
#=============================================================================================

def compute_min_dist(mol_positions, *args):
    """Compute the minimum distance between a molecule and a set of other molecules.

    All the positions must be expressed in the same unit of measure.

    Parameters
    ----------
    mol_positions : numpy.ndarray
        An Nx3 array where, N is the number of atoms, containing the positions of
        the atoms of the molecule for which we want to compute the minimum distance
        from the others

    Other parameters
    ----------------
    args
        A series of numpy.ndarrays containing the positions of the atoms of the other
        molecules

    Returns
    -------
    min_dist : float
        The minimum distance between mol_positions and the other set of positions

    """
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

def compute_dist_bound(mol_positions, *args):
    """Compute minimum and maximum distances between a molecule and a set of
    other molecules.

    All the positions must be expressed in the same unit of measure.

    Parameters
    ----------
    mol_positions : numpy.ndarray
        An Nx3 array where, N is the number of atoms, containing the positions of
        the atoms of the molecule for which we want to compute the minimum distance
        from the others

    Other parameters
    ----------------
    args
        A series of numpy.ndarrays containing the positions of the atoms of the other
        molecules

    Returns
    -------
    min_dist : float
        The minimum distance between mol_positions and the atoms of the other positions
    max_dist : float
        The maximum distance between mol_positions and the atoms of the other positions

    Examples
    --------
    >>> mol1_pos = np.array([[-1, -1, -1], [1, 1, 1]], np.float)
    >>> mol2_pos = np.array([[2, 2, 2], [2, 4, 5]], np.float)  # determine min dist
    >>> mol3_pos = np.array([[3, 3, 3], [3, 4, 5]], np.float)  # determine max dist
    >>> min_dist, max_dist = compute_dist_bound(mol1_pos, mol2_pos, mol3_pos)
    >>> min_dist == np.linalg.norm(mol1_pos[1] - mol2_pos[0])
    True
    >>> max_dist == np.linalg.norm(mol1_pos[1] - mol3_pos[1])
    True

    """
    min_dist = None

    for arg_pos in args:
        # Compute squared distances of all atoms. Each row is an array
        # of distances from an atom in arg_pos to all the atoms in arg_pos
        distances2 = np.array([((mol_positions - atom)**2).sum(1) for atom in arg_pos])

        # Find distances of each arg_pos atom to mol_positions
        distances2 = np.amin(distances2, axis=1)

        # Find closest and distant atom
        if min_dist is None:
            min_dist = np.sqrt(distances2.min())
            max_dist = np.sqrt(distances2.max())
        else:
            min_dist = min(min_dist, np.sqrt(distances2.min()))
            max_dist = max(max_dist, np.sqrt(distances2.max()))

    return min_dist, max_dist

def remove_overlap(mol_positions, *args, **kwargs):
    """Remove any eventual overlap between a molecule and a set of others.

    The method both randomly shifts and rotates the molecule (when overlapping atoms
    are detected) until it does not clash with any other given molecule anymore. All
    the others are kept fixed.

    All the positions must be expressed in the same unit of measure.

    Parameters
    ----------
    mol_positions : numpy.ndarray
        An Nx3 array where, N is the number of atoms, containing the positions of
        the atoms of the molecule that we want to not clash with the others.
    min_distance : float
        The minimum distance accepted to consider the molecule not clashing with
        the others. Must be in the same unit of measure of the positions.
    sigma : float
        The maximum displacement for a single step. Must be in the same unit of
        measure of the positions.

    Other parameters
    ----------------
    args
        A series of numpy.ndarrays containing the positions of the atoms of the
        molecules that are kept fixed.

    Returns
    -------
    x : numpy.ndarray
        Positions of the atoms of the given molecules that do not clash.

    """
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

def pack_transformation(mol1_pos, mol2_pos, min_distance, max_distance):
    """Compute an affine transformation that solve clashes and fit mol2 in the box.

    The method randomly shifts and rotates mol2 until all its atoms are within
    min_distance and max_distance from mol1. The position of mol1 is kept fixed.
    Every 200 failed iterations, the algorithm increases max_distance by 50%. It
    raise an exception after 1000 iterations.

    All the positions must be expressed in the same unit of measure.

    Parameters
    ----------
    mol1_pos : numpy.ndarray
        An Nx3 array where, N is the number of atoms, containing the positions of
        the atoms of the molecule that will be kept fixed.
    mol2_pos : numpy.ndarray
        An Nx3 array where, N is the number of atoms, containing the positions of
        the atoms of the molecule that will be eventually moved.
    min_distance : float
        The minimum distance accepted to consider mol2 not clashing with mol1. It
        must be in the same unit of measure of the positions.
    max_distance : float
        The maximum distance from mol1 to consider mol2 within the box. It must
        be in the same unit of measure of the positions.

    Returns
    -------
    transformation : numpy.ndarray
        A 4x4 ndarray representing the affine transformation that translate and
        rotate mol2.

    """
    translation = None  # we'll use this to check if we made changes to mol2_pos
    transformation = np.identity(4)

    # Compute center of geometry
    x0 = mol2_pos.mean(0)

    # Try until we have a non-overlapping conformation w.r.t. all fixed molecules
    i = 0
    min_dist, max_dist = compute_dist_bound(mol1_pos, mol2_pos)
    while min_dist < min_distance or max_distance <= max_dist:
        # Select random atom of fixed molecule and use it to propose new x0 position
        mol1_atom_idx = np.random.random_integers(0, len(mol1_pos) - 1)
        translation = mol1_pos[mol1_atom_idx] + max_distance * np.random.randn(3) - x0

        # Generate random rotation matrix
        q = ModifiedHamiltonianExchange._generate_uniform_quaternion()
        Rq = ModifiedHamiltonianExchange._rotation_matrix_from_quaternion(q)

        # Apply random transformation and test
        x = ((Rq * np.matrix(mol2_pos - x0).T).T + x0).A + translation
        min_dist, max_dist = compute_dist_bound(mol1_pos, x)

        # Check n iterations
        i += 1
        if i % 200 == 0:
            max_distance *= 1.5
        if i >= 1000:
            err_msg = 'Cannot fit mol2 into solvation box!'
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    # Generate 4x4 affine transformation in molecule reference frame
    if translation is not None:
        transl_to_origin, transl_to_x0, rot_transl_matrix = (np.identity(4) for _ in range(3))
        transl_to_origin[:3, 3] = -x0  # translate the molecule from x0 to origin
        rot_transl_matrix[:3, :3] = Rq  # rotate molecule in origin
        rot_transl_matrix[:3, 3] = translation  # translate molecule
        transl_to_x0[:3, 3] = x0  # translate the molecule from origin to x0
        transformation = transl_to_x0.dot(rot_transl_matrix.dot(transl_to_origin))

    return transformation

def pull_close(fixed_mol_pos, translated_mol_pos, min_bound, max_bound):
    """Heuristic algorithm to quickly translate the ligand close to the receptor.

    The distance of the ligand from the receptor here is defined as the shortest
    Euclidean distance between an atom of the ligand and one of the receptor.
    The molecules positions will not be modified if the ligand is already at a
    distance in the interval [min_bound, max_bound].

    Parameters
    ----------
    fixed_mol_pos : numpy.array
        The positions of the molecule to keep fixed as a Nx3 array.
    translated_mol_pos : numpy.array
        The positions of the molecule to translate as a Nx3 array.
    min_bound : float
        Minimum distance from the receptor to the ligand. This should be high
        enough for the ligand to not overlap the receptor atoms at the beginning
        of the simulation.
    max_bound : float
        Maximum distance from the receptor to the ligand. This should be short
        enough to make the ligand and the receptor interact since the beginning
        of the simulation.

    Returns
    -------
    translation : numpy.array
        A 1x3 array containing the translation vector to apply to translated_mol_pos
        to move the molecule at a distance between min_bound and max_bound from
        fixed_mol_pos.

    """

    goal_distance = (min_bound + max_bound) / 2
    trans_pos = copy.deepcopy(translated_mol_pos)  # positions that we can modify

    # Find translation
    final_translation = np.zeros(3)
    while True:

        # Compute squared distances between all atoms
        # Each row is an array of distances from a translated atom to all fixed atoms
        # We don't need to apply square root to everything
        distances2 = np.array([((fixed_mol_pos - pos)**2).sum(1) for pos in trans_pos])

        # Find closest atoms and their distance
        min_idx = np.unravel_index(distances2.argmin(), distances2.shape)
        min_dist = np.sqrt(distances2[min_idx])

        # If closest atom is between boundaries translate ligand
        if min_bound <= min_dist <= max_bound:
            break

        # Compute unit vector that connects receptor and ligand atom
        if min_dist != 0:
            direction = fixed_mol_pos[min_idx[1]] - trans_pos[min_idx[0]]
        else:  # any deterministic direction
            direction = np.array([1, 1, 1])
        direction = direction / np.sqrt((direction**2).sum())  # normalize

        if max_bound < min_dist:  # the atom is far away
            translation = (min_dist - goal_distance) * direction
            trans_pos += translation
            final_translation += translation
        elif min_dist < min_bound:  # the two molecules overlap
            max_dist = np.sqrt(distances2.max())
            translation = (max_dist + goal_distance) * direction
            trans_pos += translation
            final_translation += translation

    return final_translation


def strip_protons(input_file_path, output_file_path):
    """Remove all hydrogens from PDB file and save the result."""
    output_file = open(output_file_path, 'w')
    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            if not (line[:6] == 'ATOM  ' and (line[12] == 'H' or line[13] == 'H')):
                output_file.write(line)
    output_file.close()


def to_openmm_app(str):
    """Converter function to be used with validate_parameters()."""
    return getattr(openmm.app, str)

#=============================================================================================
# UTILITY CLASSES
#=============================================================================================

class YamlParseError(Exception):
    """Represent errors occurring during parsing of Yank YAML file."""
    def __init__(self, message):
        super(YamlParseError, self).__init__(message)
        logger.error(message)

class YankDumper(yaml.Dumper):
    """PyYAML Dumper that always return sequences in flow style and maps in block style."""
    def represent_sequence(self, tag, sequence, flow_style=None):
        return yaml.Dumper.represent_sequence(self, tag, sequence, flow_style=True)

    def represent_mapping(self, tag, mapping, flow_style=None):
        return yaml.Dumper.represent_mapping(self, tag, mapping, flow_style=False)

class SetupDatabase:
    """Provide utility functions to set up systems and molecules.

    The object allows to access molecules, systems and solvents by 'id' and takes
    care of parametrizing molecules and creating the AMBER prmtop and inpcrd files
    describing systems.

    Attributes
    ----------
    setup_dir : str
        Path to the main setup directory. Changing this means changing the database.
    molecules : dict
        Dictionary of molecule_id -> molecule YAML description.
    solvents : dict
        Dictionary of solvent_id -> molecule YAML description.

    """

    SYSTEMS_DIR = 'systems'
    MOLECULES_DIR = 'molecules'
    CLASH_THRESHOLD = 1.5  # distance in Angstroms to consider two atoms clashing

    def __init__(self, setup_dir, molecules=None, solvents=None):
        """Initialize the database.

        Parameters
        ----------
        setup_dir : str
            Path to the main setup directory.
        molecules : dict
            YAML description of the molecules (default is None).
        solvents : dict
            YAML description of the solvents (default is None).

        """
        self.setup_dir = setup_dir
        self.molecules = molecules
        self.solvents = solvents

        # Private attributes
        self._pos_cache = {}  # cache positions of molecules
        self._processed_mols = set()  # keep track of parametrized molecules

    def get_molecule_dir(self, molecule_id):
        """Return the directory where the parameter files are stored.

        Parameters
        ----------
        molecule_id : str
            The ID of the molecule.

        Returns
        -------
        str
            The path to the molecule directory.
        """
        return os.path.join(self.setup_dir, self.MOLECULES_DIR, molecule_id)

    def get_system_dir(self, receptor, ligand, solvent):
        """Return the directory where the prmtop and inpcrd files are stored.

        The system is uniquely defined by the ids of the receptor, ligand and
        solvent.

        Parameters
        ----------
        receptor : str
            The id of the receptor.
        ligand : str
            The id of the ligand.
        solvent : str
            The id of the solvent.

        Returns
        -------
        system_dir : str
            Path to the directory containing the AMBER system files.

        """
        system_dir = '_'.join((receptor, ligand, solvent))
        system_dir = os.path.join(self.setup_dir, self.SYSTEMS_DIR, system_dir)
        return system_dir

    def is_molecule_setup(self, molecule_id):
        """Check whether the molecule has been processed previously.

        The molecule must be set up if it needs to be parametrize by antechamber
        (and the gaff.mol2 and frcmod files do not exist), if the molecule must be
        generated by OpenEye, or if it needs to be extracted by a multi-molecule file.

        An example to clarify the difference between the two return values: a protein
        in a single-frame pdb does not have to be processed (since it does not go through
        antechamber) thus the function will return is_setup=True and is_processed=False.

        Parameters
        ----------
        molecule_id : str
            The id of the molecule.

        Returns
        -------
        is_setup : bool
            True if the molecule's parameter files have been specified by the user
            or if they have been generated by SetupDatabase.
        is_processed : bool
            True if parameter files have been generated previously by SetupDatabase
            (i.e. if the parameter files were not manually specified by the user).

        """

        # The only way to check if we processed the molecule in the current run is
        # through self._processed_mols as 'parameters' will be changed after setup
        if molecule_id in self._processed_mols:
            return True, True

        # Some convenience variables
        molecule_descr = self.molecules[molecule_id]
        molecule_dir = self.get_molecule_dir(molecule_id)
        molecule_id_path = os.path.join(molecule_dir, molecule_id)
        try:
            extension = os.path.splitext(molecule_descr['filepath'])[1]
        except KeyError:
            extension = None

        # The following checks must be performed in reverse order w.r.t. how they
        # are executed in _setup_molecules()

        files_to_check = {}

        # If the molecule must go through antechamber we search for its output
        if (molecule_descr['parameters'] == 'antechamber' or
                    molecule_descr['parameters'] == 'openeye:am1bcc-gaff'):
            files_to_check['filepath'] = molecule_id_path + '.gaff.mol2'
            files_to_check['parameters'] = molecule_id_path + '.frcmod'

        # If the molecule must be generated by OpenEye, a mol2 should have been created
        elif extension is None or extension == '.smiles' or extension == '.csv':
            files_to_check['filepath'] = molecule_id_path + '.mol2'

        # If we have to strip the protons off a PDB, a new PDB should have been created
        elif 'strip_protons' in molecule_descr and molecule_descr['strip_protons']:
            files_to_check['filepath'] = molecule_id_path + '.pdb'

        # If a single structure must be extracted we search for output
        elif 'select' in molecule_descr:
            files_to_check['filepath'] = molecule_id_path + extension

        # Check if this needed to be processed at all
        if not files_to_check:
            return True, False

        # Check if all output files exist
        all_file_exist = True
        for descr_key, file_path in files_to_check.items():
            all_file_exist &= os.path.isfile(file_path) and os.path.getsize(file_path) > 0
            if all_file_exist:  # Make sure internal description is correct
                molecule_descr[descr_key] = file_path
                extension = os.path.splitext(molecule_descr['filepath'])[1]
                if extension == '.mol2':
                    molecule_descr['net_charge'] = utils.get_mol2_net_charge(molecule_descr['filepath'])

        return all_file_exist, all_file_exist

    def is_system_setup(self, receptor, ligand, solvent):
        """Check whether the system has been already set up.

        Parameters
        ----------
        receptor : str
            The id of the receptor.
        ligand : str
            The id of the ligand.
        solvent : str
            The id of the solvent.

        Returns
        -------
        system_dir : bool
            True if the system has been already set up.

        """
        system_dir = self.get_system_dir(receptor, ligand, solvent)
        is_setup = (os.path.exists(os.path.join(system_dir, 'complex.prmtop')) and
                    os.path.exists(os.path.join(system_dir, 'complex.inpcrd')) and
                    os.path.exists(os.path.join(system_dir, 'solvent.prmtop')) and
                    os.path.exists(os.path.join(system_dir, 'solvent.inpcrd')))
        return is_setup

    def get_system(self, components, pack=True):
        """Make sure that the system files are set up and return the system folder.

        If necessary, create the prmtop and inpcrd files from the given components.
        The system files are generated with tleap. If no molecule specifies a general
        force field, leaprc.ff14SB is loaded.

        Parameters
        ----------
        setup_dir : str
            The path to the main setup directory specified by the user in the YAML options
        components : dict
            A dictionary containing the keys 'receptor', 'ligand' and 'solvent' with the ids
            of molecules and solvents
        pack : bool
            If True and the ligand is far away from the protein or closer than the clashing
            threshold, this try to find a better position (default is True).

        Returns
        -------
        system_dir : str
            The path to the directory containing the prmtop and inpcrd files

        """

        # Identify system
        receptor_id = components['receptor']
        ligand_id = components['ligand']
        solvent_id = components['solvent']
        system_dir = self.get_system_dir(receptor_id, ligand_id, solvent_id)

        # Check if system has been already processed
        if self.is_system_setup(receptor_id, ligand_id, solvent_id):
            return system_dir

        # We still need to check if system_dir exists because the set up may
        # have been interrupted
        if not os.path.exists(system_dir):
            os.makedirs(system_dir)

        # Setup molecules
        self._setup_molecules(receptor_id, ligand_id)

        # Identify components
        receptor = self.molecules[receptor_id]
        ligand = self.molecules[ligand_id]
        solvent = self.solvents[solvent_id]

        # Create tleap script
        tleap = utils.TLeap()
        tleap.new_section('Load GAFF parameters')
        tleap.load_parameters('leaprc.gaff')

        # Check that AMBER force field is specified
        if 'leaprc.' in receptor['parameters']:
            amber_ff = receptor['parameters']
        elif 'leaprc.' in ligand['parameters']:
            amber_ff = ligand['parameters']
        else:
            amber_ff = 'leaprc.ff14SB'
        tleap.load_parameters(amber_ff)

        # In ff12SB and ff14SB ions parameters must be loaded separately
        if (('positive_ion' in solvent or 'negative_ion' in solvent) and
                (amber_ff == 'leaprc.ff12SB' or amber_ff == 'leaprc.ff14SB')):
            tleap.add_commands('loadAmberParams frcmod.ionsjc_tip3p')

        # Load receptor and ligand
        for group_name in ['receptor', 'ligand']:
            group = self.molecules[components[group_name]]
            tleap.new_section('Load ' + group_name)
            tleap.load_parameters(group['parameters'])
            tleap.load_group(name=group_name, file_path=group['filepath'])

        # Check that molecules don't have clashing atoms. Also, if the ligand
        # is too far away from the molecule we want to pull it closer
        # TODO this check should be available even without OpenEye
        if pack and utils.is_openeye_installed():
            mol_id_list = [receptor_id, ligand_id]
            positions = [0 for _ in mol_id_list]
            for i, mol_id in enumerate(mol_id_list):
                if mol_id not in self._pos_cache:
                    self._pos_cache[mol_id] = utils.get_oe_mol_positions(
                            utils.read_oe_molecule(self.molecules[mol_id]['filepath']))
                positions[i] = self._pos_cache[mol_id]

            # Find the transformation
            try:
                max_dist = solvent['clearance'].value_in_unit(unit.angstrom) / 1.5
            except KeyError:
                max_dist = 10.0
            transformation = pack_transformation(positions[0], positions[1],
                                                 self.CLASH_THRESHOLD, max_dist)
            if (transformation != np.identity(4)).any():
                logger.warning('Changing starting ligand positions.')
                tleap.transform('ligand', transformation)

        # Create complex
        tleap.new_section('Create complex')
        tleap.combine('complex', 'receptor', 'ligand')

        # Configure solvent
        if solvent['nonbonded_method'] == openmm.app.NoCutoff:
            if 'implicit_solvent' in solvent:  # GBSA implicit solvent
                tleap.new_section('Set GB radii to recommended values for OBC')
                tleap.add_commands('set default PBRadii mbondi2')
        else:  # explicit solvent
            tleap.new_section('Solvate systems')

            # Add ligand-neutralizing ions
            if 'net_charge' in ligand and ligand['net_charge'] != 0:
                net_charge = ligand['net_charge']

                # Check that solvent is configured for charged ligand
                if not ('positive_ion' in solvent and 'negative_ion' in solvent):
                    err_msg = ('Found charged ligand but no indications for ions in '
                               'solvent {}').format(solvent_id)
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)

                # Add ions to the system
                if net_charge > 0:
                    ion = solvent['positive_ion']
                else:
                    ion = solvent['negative_ion']
                tleap.add_ions(unit='complex', ion=ion, num_ions=abs(net_charge))

            # Neutralizing solvation box
            if 'positive_ion' in solvent:
                tleap.add_ions(unit='complex', ion=solvent['positive_ion'])
                tleap.add_ions(unit='ligand', ion=solvent['positive_ion'])
            if 'negative_ion' in solvent:
                tleap.add_ions(unit='complex', ion=solvent['negative_ion'])
                tleap.add_ions(unit='ligand', ion=solvent['negative_ion'])

            clearance = float(solvent['clearance'].value_in_unit(unit.angstroms))
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

    def _generate_molecule(self, molecule_id):
        """Generate molecule using the OpenEye toolkit from name or smiles.

        The molecule is charged with OpenEye's recommended AM1BCC charge
        selection scheme and it is saved into the OpenEye molecules cache.

        Parameters
        ----------
        molecule_id : str
            The id of the molecule as given in the YAML script

        Returns
        -------
        molecule : OEMol
            The generated molecule.

        """
        mol_descr = self.molecules[molecule_id]  # molecule description
        try:
            if 'name' in mol_descr:
                molecule = omt.openeye.iupac_to_oemol(mol_descr['name'])
            elif 'smiles' in mol_descr:
                molecule = omt.openeye.smiles_to_oemol(mol_descr['smiles'])
            molecule = omt.openeye.get_charges(molecule, keep_confs=1)
        except ImportError as e:
            error_msg = ('requested molecule generation from name or smiles but '
                         'could not find OpenEye toolkit: ' + str(e))
            raise YamlParseError(error_msg)

        return molecule

    def _setup_molecules(self, *args):
        """Set up the files needed to generate the system for all the molecules.

        If OpenEye tools are installed, this generate the molecules when the source is
        not a file. If two (or more) molecules generated by OpenEye have overlapping
        atoms, the molecules are randomly shifted and rotated until the clash is resolved.
        With the OpenEye toolkit installed, we also perform a sanity check to verify that
        the molecules from files do not have overlapping atoms. An Exception is raised if
        this is not the case.

        If the Schrodinger's suite is install, this can enumerate tautomeric and protonation
        states with epik when requested.

        This also parametrize the molecule with antechamber when requested.

        Other parameters
        ----------------
        args
            All the molecules ids that compose the system. These molecules are the only
            ones considered when trying to resolve the overlapping atoms.

        """

        for mol_id in args:
            net_charge = None  # used by antechamber
            mol_descr = self.molecules[mol_id]

            # Have we already processed this molecule? Do we have to do it at all?
            # We don't want to create the output folder if we don't need to
            if self.is_molecule_setup(mol_id)[0]:
                continue

            # Create output directory if it doesn't exist
            mol_dir = self.get_molecule_dir(mol_id)
            if not os.path.exists(mol_dir):
                os.makedirs(mol_dir)

            try:
                extension = os.path.splitext(mol_descr['filepath'])[1]
            except KeyError:
                extension = None

            # Extract single model if this is a multi-model file
            if 'select' in mol_descr:
                model_idx = mol_descr['select']
                single_file_path = os.path.join(mol_dir, mol_id + extension)
                if extension == '.pdb':
                    # Create single-model PDB file
                    pdb_file = PDBFile(mol_descr['filepath'])
                    with open(single_file_path, 'w') as f:
                        PDBFile.writeHeader(pdb_file.topology, file=f)
                        PDBFile.writeModel(pdb_file.topology, pdb_file.getPositions(frame=model_idx), file=f)
                    # We might as well already cache the positions
                    self._pos_cache[mol_id] = pdb_file.getPositions(asNumpy=True, frame=model_idx) / unit.angstrom
                elif extension == '.smiles' or extension == '.csv':
                    # Extract the correct line and save it in a new file
                    # We ignore blank-lines with filter() when counting models
                    with open(mol_descr['filepath'], 'r') as smiles_file:
                        smiles_lines = filter(bool, smiles_file.readlines())
                    with open(single_file_path, 'w') as f:
                        f.write(smiles_lines[model_idx])
                elif extension == '.mol2' or extension == '.sdf':
                    if not utils.is_openeye_installed():
                        raise RuntimeError('Cannot support {} files selection without OpenEye'.format(
                                extension[1:]))
                    oe_molecule = utils.read_oe_molecule(mol_descr['filepath'], conformer_idx=model_idx)
                    utils.write_oe_molecule(oe_molecule, single_file_path)
                else:
                    raise RuntimeError('Model selection is not supported for {} files'.format(extension[1:]))

                # Save new file path
                mol_descr['filepath'] = single_file_path

            # Strip off protons if required
            if 'strip_protons' in mol_descr and mol_descr['strip_protons']:
                if extension != '.pdb':
                    raise RuntimeError('Cannot strip protons off {} files.'.format(extension[1:]))
                output_file_path = os.path.join(mol_dir, mol_id + '.pdb')
                strip_protons(mol_descr['filepath'], output_file_path)
                mol_descr['filepath'] = output_file_path

            # Generate missing molecules with OpenEye. At the end of parametrization
            # we update the 'filepath' key also for OpenEye-generated molecules so
            # we don't need to keep track of the molecules we have already generated
            if extension is None or extension == '.smiles' or extension == '.csv':
                if not utils.is_openeye_installed():
                    raise RuntimeError('Cannot support {} files without OpenEye'.format(extension[1:]))

                # Retrieve the first SMILES string, we take the last column
                if extension is not None:
                    with open(mol_descr['filepath'], 'r') as smiles_file:
                        smiles_str = smiles_file.readline().strip().split(',')[-1]
                    mol_descr['smiles'] = smiles_str  # prepare for _generate_molecule()

                # Generate molecule and cache atom positions
                oe_molecule = self._generate_molecule(mol_id)
                self._pos_cache[mol_id] = utils.get_oe_mol_positions(oe_molecule)

                # Write OpenEye generated molecules in mol2 files
                # We update the 'filepath' key in the molecule description
                mol_descr['filepath'] = os.path.join(mol_dir, mol_id + '.mol2')

                # We set the residue name as the first three uppercase letters of mol_id
                residue_name = re.sub('[^A-Za-z]+', '', mol_id.upper())[:3]
                omt.openeye.molecule_to_mol2(oe_molecule, mol_descr['filepath'],
                                             residue_name=residue_name)

            # Enumerate protonation states with epik
            if 'epik' in mol_descr:
                epik_idx = mol_descr['epik']
                epik_base_path = os.path.join(mol_dir, mol_id + '-epik.')
                epik_mae_file = epik_base_path + 'mae'
                epik_mol2_file = epik_base_path + 'mol2'
                epik_sdf_file = epik_base_path + 'sdf'

                # Run epik and convert from maestro to both mol2 and sdf
                # to not lose neither the penalties nor the residue name
                omt.schrodinger.run_epik(mol_descr['filepath'], epik_mae_file, tautomerize=True,
                                         extract_range=epik_idx)
                omt.schrodinger.run_structconvert(epik_mae_file, epik_sdf_file)
                omt.schrodinger.run_structconvert(epik_mae_file, epik_mol2_file)

                # Save new net charge from the i_epik_Tot_Q property
                net_charge = int(omt.schrodinger.run_proplister(epik_sdf_file)[0]['i_epik_Tot_Q'])

                # Keep filepath consistent
                mol_descr['filepath'] = epik_mol2_file

            # Antechamber does not support sdf files so we need to convert them
            extension = os.path.splitext(mol_descr['filepath'])[1]
            if extension == '.sdf':
                if not utils.is_openeye_installed():
                    raise RuntimeError('Cannot support sdf files without OpenEye')
                mol2_file_path = os.path.join(mol_dir, mol_id + '.mol2')
                oe_molecule = utils.read_oe_molecule(mol_descr['filepath'])

                # We set the residue name as the first three uppercase letters of mol_id
                residue_name = re.sub('[^A-Za-z]+', '', mol_id.upper())[:3]
                omt.openeye.molecule_to_mol2(oe_molecule, mol2_file_path,
                                             residue_name=residue_name)

                # Update filepath information
                mol_descr['filepath'] = mol2_file_path

            # Parametrize the molecule with antechamber
            if (mol_descr['parameters'] == 'antechamber' or
                mol_descr['parameters'] == 'openeye:am1bcc-gaff'):

                # Generate charges with OpenEye requested
                if mol_descr['parameters'] == 'openeye:am1bcc-gaff':
                    if not utils.is_openeye_installed():
                        err_msg = ('Cannot find OpenEye toolkit to compute charges '
                                   'for molecule {}').format(mol_id)
                        logger.error(err_msg)
                        raise RuntimeError(err_msg)
                    mol2_file_path = os.path.join(mol_dir, mol_id + '.mol2')
                    oe_molecule = utils.read_oe_molecule(mol_descr['filepath'])

                    # Setting keep_confs = None keeps the original conformation
                    oe_molecule = omt.openeye.get_charges(oe_molecule, keep_confs=None)
                    residue_name = utils.get_mol2_resname(mol_descr['filepath'])
                    omt.openeye.molecule_to_mol2(oe_molecule, mol2_file_path,
                                                 residue_name=residue_name)

                    charge_method = None  # antechamber read charges from mol2
                    net_charge = None  # we don't need Epik's net charge
                    mol_descr['filepath'] = mol2_file_path
                else:  # use antechamber and sqm
                    charge_method = 'bcc'

                # Generate parameters
                input_mol_path = os.path.abspath(mol_descr['filepath'])
                with utils.temporary_cd(mol_dir):
                    omt.amber.run_antechamber(mol_id, input_mol_path,
                                              charge_method=charge_method,
                                              net_charge=net_charge)

                # Save new parameters paths
                mol_descr['filepath'] = os.path.join(mol_dir, mol_id + '.gaff.mol2')
                mol_descr['parameters'] = os.path.join(mol_dir, mol_id + '.frcmod')

            # Determine small molecule net charge
            extension = os.path.splitext(mol_descr['filepath'])[1]
            if extension == '.mol2':
                if net_charge is not None:
                    mol_descr['net_charge'] = net_charge
                else:
                    mol_descr['net_charge'] = utils.get_mol2_net_charge(mol_descr['filepath'])

            # Keep track of processed molecule
            self._processed_mols.add(mol_id)

#=============================================================================================
# BUILDER CLASS
#=============================================================================================

class YamlBuilder:
    """Parse YAML configuration file and build the experiment.

    The relative paths indicated in the script are assumed to be relative to
    the script directory. However, if YamlBuilder is initiated with a string
    rather than a file path, the paths will be relative to the user's working
    directory.

    The class firstly perform a dry run to check if this is going to overwrite
    some files and raises an exception if it finds already existing output folders
    unless the options resume_setup or resume_simulation are True.

    Properties
    ----------
    yank_options : dict
        The options specified in the parsed YAML file that will be passed to Yank.
        These are not the full range of options specified in the script since some
        of them are used to configure YamlBuilder and not the Yank object.

    Examples
    --------
    >>> import textwrap
    >>> setup_dir = utils.get_data_filename(os.path.join('..', 'examples',
    ...                                     'p-xylene-implicit', 'setup'))
    >>> pxylene_path = os.path.join(setup_dir, 'ligand.tripos.mol2')
    >>> lysozyme_path = os.path.join(setup_dir, 'receptor.pdbfixer.pdb')
    >>> with utils.temporary_directory() as tmp_dir:
    ...     yaml_content = '''
    ...     ---
    ...     options:
    ...       number_of_iterations: 1
    ...       output_dir: {}
    ...     molecules:
    ...       T4lysozyme:
    ...         filepath: {}
    ...         parameters: oldff/leaprc.ff99SBildn
    ...       p-xylene:
    ...         filepath: {}
    ...         parameters: antechamber
    ...     solvents:
    ...       vacuum:
    ...         nonbonded_method: NoCutoff
    ...     protocols:
    ...       absolute-binding:
    ...         phases:
    ...           complex:
    ...             alchemical_path:
    ...               lambda_electrostatics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
    ...               lambda_sterics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
    ...           solvent:
    ...             alchemical_path:
    ...               lambda_electrostatics: [1.0, 0.8, 0.6, 0.3, 0.0]
    ...               lambda_sterics: [1.0, 0.8, 0.6, 0.3, 0.0]
    ...     experiments:
    ...       components:
    ...         receptor: T4lysozyme
    ...         ligand: p-xylene
    ...         solvent: vacuum
    ...       protocol: absolute-binding
    ...     '''.format(tmp_dir, lysozyme_path, pxylene_path)
    >>> yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
    >>> yaml_builder.build_experiment()

    """

    DEFAULT_OPTIONS = {
        'verbose': False,
        'mpi': False,
        'resume_setup': False,
        'resume_simulation': False,
        'output_dir': 'output',
        'setup_dir': 'setup',
        'experiments_dir': 'experiments',
        'pack': True,
        'temperature': 298 * unit.kelvin,
        'pressure': 1 * unit.atmosphere,
        'constraints': openmm.app.HBonds,
        'hydrogen_mass': 1 * unit.amu
    }

    def _expand_molecules(self, yaml_content):
        """Expand combinatorial molecules.

        Generate new YAML content with no combinatorial molecules. The new content
        is identical to the old one but combinatorial molecules are substituted by
        the description of all the non-combinatorial molecules that they generate.
        Moreover, the components of experiments that use combinatorial molecules
        are resolved.

        Parameters
        ----------
        yaml_content : dict
            The YAML content as returned by yaml.load().

        Returns
        -------
        expanded_content : dict
            The new YAML content with combinatorial molecules expanded.

        Examples
        --------
        >>> import textwrap
        >>> yaml_content = '''
        ... ---
        ... molecules:
        ...     rec:
        ...         filepath: [conf1.pdb, conf2.pdb]
        ...         parameters: oldff/leaprc.ff99SBildn
        ...     lig:
        ...         name: iupacname
        ...         parameters: antechamber
        ... solvents:
        ...     solv1:
        ...         nonbonded_method: NoCutoff
        ... protocols:
        ...     absolute-binding:
        ...         phases:
        ...             complex:
        ...                 alchemical_path:
        ...                     lambda_electrostatics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
        ...                     lambda_sterics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
        ...             solvent:
        ...                 alchemical_path:
        ...                     lambda_electrostatics: [1.0, 0.8, 0.6, 0.3, 0.0]
        ...                     lambda_sterics: [1.0, 0.8, 0.6, 0.3, 0.0]
        ... experiments:
        ...     components:
        ...         receptor: rec
        ...         ligand: lig
        ...         solvent: solv1
        ...     protocol: absolute-binding
        ... '''
        >>> expected_content = '''
        ... ---
        ... molecules:
        ...     rec_conf1pdb:
        ...         filepath: conf1.pdb
        ...         parameters: oldff/leaprc.ff99SBildn
        ...     rec_conf2pdb:
        ...         filepath: conf2.pdb
        ...         parameters: oldff/leaprc.ff99SBildn
        ...     lig:
        ...         name: iupacname
        ...         parameters: antechamber
        ... solvents:
        ...     solv1:
        ...         nonbonded_method: NoCutoff
        ... protocols:
        ...     absolute-binding:
        ...         phases:
        ...             complex:
        ...                 alchemical_path:
        ...                     lambda_electrostatics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
        ...                     lambda_sterics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
        ...             solvent:
        ...                 alchemical_path:
        ...                     lambda_electrostatics: [1.0, 0.8, 0.6, 0.3, 0.0]
        ...                     lambda_sterics: [1.0, 0.8, 0.6, 0.3, 0.0]
        ... experiments:
        ...     components:
        ...         receptor: [rec_conf1pdb, rec_conf2pdb]
        ...         ligand: lig
        ...         solvent: solv1
        ...     protocol: absolute-binding
        ... '''
        >>> yaml_content = textwrap.dedent(yaml_content)
        >>> raw = yaml.load(yaml_content)
        >>> expanded = YamlBuilder(yaml_content)._expand_molecules(raw)
        >>> expanded == yaml.load(textwrap.dedent(expected_content))
        True

        """
        expanded_content = copy.deepcopy(yaml_content)

        if 'molecules' in expanded_content:
            all_comb_mols = set()  # keep track of all combinatorial molecules
            for comb_mol_name, comb_molecule in expanded_content['molecules'].items():

                # First transform "select: all" syntax into a combinatorial "select"
                if 'select' in comb_molecule and comb_molecule['select'] == 'all':
                    # We need a file to select from
                    # TODO this should be checked only in _parse_molecules(), will be
                    # TODO easy when we'll have automatic data validation
                    if 'filepath' not in comb_molecule:
                        raise YamlParseError('Molecule {}: Need filepath with select: all'.format(
                            comb_mol_name))

                    # Get the number of models in the file
                    extension = os.path.splitext(comb_molecule['filepath'])[1][1:]  # remove dot
                    with utils.temporary_cd(self._script_dir):
                        if extension == 'pdb':
                            n_models = PDBFile(comb_molecule['filepath']).getNumFrames()
                        elif extension == 'csv' or extension == 'smiles':
                            with open(comb_molecule['filepath'], 'r') as smiles_file:
                                n_models = len(filter(bool, smiles_file.readlines()))  # remove blank lines
                        elif extension == 'sdf' or extension == 'mol2':
                            if not utils.is_openeye_installed():
                                err_msg = 'Molecule {}: Cannot "select" from {} file without OpenEye toolkit'
                                raise RuntimeError(err_msg.format(comb_mol_name, extension))
                            n_models = utils.read_oe_molecule(comb_molecule['filepath']).NumConfs()
                        else:
                            raise YamlParseError('Molecule {}: Cannot "select" from {} file'.format(
                                    comb_mol_name, extension))

                    # Substitute select: all with list of all models indices to trigger combinations
                    comb_molecule['select'] = range(n_models)

                # Find all combinations
                comb_molecule = utils.CombinatorialTree(comb_molecule)
                combinations = {comb_mol_name + '_' + name: mol
                                for name, mol in comb_molecule.named_combinations(
                                                    separator='_', max_name_length=30)}
                if len(combinations) > 1:
                    all_comb_mols.add(comb_mol_name)  # the combinatorial molecule will be removed
                    expanded_content['molecules'].update(combinations)  # add all combinations

                    # Check if experiments is a list or a dict
                    if isinstance(expanded_content['experiments'], list):
                        experiment_names = expanded_content['experiments']
                    else:
                        experiment_names = ['experiments']

                    # Resolve combinatorial molecules in experiments
                    for exp_name in experiment_names:
                        components = expanded_content[exp_name]['components']
                        for component_name in ['receptor', 'ligand']:
                            component = components[component_name]
                            if isinstance(component, list):
                                try:
                                    i = component.index(comb_mol_name)
                                    component[i:i+1] = combinations.keys()
                                except ValueError:
                                    pass
                            elif component == comb_mol_name:
                                components[component_name] = combinations.keys()

            # Delete old combinatorial molecules
            for comb_mol_name in all_comb_mols:
                del expanded_content['molecules'][comb_mol_name]

        return expanded_content


    @property
    def yank_options(self):
        return self._isolate_yank_options(self.options)

    def __init__(self, yaml_source):
        """Parse the given YAML configuration file.

        This does not build the actual experiment but simply checks that the syntax
        is correct and loads the configuration into memory.

        Parameters
        ----------
        yaml_source : str
            A path to the YAML script or the YAML content.

        """

        self._mpicomm = None  # MPI communicator

        # TODO check version of yank-yaml language
        # TODO what if there are multiple streams in the YAML file?
        # Load YAML script and decide working directory for relative paths
        try:
            with open(yaml_source, 'r') as f:
                yaml_content = yaml.load(f)
            self._script_dir = os.path.dirname(yaml_source)
        except IOError:
            yaml_content = yaml.load(yaml_source)
            self._script_dir = os.getcwd()

        if yaml_content is None:
            raise YamlParseError('The YAML file is empty!')

        # Expand combinatorial molecules
        yaml_content = self._expand_molecules(yaml_content)

        # Save raw YAML content that will be needed when generating the YAML files
        self._raw_yaml = copy.deepcopy({key: yaml_content.get(key, {})
                                        for key in ['options', 'molecules', 'solvents', 'protocols']})

        # Validate and store options
        self._parse_options(yaml_content)

        # Initialize and configure database
        self._db = SetupDatabase(setup_dir=self._get_setup_dir(self.options))
        self._parse_molecules(yaml_content)
        self._parse_solvents(yaml_content)

        # Validate protocols
        self._parse_protocols(yaml_content)

        # Validate experiments
        self._parse_experiments(yaml_content)

    def build_experiment(self):
        """Set up and run all the Yank experiments."""
        # Throw exception if there are no experiments
        if len(self._experiments) == 0:
            raise YamlParseError('No experiments specified!')

        # Configure MPI, if requested
        if self.options['mpi']:
            from mpi4py import MPI
            MPI.COMM_WORLD.barrier()
            self._mpicomm = MPI.COMM_WORLD

        # Run all experiments with paths relative to the script directory
        with utils.temporary_cd(self._script_dir):
            # This is hard disk intensive, only process 0 should do it
            overwrite = True
            if self._mpicomm is None or self._mpicomm.rank == 0:
                overwrite, err_msg = self._check_resume()
            if self._mpicomm:
                overwrite = self._mpicomm.bcast(overwrite, root=0)
                if overwrite and self._mpicomm.rank != 0:
                    sys.exit()
            if overwrite:  # self._mpicomm is None or rank == 0
                raise YamlParseError(err_msg)

            for output_dir, combination in self._expand_experiments():
                self._run_experiment(combination, output_dir)

    def _validate_options(self, options):
        """Return a dictionary with YamlBuilder and Yank options validated."""
        template_options = self.DEFAULT_OPTIONS.copy()
        template_options.update(Yank.default_parameters)
        template_options.update(ReplicaExchange.default_parameters)
        template_options.update(utils.get_keyword_args(AbsoluteAlchemicalFactory.__init__))
        openmm_app_type = {'constraints': to_openmm_app}
        try:
            valid = utils.validate_parameters(options, template_options, check_unknown=True,
                                              process_units_str=True, float_to_int=True,
                                              special_conversions=openmm_app_type)
        except (TypeError, ValueError) as e:
            raise YamlParseError(str(e))
        return valid

    def _isolate_yank_options(self, options):
        """Return the options that do not belong to YamlBuilder."""
        return {opt: val for opt, val in options.items()
                if opt not in self.DEFAULT_OPTIONS}

    def _determine_experiment_options(self, experiment):
        """Merge the options specified in the experiment section with the general ones.

        Options in the general section have priority. Options in the experiment section
        are validated.

        Parameters
        ----------
        experiment : dict
            The dictionary encoding the experiment.

        Returns
        -------
        exp_options : dict
            A new dictionary containing the all the options that apply for the experiment.

        """
        exp_options = self.options.copy()
        exp_options.update(self._validate_options(experiment.get('options', {})))
        return exp_options

    def _parse_options(self, yaml_content):
        """Validate and store options in the script.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        """
        # Merge options and metadata and validate
        temp_options = yaml_content.get('options', {})
        temp_options.update(yaml_content.get('metadata', {}))

        # Validate options and fill in default values
        self.options = self.DEFAULT_OPTIONS.copy()
        self.options.update(self._validate_options(temp_options))

    def _parse_molecules(self, yaml_content):
        """Load molecules information and check that their syntax is correct.

        One and only one source must be specified (e.g. filepath, name). Also
        the parameters must be specified, and the extension of filepath must
        match one of the supported file formats.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        """
        file_formats = set(['mol2', 'sdf', 'pdb', 'smiles', 'csv'])
        sources = set(['filepath', 'name', 'smiles'])
        template_mol = {'filepath': 'str', 'name': 'str', 'smiles': 'str',
                        'parameters': 'str', 'epik': 0, 'select': 0,
                        'strip_protons': False}

        self._db.molecules = yaml_content.get('molecules', {})

        # First validate and convert
        for molecule_id, molecule in self._db.molecules.items():
            try:
                self._db.molecules[molecule_id] = utils.validate_parameters(molecule, template_mol,
                                                                            check_unknown=True)
            except (TypeError, ValueError) as e:
                raise YamlParseError(str(e))

        err_msg = ''
        for molecule_id, molecule in self._db.molecules.items():
            fields = set(molecule.keys())

            # Check that only one source is specified
            specified_sources = sources & fields
            if not specified_sources or len(specified_sources) > 1:
                err_msg = ('need only one between {} for molecule {}').format(
                    ', '.join(list(sources)), molecule_id)

            # Check supported file formats
            elif 'filepath' in specified_sources:
                extension = os.path.splitext(molecule['filepath'])[1][1:]  # remove '.'
                if extension not in file_formats:
                    err_msg = 'molecule {}, only {} files supported'.format(
                        molecule_id, ', '.join(file_formats))

            # Check that parameters are specified
            if 'parameters' not in fields:
                err_msg = 'no parameters specified for molecule {}'.format(molecule_id)

            # Check selection from multi-model input files
            if 'select' in fields and 'filepath' not in molecule:
                err_msg = 'molecule {}, need filepath select'.format(molecule_id)

            if err_msg != '':
                raise YamlParseError(err_msg)

    def _parse_solvents(self, yaml_content):
        """Load solvents information and check that their syntax is correct.

        The option nonbonded_method must be specified. All quantities are converted to
        simtk.app.Quantity objects or openmm.app.TYPE (e.g. app.PME, app.OBC2). This
        also perform some consistency checks to verify that the user did not mix
        implicit and explicit solvent parameters.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        """
        template_parameters = {'nonbonded_method': openmm.app.PME, 'nonbonded_cutoff': 1 * unit.nanometer,
                               'implicit_solvent': openmm.app.OBC2, 'clearance': 10.0 * unit.angstroms,
                               'positive_ion': 'string', 'negative_ion': 'string'}
        openmm_app_type = ('nonbonded_method', 'implicit_solvent')
        openmm_app_type = {option: to_openmm_app for option in openmm_app_type}

        self._db.solvents = yaml_content.get('solvents', {})

        # First validate and convert
        for solvent_id, solvent in self._db.solvents.items():
            try:
                self._db.solvents[solvent_id] = utils.validate_parameters(solvent, template_parameters,
                                                         check_unknown=True, process_units_str=True,
                                                         special_conversions=openmm_app_type)
            except (TypeError, ValueError, AttributeError) as e:
                raise YamlParseError(str(e))

        err_msg = ''
        for solvent_id, solvent in self._db.solvents.items():

            # Test mandatory parameters
            if 'nonbonded_method' not in solvent:
                err_msg = 'solvent {} must specify nonbonded_method'.format(solvent_id)
                raise YamlParseError(err_msg)

            # Test solvent consistency
            nonbonded_method = solvent['nonbonded_method']
            if nonbonded_method == openmm.app.NoCutoff:
                if 'nonbonded_cutoff' in solvent:
                    err_msg = ('solvent {} specify both nonbonded_method: NoCutoff and '
                               'and nonbonded_cutoff').format(solvent_id)
            else:
                if 'implicit_solvent' in solvent:
                    err_msg = ('solvent {} specify both nonbonded_method: {} '
                               'and implicit_solvent').format(solvent_id, nonbonded_method)
                elif 'clearance' not in solvent:
                    err_msg = ('solvent {} uses explicit solvent but '
                               'no clearance specified').format(solvent_id)

            # Raise error
            if err_msg != '':
                raise YamlParseError(err_msg)

    def _parse_protocols(self, yaml_content):
        """Validate protocols.

        Each protocol must contain a complex and a solvent phase and it must specify
        an alchemical_path with at least lambda_sterics and lambda_electrostatics.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        """
        self._protocols = yaml_content.get('protocols', {})

        for protocol_id, protocol in self._protocols.items():
            # Phases must be specified
            if 'phases' not in protocol:
                err_msg = 'Protocol {} must specify phases'
                raise YamlParseError(err_msg.format(protocol_id))
            phases = protocol['phases']

            # There must be complex and solvent phases
            if 'complex' not in phases or 'solvent' not in phases:
                err_msg = 'Protocol {} must specify complex and solvent phases.'
                raise YamlParseError(err_msg.format(protocol_id))

            for phase_name, phase in phases.items():
                # alchemical_path must be specified with sterics and electrostatics
                if 'alchemical_path' not in phase:
                    err_msg = 'Protocol {} phase {} must specify alchemical_path.'
                    raise YamlParseError(err_msg.format(protocol_id, phase_name))

                alchemical_path = phase['alchemical_path']
                if ('lambda_sterics' not in alchemical_path or
                            'lambda_electrostatics' not in alchemical_path):
                    err_msg = ('Protocol {} phase {} must specify both'
                               'lambda_sterics and lambda_electrostatics')
                    raise YamlParseError(err_msg.format(protocol_id, phase_name))

            # Check that lambda variables all have same dimensions
            self._get_alchemical_paths(protocol_id)

    def _expand_experiments(self):
        """Generates all possible combinations of experiment.

        Each generated experiment is uniquely named.

        Returns
        -------
        output_dir : str
            A unique path where to save the experiment output files relative to
            the main output directory specified by the user in the options.
        combination : dict
            The dictionary describing a single experiment.

        """
        output_dir = ''
        for exp_name, experiment in self._experiments.items():
            if len(self._experiments) > 1:
                output_dir = exp_name

            # Loop over all combinations
            for name, combination in experiment.named_combinations(separator='_', max_name_length=50):
                yield os.path.join(output_dir, name), combination

    def _parse_experiments(self, yaml_content):
        """Perform dry run and validate components, protocol and options of every combination.

        Receptor, ligand, solvent and protocol must be already loaded. If they are
        not found an exception is raised. Experiments options are validated as well.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        """
        experiment_template = {'components': {}, 'options': {}, 'protocol': 'str'}
        components_template = {'receptor': 'str', 'ligand': 'str', 'solvent': 'str'}

        if 'experiments' not in yaml_content:
            self._experiments = {}
            return

        # Check if there is a sequence of experiments or a single one
        if isinstance(yaml_content['experiments'], list):
            self._experiments = {exp_name: utils.CombinatorialTree(yaml_content[exp_name])
                                 for exp_name in yaml_content['experiments']}
        else:
            self._experiments = {'experiments': utils.CombinatorialTree(yaml_content['experiments'])}

        # Check validity of every experiment combination
        err_msg = ''
        for exp_name, exp in self._expand_experiments():
            if exp_name == '':
                exp_name = 'experiments'

            # Check if we can identify components
            if 'components' not in exp:
                raise YamlParseError('Cannot find components for {}'.format(exp_name))
            components = exp['components']

            # Validate and check for unknowns
            try:
                utils.validate_parameters(exp, experiment_template, check_unknown=True)
                utils.validate_parameters(components, components_template, check_unknown=True)
                self._validate_options(exp.get('options', {}))
            except (ValueError, TypeError) as e:
                raise YamlParseError(str(e))

            # Check that components have been specified
            if components['receptor'] not in self._db.molecules:
                err_msg = 'Cannot identify receptor for {}'.format(exp_name)
            elif components['ligand'] not in self._db.molecules:
                err_msg = 'Cannot identify ligand for {}'.format(exp_name)
            elif components['solvent'] not in self._db.solvents:
                err_msg = 'Cannot identify solvent for {}'.format(exp_name)

            # Check that protocol has been specified
            if 'protocol' not in exp:
                err_msg = 'Cannot find protocol for {}.'.format(exp_name)
            elif exp['protocol'] not in self._protocols:
                err_msg = 'Cannot identify protocol for {}'.format(exp_name)

            if err_msg != '':
                raise YamlParseError(err_msg)

    @staticmethod
    def _get_setup_dir(experiment_options):
        """Return the path to the directory where the setup output files
        should be stored.

        Parameters
        ----------
        experiment_options : dict
            A dictionary containing the validated options that apply to the
            experiment as obtained by _determine_experiment_options().

        """
        return os.path.join(experiment_options['output_dir'], experiment_options['setup_dir'])

    @staticmethod
    def _get_experiment_dir(experiment_options, experiment_subdir):
        """Return the path to the directory where the experiment output files
        should be stored.

        Parameters
        ----------
        experiment_options : dict
            A dictionary containing the validated options that apply to the
            experiment as obtained by _determine_experiment_options().
        experiment_subdir : str
            The relative path w.r.t. the main experiments directory (determined
            through the options) of the experiment-specific subfolder.

        """
        return os.path.join(experiment_options['output_dir'],
                            experiment_options['experiments_dir'], experiment_subdir)

    def _check_resume_experiment(self, experiment_dir):
        """Check if Yank output files already exist.

        Parameters
        ----------
        experiment_dir : dict
            The path to the directory that should contain the output files.

        Returns
        -------
        bool
            True if NetCDF output files already exist, False otherwise.

        """
        # Check that output directory exists
        if not os.path.isdir(experiment_dir):
            return False

        # Check that complex and solvent NetCDF files exist
        complex_file_path = glob.glob(os.path.join(experiment_dir, 'complex-*.nc'))
        solvent_file_path = glob.glob(os.path.join(experiment_dir, 'solvent-*.nc'))
        if len(complex_file_path) == 0 or len(solvent_file_path) == 0:
            return False

        output_file_paths = complex_file_path + solvent_file_path
        return all(os.path.getsize(f) > 0 for f in output_file_paths)

    def _check_resume(self):
        """Perform dry run to check if we are going to overwrite files.

        If we find folders that YamlBuilder should create we return True
        unless resume_setup or resume_simulation are found, in which case we
        assume we need to use the existing files. We never overwrite files, the
        user is responsible to delete them or move them.

        It's important to check all possible combinations at the beginning to
        avoid interrupting the user simulation after few experiments.

        Returns
        -------
        overwrite : bool
            True in case we will have to overwrite some files, False otherwise.
        err_msg : str
            An error message in case overwrite is True, an empty string otherwise.

        """
        err_msg = ''
        overwrite = False
        for exp_sub_dir, combination in self._expand_experiments():
            # Determine and validate options
            exp_options = self._determine_experiment_options(combination)
            resume_sim = exp_options['resume_simulation']
            resume_setup = exp_options['resume_setup']

            # Identify components
            components = combination['components']
            receptor_id = components['receptor']
            ligand_id = components['ligand']
            solvent_id = components['solvent']

            # Check experiment dir
            experiment_dir = self._get_experiment_dir(exp_options, exp_sub_dir)
            if self._check_resume_experiment(experiment_dir) and not resume_sim:
                err_msg = 'experiment files in directory {}'.format(experiment_dir)
                solving_option = 'resume_simulation'
            else:
                # Check system and molecule setup dirs
                self._db.setup_dir = self._get_setup_dir(exp_options)
                is_sys_setup = self._db.is_system_setup(receptor_id, ligand_id, solvent_id)
                if is_sys_setup and not resume_setup:
                    system_dir = self._db.get_system_dir(receptor_id, ligand_id, solvent_id)
                    err_msg = 'system setup directory {}'.format(system_dir)
                else:
                    for molecule_id in [receptor_id, ligand_id]:
                        is_processed = self._db.is_molecule_setup(molecule_id)[1]
                        if is_processed and not resume_setup:
                            err_msg = 'molecule {} file'.format(molecule_id)
                            break

                if err_msg != '':
                    solving_option = 'resume_setup'

            # Check for errors
            if err_msg != '':
                overwrite = True
                err_msg += (' already exists; cowardly refusing to proceed. Move/delete '
                            'directory or set {} options').format(solving_option)
                break

        return overwrite, err_msg

    def _generate_yaml(self, experiment, file_path):
        """Generate the minimum YAML file needed to reproduce the experiment.

        Parameters
        ----------
        experiment : dict
            The dictionary describing a single experiment.
        file_path : str
            The path to the file to save.

        """
        yaml_dir = os.path.dirname(file_path)
        components = set(experiment['components'].values())

        # Molecules section data
        mol_section = {mol_id: molecule for mol_id, molecule in self._raw_yaml['molecules'].items()
                       if mol_id in components}

        # Solvents section data
        sol_section = {solvent_id: solvent for solvent_id, solvent in self._raw_yaml['solvents'].items()
                       if solvent_id in components}

        # Protocols section data
        prot_section = {protocol_id: protocol for protocol_id, protocol in self._raw_yaml['protocols'].items()
                        if protocol_id in experiment['protocol']}

        # We pop the options section in experiment and merge it to the general one
        exp_section = experiment.copy()
        opt_section = self._raw_yaml['options'].copy()
        opt_section.update(exp_section.pop('options', {}))

        # Convert relative paths to new script directory
        mol_section = copy.deepcopy(mol_section)  # copy to avoid modifying raw yaml
        for molecule in mol_section.values():
            if 'filepath' in molecule and not os.path.isabs(molecule['filepath']):
                molecule['filepath'] = os.path.relpath(molecule['filepath'], yaml_dir)

        try:
            output_dir = opt_section['output_dir']
        except KeyError:
            output_dir = self.DEFAULT_OPTIONS['output_dir']
        if not os.path.isabs(output_dir):
            opt_section['output_dir'] = os.path.relpath(output_dir, yaml_dir)

        # If we are converting a combinatorial experiment into a
        # single one we must set the correct experiment directory
        experiment_dir = os.path.relpath(yaml_dir, output_dir)
        if experiment_dir != self.DEFAULT_OPTIONS['experiments_dir']:
            opt_section['experiments_dir'] = experiment_dir

        # Create YAML with the sections in order
        dump_options = {'Dumper': YankDumper, 'line_break': '\n', 'indent': 4}
        yaml_content = yaml.dump({'options': opt_section}, explicit_start=True, **dump_options)
        yaml_content += yaml.dump({'molecules': mol_section},  **dump_options)
        yaml_content += yaml.dump({'solvents': sol_section},  **dump_options)
        yaml_content += yaml.dump({'protocols': prot_section},  **dump_options)
        yaml_content += yaml.dump({'experiments': exp_section},  **dump_options)

        # Export YAML into a file
        with open(file_path, 'w') as f:
            f.write(yaml_content)

    def _get_alchemical_paths(self, protocol_id):
        """Return the list of AlchemicalStates specified in the protocol.

        Parameters
        ----------
        protocol_id : str
            The protocol ID specified in the YAML script.

        Returns
        -------
        alchemical_paths : dict of list of AlchemicalState
            alchemical_paths[phase] is the list of AlchemicalStates specified in the
            YAML script for protocol 'protocol_id' and phase 'phase'.

        """
        alchemical_protocol = {}
        for phase_name, phase in self._protocols[protocol_id]['phases'].items():
            # Separate lambda variables names from their associated lists
            lambdas, values = zip(*phase['alchemical_path'].items())

            # Transpose so that each row contains single alchemical state values
            values = zip(*values)

            alchemical_protocol[phase_name] = [AlchemicalState(
                                               **{var: val for var, val in zip(lambdas, state_values)}
                                               ) for state_values in values]
        return alchemical_protocol

    def _run_experiment(self, experiment, experiment_dir):
        """Prepare and run a single experiment.

        Parameters
        ----------
        experiment : dict
            A dictionary describing a single experiment
        experiment_dir : str
            The directory where to store the output files relative to the main
            output directory as specified by the user in the YAML script

        """
        components = experiment['components']
        exp_name = 'experiments' if experiment_dir == '' else os.path.basename(experiment_dir)

        # Get and validate experiment sub-options
        exp_opts = self._determine_experiment_options(experiment)
        yank_opts = self._isolate_yank_options(exp_opts)

        # Set database path
        self._db.setup_dir = self._get_setup_dir(exp_opts)

        # TODO configure platform and precision when they are fixed in Yank

        # Create directory and determine if we need to resume the simulation
        results_dir = self._get_experiment_dir(exp_opts, experiment_dir)
        resume = os.path.isdir(results_dir)
        if self._mpicomm is None or self._mpicomm.rank == 0:
            if not resume:
                os.makedirs(results_dir)
            else:
                resume = self._check_resume_experiment(results_dir)
        if self._mpicomm:  # process 0 send result to other processes
            resume = self._mpicomm.bcast(resume, root=0)

        # Configure logger for this experiment
        utils.config_root_logger(exp_opts['verbose'], os.path.join(results_dir, exp_name + '.log'),
                                 self._mpicomm)

        # Initialize simulation
        yank = Yank(results_dir, mpicomm=self._mpicomm, **yank_opts)

        if resume:
            yank.resume()
        else:
            if self._mpicomm is None or self._mpicomm.rank == 0:
                # Export YAML file for reproducibility
                self._generate_yaml(experiment, os.path.join(results_dir, exp_name + '.yaml'))

                # Setup the system
                logger.info('Setting up the system for {}, {} and {}'.format(*components.values()))
                system_dir = self._db.get_system(components, exp_opts['pack'])

                # Get ligand resname for alchemical atom selection
                ligand_descr = self._db.molecules[components['ligand']]
                ligand_dsl = utils.get_mol2_resname(ligand_descr['filepath'])
                if ligand_dsl is None:
                    ligand_dsl = 'MOL'
                ligand_dsl = 'resname ' + ligand_dsl

                # System configuration
                create_system_filter = set(('nonbonded_method', 'nonbonded_cutoff', 'implicit_solvent',
                                            'constraints', 'hydrogen_mass'))
                solvent = self._db.solvents[components['solvent']]
                system_pars = {opt: solvent[opt] for opt in create_system_filter if opt in solvent}
                system_pars.update({opt: exp_opts[opt] for opt in create_system_filter
                                    if opt in exp_opts})

                # Convert underscore_parameters to camelCase for OpenMM API
                system_pars = {utils.underscore_to_camelcase(opt): value
                               for opt, value in system_pars.items()}

                # Prepare system
                phases, systems, positions, atom_indices = pipeline.prepare_amber(system_dir, ligand_dsl,
                                                                 system_pars, ligand_descr['net_charge'])

                # Create thermodynamic state
                thermodynamic_state = ThermodynamicState(temperature=exp_opts['temperature'],
                                                         pressure=exp_opts['pressure'])

                # Get protocols as list of AlchemicalStates and adjust phase name
                alchemical_paths = self._get_alchemical_paths(experiment['protocol'])
                suffix = 'explicit' if 'explicit' in phases[0] else 'implicit'
                alchemical_paths = {phase + '-' + suffix: protocol
                                    for phase, protocol in alchemical_paths.items()}

                # Create new simulation
                yank.create(phases, systems, positions, atom_indices,
                            thermodynamic_state, protocols=alchemical_paths)

            # Run the simulation
            if self._mpicomm:  # wait for the simulation to be prepared
                debug_msg = 'Node {}/{}: MPI barrier'.format(self._mpicomm.rank, self._mpicomm.size)
                logger.debug(debug_msg + ' - waiting for the simulation to be created.')
                self._mpicomm.barrier()
                if self._mpicomm.rank != 0:
                    yank.resume()  # resume from netcdf file created by root node
        yank.run()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
