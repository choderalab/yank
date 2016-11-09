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
import csv
import copy
import yaml
import logging
import collections

logger = logging.getLogger(__name__)

import numpy as np
import openmoltools as omt
from simtk import unit, openmm
from simtk.openmm.app import PDBFile, AmberPrmtopFile
from alchemy import AlchemicalState, AbsoluteAlchemicalFactory
from schema import Schema, And, Or, Use, Optional, SchemaError

from . import utils
from . import pipeline
from .yank import Yank
from .repex import ReplicaExchange, ThermodynamicState
from .sampling import ModifiedHamiltonianExchange


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


# ============================================================================================
# UTILITY CLASSES
# ============================================================================================

class YamlParseError(Exception):
    """Represent errors occurring during parsing of Yank YAML file."""
    def __init__(self, message):
        super(YamlParseError, self).__init__(message)
        logger.error(message)


class YankLoader(yaml.Loader):
    """PyYAML Loader that recognized !Combinatorial nodes and load OrderedDicts."""
    def __init__(self, *args, **kwargs):
        super(YankLoader, self).__init__(*args, **kwargs)
        self.add_constructor(u'!Combinatorial', self.combinatorial_constructor)
        self.add_constructor(u'!Ordered', self.ordered_constructor)

    @staticmethod
    def combinatorial_constructor(loader, node):
        """Constructor for YAML !Combinatorial entries."""
        return utils.CombinatorialLeaf(loader.construct_sequence(node))

    @staticmethod
    def ordered_constructor(loader, node):
        """Constructor for YAML !Ordered tag."""
        loader.flatten_mapping(node)
        return collections.OrderedDict(loader.construct_pairs(node))


class YankDumper(yaml.Dumper):
    """PyYAML Dumper that always return sequences in flow style and maps in block style."""
    def __init__(self, *args, **kwargs):
        super(YankDumper, self).__init__(*args, **kwargs)
        self.add_representer(utils.CombinatorialLeaf, self.combinatorial_representer)
        self.add_representer(collections.OrderedDict, self.ordered_representer)

    def represent_sequence(self, tag, sequence, flow_style=None):
        return yaml.Dumper.represent_sequence(self, tag, sequence, flow_style=True)

    def represent_mapping(self, tag, mapping, flow_style=None):
        return yaml.Dumper.represent_mapping(self, tag, mapping, flow_style=False)

    @staticmethod
    def combinatorial_representer(dumper, data):
        """YAML representer CombinatorialLeaf nodes."""
        return dumper.represent_sequence(u'!Combinatorial', data)

    @staticmethod
    def ordered_representer(dumper, data):
        """YAML representer OrderedDict nodes."""
        return dumper.represent_mapping(u'!Ordered', data)


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

    def __init__(self, setup_dir, molecules=None, solvents=None, systems=None):
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
        self.systems = systems

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

    def get_system_files_paths(self, system_id):
        """Return the paths to the systems files.

        Parameters
        ----------
        system_id : str
            The ID of the system.

        Returns
        -------
        system_files_paths : list of namedtuple
            Elements of the list contain the paths to the system files for
            each phase. Each namedtuple contains the fields position_path (e.g.
            inpcrd or gro) and topology_path (e.g. prmtop or top).

        """
        Paths = collections.namedtuple('Paths', ['position_path', 'topology_path'])
        system_dir = os.path.join(self.setup_dir, self.SYSTEMS_DIR, system_id)
        if 'receptor' in self.systems[system_id]:
            system_files_paths = [
                Paths(position_path=os.path.join(system_dir, 'complex.inpcrd'),
                      topology_path=os.path.join(system_dir, 'complex.prmtop')),
                Paths(position_path=os.path.join(system_dir, 'solvent.inpcrd'),
                      topology_path=os.path.join(system_dir, 'solvent.prmtop'))
            ]
        elif 'solute' in self.systems[system_id]:
            system_files_paths = [
                Paths(position_path=os.path.join(system_dir, 'solvent1.inpcrd'),
                      topology_path=os.path.join(system_dir, 'solvent1.prmtop')),
                Paths(position_path=os.path.join(system_dir, 'solvent2.inpcrd'),
                      topology_path=os.path.join(system_dir, 'solvent2.prmtop'))
            ]
        else:
            system_files_paths = [
                Paths(position_path=self.systems[system_id]['phase1_path'][0],
                      topology_path=self.systems[system_id]['phase1_path'][1]),
                Paths(position_path=self.systems[system_id]['phase2_path'][0],
                      topology_path=self.systems[system_id]['phase2_path'][1])
            ]

        return system_files_paths

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
        if 'antechamber' in molecule_descr:
            files_to_check = [('filepath', molecule_id_path + '.gaff.mol2'),
                              (['leap', 'parameters'], molecule_id_path + '.frcmod')]

        # If the molecule must be generated by OpenEye, a mol2 should have been created
        elif extension is None or extension == '.smiles' or extension == '.csv':
            files_to_check = [('filepath', molecule_id_path + '.mol2')]

        # If we have to strip the protons off a PDB, a new PDB should have been created
        elif 'strip_protons' in molecule_descr and molecule_descr['strip_protons']:
            files_to_check = [('filepath', molecule_id_path + '.pdb')]

        # If a single structure must be extracted we search for output
        elif 'select' in molecule_descr:
            files_to_check = [('filepath', molecule_id_path + extension)]

        # Check if this needed to be processed at all
        if not files_to_check:
            return True, False

        # Check if all output files exist
        all_file_exist = True
        for descr_key, file_path in files_to_check:
            all_file_exist &= os.path.isfile(file_path) and os.path.getsize(file_path) > 0
            if all_file_exist:  # Make sure internal description is correct
                try:
                    molecule_descr[descr_key] = file_path
                except TypeError:  # nested key, list are unhashable
                    molecule_descr[descr_key[0]][descr_key[1]].append(file_path)

        # Compute and update small molecule net charge
        if all_file_exist:
            extension = os.path.splitext(molecule_descr['filepath'])[1]
            # TODO what if this is a peptide? This should be computed in get_system()
            if extension == '.mol2':
                molecule_descr['net_charge'] = utils.Mol2File(molecule_descr['filepath']).net_charge

        return all_file_exist, all_file_exist

    def is_system_setup(self, system_id):
        """Check whether the system has been already processed.

        Parameters
        ----------
        system_id : str
            The ID of the system.

        Returns
        -------
        is_setup : bool
            True if the system is ready to be used for an experiment. Either because
            the system has directly provided the system files, or because it already
            went through the setup pipeline.
        is_processed : bool
            True if the system has already gone through the setup pipeline.

        """
        if 'ligand' in self.systems[system_id] or 'solute' in self.systems[system_id]:
            system_files_paths = self.get_system_files_paths(system_id)
            is_setup = (os.path.exists(system_files_paths[0].position_path) and
                        os.path.exists(system_files_paths[0].topology_path) and
                        os.path.exists(system_files_paths[1].position_path) and
                        os.path.exists(system_files_paths[1].topology_path))
            return is_setup, is_setup
        else:
            return True, False

    def get_system(self, system_id):
        """Make sure that the system files are set up and return the system folder.

        If necessary, create the prmtop and inpcrd files from the given components.
        The system files are generated with tleap. If no molecule specifies a general
        force field, leaprc.ff14SB is loaded.

        Parameters
        ----------
        system_id : str
            The ID of the system.

        Returns
        -------
        system_files_paths : list of namedtuple
            Elements of the list contain the paths to the system files for
            each phase. Each namedtuple contains the fields position_path (e.g.
            inpcrd or gro) and topology_path (e.g. prmtop or top).

        """
        # Check if system has been already processed
        system_files_paths = self.get_system_files_paths(system_id)
        if self.is_system_setup(system_id)[0]:
            return system_files_paths

        system_descr = self.systems[system_id]

        if 'receptor' in system_descr:  # binding free energy calculation
            receptor_id = system_descr['receptor']
            ligand_id = system_descr['ligand']
            solvent_id = system_descr['solvent']
            system_parameters = system_descr['leap']['parameters']

            # solvent phase
            self._setup_system(system_files_paths[1].position_path, False,
                               0, system_parameters, solvent_id, ligand_id)

            try:
                alchemical_charge = int(round(self.molecules[ligand_id]['net_charge']))
            except KeyError:
                alchemical_charge = 0

            # complex phase
            self._setup_system(system_files_paths[0].position_path,
                               system_descr['pack'], alchemical_charge,
                               system_parameters,  solvent_id, receptor_id,
                               ligand_id)
        else:  # partition/solvation free energy calculation
            solute_id = system_descr['solute']
            solvent1_id = system_descr['solvent1']
            solvent2_id = system_descr['solvent2']
            system_parameters = system_descr['leap']['parameters']

            # solvent1 phase
            self._setup_system(system_files_paths[0].position_path, False,
                               0, system_parameters, solvent1_id, solute_id)

            # solvent2 phase
            self._setup_system(system_files_paths[1].position_path, False,
                               0, system_parameters, solvent2_id, solute_id)

        return system_files_paths

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
                        # TODO: Make sure this is working after Py 3.X conversion
                        smiles_lines = [line for line in smiles_file.readlines() if bool(line)]
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
                    if extension is None:
                        raise RuntimeError('Cannot generate molecule {} without OpenEye'.format(mol_id))
                    else:
                        raise RuntimeError('Cannot support {} files without OpenEye'.format(extension[1:]))

                # Retrieve the first SMILES string (eventually extracted
                # while handling of the 'select' keyword above)
                if extension is not None:
                    with open(mol_descr['filepath'], 'r') as smiles_file:
                        # Automatically detect if delimiter is comma or semicolon
                        first_line = smiles_file.readline()
                        for delimiter in ',;':
                            logger.debug("Attempt to parse smiles file with delimiter '{}'".format(delimiter))
                            line_fields = first_line.split(delimiter)
                            # If there is only one column, take that, otherwise take second
                            if len(line_fields) > 1:
                                smiles_str = line_fields[1].strip()
                            else:
                                smiles_str = line_fields[0].strip()

                            # try to generate the smiles and try new delimiter if it fails
                            mol_descr['smiles'] = smiles_str
                            try:
                                oe_molecule = self._generate_molecule(mol_id)
                            except (ValueError, RuntimeError):
                                oe_molecule = None
                                continue
                            break

                        # Raise an error if no delimiter worked
                        if oe_molecule is None:
                            raise RuntimeError('Cannot detect SMILES file format.')
                else:
                    # Generate molecule from mol_descr['smiles']
                    oe_molecule = self._generate_molecule(mol_id)

                # Cache atom positions
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
                epik_base_path = os.path.join(mol_dir, mol_id + '-epik.')
                epik_mae_file = epik_base_path + 'mae'
                epik_mol2_file = epik_base_path + 'mol2'
                epik_sdf_file = epik_base_path + 'sdf'

                # Run epik and convert from maestro to both mol2 and sdf
                # to not lose neither the penalties nor the residue name
                epik_kwargs = mol_descr['epik']
                omt.schrodinger.run_epik(mol_descr['filepath'], epik_mae_file, **epik_kwargs)
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
            if 'antechamber' in mol_descr:
                # Generate charges with OpenEye if requested
                if 'openeye' in mol_descr:
                    if not utils.is_openeye_installed():
                        err_msg = ('Cannot find OpenEye toolkit to compute charges '
                                   'for molecule {}').format(mol_id)
                        logger.error(err_msg)
                        raise RuntimeError(err_msg)
                    mol2_file_path = os.path.join(mol_dir, mol_id + '.mol2')
                    oe_molecule = utils.read_oe_molecule(mol_descr['filepath'])

                    # Setting keep_confs = None keeps the original conformation
                    oe_molecule = omt.openeye.get_charges(oe_molecule, keep_confs=None)
                    residue_name = utils.Mol2File(mol_descr['filepath']).resname
                    omt.openeye.molecule_to_mol2(oe_molecule, mol2_file_path,
                                                 residue_name=residue_name)

                    utils.Mol2File(mol2_file_path).net_charge = None  # normalize charges
                    net_charge = None  # we don't need Epik's net charge
                    mol_descr['filepath'] = mol2_file_path

                # Generate parameters
                charge_method = mol_descr['antechamber']['charge_method']
                input_mol_path = os.path.abspath(mol_descr['filepath'])
                with omt.utils.temporary_cd(mol_dir):
                    omt.amber.run_antechamber(mol_id, input_mol_path,
                                              charge_method=charge_method,
                                              net_charge=net_charge)

                # Save new parameters paths
                mol_descr['filepath'] = os.path.join(mol_dir, mol_id + '.gaff.mol2')
                mol_descr['leap']['parameters'].append(os.path.join(mol_dir, mol_id + '.frcmod'))

                # Normalize charges if not done before
                if 'openeye' not in mol_descr:
                    utils.Mol2File(mol_descr['filepath']).net_charge = None

            # Determine small molecule net charge
            extension = os.path.splitext(mol_descr['filepath'])[1]
            if extension == '.mol2':
                # TODO what if this is a peptide? this should be computed in get_system()
                mol_descr['net_charge'] = utils.Mol2File(mol_descr['filepath']).net_charge

            # Keep track of processed molecule
            self._processed_mols.add(mol_id)

    def _setup_system(self, system_file_path, pack, alchemical_charge,
                      system_parameters, solvent_id, *molecule_ids):
        """Setup a system and create its prmtop/inpcrd files.

        IMPORTANT: This function does not check if it's about to overwrite
        files. Use get_system() for safe setup.

        Parameters
        ----------
        system_file_path : str
            The path to either the prmtop or inpcrd output file. The other one
            will be saved in the same folder with the same base name.
        pack : bool
            True to automatically solve atom clashes and reduce box dimension.
        alchemical_charge : int
            Number of counterions to alchemically modify during the simulation.
        system_parameters : list of str
            Contain the parameters file that must be loaded in tleap for the
            system in addition to the molecule-specific ones.
        solvent_id : str
            The ID of the solvent.
        *molecule_ids : list-like of str
            List the IDs of the molecules to pack together in the system.

        """
        solvent = self.solvents[solvent_id]
        self._setup_molecules(*molecule_ids)  # Be sure molecules are set up

        # Create tleap script
        tleap = utils.TLeap()

        # Load all parameters
        # --------------------
        tleap.new_section('Load parameters')

        for mol_id in molecule_ids:
            molecule_parameters = self.molecules[mol_id]['leap']['parameters']
            tleap.load_parameters(*molecule_parameters)

        tleap.load_parameters(*system_parameters)

        # Load molecules and create complexes
        # ------------------------------------
        tleap.new_section('Load molecules')
        for mol_id in molecule_ids:
            tleap.load_group(name=mol_id, file_path=self.molecules[mol_id]['filepath'])

        if len(molecule_ids) > 1:
            # Check that molecules don't have clashing atoms. Also, if the ligand
            # is too far away from the molecule we want to pull it closer
            # TODO this check should be available even without OpenEye
            if pack and utils.is_openeye_installed():

                # Load atom positions of all molecules
                positions = [0 for _ in molecule_ids]
                for i, mol_id in enumerate(molecule_ids):
                    if mol_id not in self._pos_cache:
                        self._pos_cache[mol_id] = utils.get_oe_mol_positions(
                                utils.read_oe_molecule(self.molecules[mol_id]['filepath']))
                    positions[i] = self._pos_cache[mol_id]

                # Find and apply the transformation to fix clashing
                # TODO this doesn't work with more than 2 molecule_ids
                try:
                    max_dist = solvent['clearance'].value_in_unit(unit.angstrom) / 1.5
                except KeyError:
                    max_dist = 10.0
                transformation = pack_transformation(positions[0], positions[1],
                                                     self.CLASH_THRESHOLD, max_dist)
                if (transformation != np.identity(4)).any():
                    logger.warning('Changing starting positions for {}.'.format(molecule_ids[1]))
                    tleap.new_section('Fix clashing atoms')
                    tleap.transform(molecule_ids[1], transformation)

            # Create complex
            tleap.new_section('Create complex')
            tleap.combine('complex', *molecule_ids)
            unit_to_solvate = 'complex'
        else:
            unit_to_solvate = molecule_ids[0]

        # Configure solvent
        # ------------------
        if solvent['nonbonded_method'] == openmm.app.NoCutoff:
            try:
                implicit_solvent = solvent['implicit_solvent']
            except KeyError:  # vacuum
                pass
            else:  # implicit solvent
                tleap.new_section('Set GB radii to recommended values for OBC')
                tleap.add_commands('set default PBRadii {}'.format(
                    pipeline.get_leap_recommended_pbradii(implicit_solvent)))
        else:  # explicit solvent
            tleap.new_section('Solvate systems')

            # Add alchemically modified ions
            if alchemical_charge != 0:
                try:
                    if alchemical_charge > 0:
                        ion = solvent['negative_ion']
                    else:
                        ion = solvent['positive_ion']
                except KeyError:
                    err_msg = ('Found charged ligand but no indications for ions in '
                               'solvent {}').format(solvent_id)
                    logger.error(err_msg)
                    raise RuntimeError(err_msg)
                tleap.add_ions(unit=unit_to_solvate, ion=ion, num_ions=abs(alchemical_charge))

            # Neutralizing solvation box
            if 'positive_ion' in solvent:
                tleap.add_ions(unit=unit_to_solvate, ion=solvent['positive_ion'])
            if 'negative_ion' in solvent:
                tleap.add_ions(unit=unit_to_solvate, ion=solvent['negative_ion'])

            # Solvate unit
            clearance = float(solvent['clearance'].value_in_unit(unit.angstroms))
            tleap.solvate(group=unit_to_solvate, water_model='TIP3PBOX', clearance=clearance)

        # Check charge
        tleap.new_section('Check charge')
        tleap.add_commands('check ' + unit_to_solvate)

        # Save output files
        # ------------------
        system_dir = os.path.dirname(system_file_path)
        base_file_path = os.path.basename(system_file_path).split('.')[0]
        base_file_path = os.path.join(system_dir, base_file_path)

        # Create output directory
        if not os.path.exists(system_dir):
            os.makedirs(system_dir)

        # Save prmtop, inpcrd and reference pdb files
        tleap.new_section('Save prmtop and inpcrd files')
        tleap.save_group(unit_to_solvate, system_file_path)
        tleap.save_group(unit_to_solvate, base_file_path + '.pdb')

        # Save tleap script for reference
        tleap.export_script(base_file_path + '.leap.in')

        # Run tleap and log warnings
        warnings = tleap.run()
        for warning in warnings:
            logger.warning('TLeap: ' + warning)


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
    >>> import openmoltools as omt
    >>> import utils
    >>> setup_dir = utils.get_data_filename(os.path.join('..', 'examples',
    ...                                     'p-xylene-implicit', 'input'))
    >>> pxylene_path = os.path.join(setup_dir, 'p-xylene.mol2')
    >>> lysozyme_path = os.path.join(setup_dir, '181L-pdbfixer.pdb')
    >>> with omt.utils.temporary_directory() as tmp_dir:
    ...     yaml_content = '''
    ...     ---
    ...     options:
    ...       number_of_iterations: 1
    ...       output_dir: {}
    ...     molecules:
    ...       T4lysozyme:
    ...         filepath: {}
    ...       p-xylene:
    ...         filepath: {}
    ...         antechamber:
    ...           charge_method: bcc
    ...     solvents:
    ...       vacuum:
    ...         nonbonded_method: NoCutoff
    ...     systems:
    ...         my_system:
    ...             receptor: T4lysozyme
    ...             ligand: p-xylene
    ...             solvent: vacuum
    ...             leap:
    ...               parameters: [leaprc.gaff, leaprc.ff14SB]
    ...     protocols:
    ...       absolute-binding:
    ...         complex:
    ...           alchemical_path:
    ...             lambda_electrostatics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
    ...             lambda_sterics: [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0]
    ...         solvent:
    ...           alchemical_path:
    ...             lambda_electrostatics: [1.0, 0.8, 0.6, 0.3, 0.0]
    ...             lambda_sterics: [1.0, 0.8, 0.6, 0.3, 0.0]
    ...     experiments:
    ...       system: my_system
    ...       protocol: absolute-binding
    ...     '''.format(tmp_dir, lysozyme_path, pxylene_path)
    >>> yaml_builder = YamlBuilder(textwrap.dedent(yaml_content))
    >>> yaml_builder.build_experiments()

    """

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    DEFAULT_OPTIONS = {
        'verbose': False,
        'resume_setup': False,
        'resume_simulation': False,
        'output_dir': 'output',
        'setup_dir': 'setup',
        'experiments_dir': 'experiments',
        'platform': 'fastest',
        'precision': 'auto',
        'temperature': 298 * unit.kelvin,
        'pressure': 1 * unit.atmosphere,
        'constraints': openmm.app.HBonds,
        'hydrogen_mass': 1 * unit.amu
    }

    @property
    def yank_options(self):
        return self._isolate_yank_options(self.options)

    def __init__(self, yaml_source=None):
        """Constructor.

        Parameters
        ----------
        yaml_source : str or dict
            A path to the YAML script or the YAML content. If not specified, you
            can load it later by using parse() (default is None).

        """
        self.options = self.DEFAULT_OPTIONS.copy()  # General options (not experiment-specific)

        self._script_dir = os.getcwd()  # basic dir for relative paths
        self._db = None  # Database containing molecules created in parse()
        self._mpicomm = None  # MPI communicator
        self._raw_yaml = {}  # Unconverted input YAML script
        self._protocols = {}  # Alchemical protocols description
        self._experiments = {}  # Experiments description

        # Parse YAML script
        if yaml_source is not None:
            self.parse(yaml_source)

    def parse(self, yaml_source):
        """Parse the given YAML configuration file.

        Validate the syntax and load the script into memory. This does not build
        the actual experiment.

        Parameters
        ----------
        yaml_source : str or dict
            A path to the YAML script or the YAML content.

        Raises
        ------
        YamlParseError
            If the input YAML script is syntactically incorrect.

        """
        # TODO check version of yank-yaml language
        # TODO what if there are multiple streams in the YAML file?
        # Load YAML script and decide working directory for relative paths
        try:
            with open(yaml_source, 'r') as f:
                yaml_content = yaml.load(f, Loader=YankLoader)
            self._script_dir = os.path.dirname(yaml_source)
        except IOError:  # string
            yaml_content = yaml.load(yaml_source, Loader=YankLoader)
        except TypeError:  # dict
            yaml_content = yaml_source.copy()

        # Check that YAML loading was successful
        if yaml_content is None:
            raise YamlParseError('The YAML file is empty!')
        if not isinstance(yaml_content, dict):
            raise YamlParseError('Cannot load YAML from source: {}'.format(yaml_source))

        # Expand combinatorial molecules and systems
        yaml_content = self._expand_molecules(yaml_content)
        yaml_content = self._expand_systems(yaml_content)

        # Save raw YAML content that will be needed when generating the YAML files
        self._raw_yaml = copy.deepcopy({key: yaml_content.get(key, {})
                                        for key in ['options', 'molecules', 'solvents',
                                                    'systems', 'protocols']})

        # Validate options and overwrite defaults
        self.options.update(self._validate_options(utils.merge_dict(yaml_content.get('options', {}),
                                                                    yaml_content.get('metadata', {}))))

        # Setup MPI and general logging
        self._mpicomm = utils.initialize_mpi()
        utils.config_root_logger(self.options['verbose'], log_file_path=None, mpicomm=self._mpicomm)
        if self._mpicomm is None:
            logger.debug('MPI disabled.')
        else:
            logger.debug('MPI enabled.')

        # Initialize and configure database with molecules, solvents and systems
        self._db = SetupDatabase(setup_dir=self._get_setup_dir(self.options))
        self._db.molecules = self._validate_molecules(yaml_content.get('molecules', {}))
        self._db.solvents = self._validate_solvents(yaml_content.get('solvents', {}))
        self._db.systems = self._validate_systems(yaml_content.get('systems', {}))

        # Validate protocols
        self._protocols = self._validate_protocols(yaml_content.get('protocols', {}))

        # Validate experiments
        self._parse_experiments(yaml_content)

    def build_experiments(self):
        """Set up and run all the Yank experiments."""
        # Throw exception if there are no experiments
        if len(self._experiments) == 0:
            raise YamlParseError('No experiments specified!')

        # Setup and run all experiments with paths relative to the script directory
        with omt.utils.temporary_cd(self._script_dir):
            self._check_resume()
            self._setup_experiments()
            for output_dir, combination in self._expand_experiments():
                self._run_experiment(combination, output_dir)

    def setup_experiments(self):
        """Set up all Yank experiments without running them."""
        # Throw exception if there are no experiments
        if len(self._experiments) == 0:
            raise YamlParseError('No experiments specified!')

        # All paths must be relative to the script directory
        with omt.utils.temporary_cd(self._script_dir):
            self._check_resume(check_experiments=False)
            self._setup_experiments()

    # --------------------------------------------------------------------------
    # Options handling
    # --------------------------------------------------------------------------

    def _isolate_yank_options(self, options):
        """Return the options that do not belong to YamlBuilder."""
        return {opt: val for opt, val in utils.listitems(options)
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

    # --------------------------------------------------------------------------
    # Combinatorial expansion
    # --------------------------------------------------------------------------

    def _expand_molecules(self, yaml_content):
        """Expand combinatorial molecules.

        Generate new YAML content with no combinatorial molecules. The new content
        is identical to the old one but combinatorial molecules are substituted by
        the description of all the non-combinatorial molecules that they generate.
        Moreover, systems that use combinatorial molecules are updated with the new
        molecules ids.

        Parameters
        ----------
        yaml_content : dict
            The YAML content as returned by yaml.load().

        Returns
        -------
        expanded_content : dict
            The new YAML content with combinatorial molecules expanded.

        """
        expanded_content = copy.deepcopy(yaml_content)

        if 'molecules' not in expanded_content:
            return expanded_content

        # First substitute all 'select: all' with the correct combination of indices
        for comb_mol_name, comb_molecule in utils.listitems(expanded_content['molecules']):
            if 'select' in comb_molecule and comb_molecule['select'] == 'all':
                # Get the number of models in the file
                extension = os.path.splitext(comb_molecule['filepath'])[1][1:]  # remove dot
                with omt.utils.temporary_cd(self._script_dir):
                    if extension == 'pdb':
                        n_models = PDBFile(comb_molecule['filepath']).getNumFrames()
                    elif extension == 'csv' or extension == 'smiles':
                        with open(comb_molecule['filepath'], 'r') as smiles_file:
                            # TODO: Make sure this is working as expected from Py 3.X conversion
                            n_models = len([line for line in smiles_file.readlines() if bool(line)]) # remove blank lines
                    elif extension == 'sdf' or extension == 'mol2':
                        if not utils.is_openeye_installed():
                            err_msg = 'Molecule {}: Cannot "select" from {} file without OpenEye toolkit'
                            raise RuntimeError(err_msg.format(comb_mol_name, extension))
                        n_models = utils.read_oe_molecule(comb_molecule['filepath']).NumConfs()
                    else:
                        raise YamlParseError('Molecule {}: Cannot "select" from {} file'.format(
                            comb_mol_name, extension))

                # Substitute select: all with list of all models indices to trigger combinations
                comb_molecule['select'] = utils.CombinatorialLeaf(range(n_models))

        # Expand molecules and update molecule ids in systems
        expanded_content = utils.CombinatorialTree(expanded_content)
        update_nodes_paths = [('systems', '*', 'receptor'), ('systems', '*', 'ligand'),
                              ('systems', '*', 'solute')]
        expanded_content = expanded_content.expand_id_nodes('molecules', update_nodes_paths)

        return expanded_content

    def _expand_systems(self, yaml_content):
        """Expand combinatorial systems.

        Generate new YAML content with no combinatorial systems. The new content
        is identical to the old one but combinatorial systems are substituted by
        the description of all the non-combinatorial systems that they generate.
        Moreover, the experiments that use combinatorial systems are updated with
        the new system ids.

        Molecules must be already expanded when calling this function.

        Parameters
        ----------
        yaml_content : dict
            The YAML content as returned by _expand_molecules().

        Returns
        -------
        expanded_content : dict
            The new YAML content with combinatorial systems expanded.

        """
        expanded_content = copy.deepcopy(yaml_content)

        if 'systems' not in expanded_content:
            return expanded_content

        # Check if we have a sequence of experiments or a single one
        try:
            if isinstance(expanded_content['experiments'], list):  # sequence of experiments
                experiment_names = expanded_content['experiments']
            else:
                experiment_names = ['experiments']
        except KeyError:
            experiment_names = []

        # Expand molecules and update molecule ids in experiments
        expanded_content = utils.CombinatorialTree(expanded_content)
        update_nodes_paths = [(e, 'system') for e in experiment_names]
        expanded_content = expanded_content.expand_id_nodes('systems', update_nodes_paths)

        return expanded_content

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
        for exp_name, experiment in utils.listitems(self._experiments):
            if len(self._experiments) > 1:
                output_dir = exp_name

            # Loop over all combinations
            for name, combination in experiment.named_combinations(separator='_', max_name_length=50):
                yield os.path.join(output_dir, name), combination

    # --------------------------------------------------------------------------
    # Parsing and syntax validation
    # --------------------------------------------------------------------------

    @classmethod
    def _validate_options(cls, options):
        """Validate molecules syntax.

        Parameters
        ----------
        options : dict
            A dictionary with the options to validate.

        Returns
        -------
        validated_options : dict
            The validated options.

        Raises
        ------
        YamlParseError
            If the syntax for any option is not valid.

        """
        template_options = cls.DEFAULT_OPTIONS.copy()
        template_options.update(Yank.default_parameters)
        template_options.update(ReplicaExchange.default_parameters)
        template_options.update(utils.get_keyword_args(AbsoluteAlchemicalFactory.__init__))
        openmm_app_type = {'constraints': to_openmm_app}
        try:
            validated_options = utils.validate_parameters(options, template_options, check_unknown=True,
                                                          process_units_str=True, float_to_int=True,
                                                          special_conversions=openmm_app_type)
        except (TypeError, ValueError) as e:
            raise YamlParseError(str(e))
        return validated_options

    @staticmethod
    def _validate_molecules(molecules_description):
        """Validate molecules syntax.

        Parameters
        ----------
        molecules_description : dict
            A dictionary representing molecules.

        Returns
        -------
        validated_molecules : dict
            The validated molecules description.

        Raises
        ------
        YamlParseError
            If the syntax for any molecule is not valid.

        """
        def is_peptide(filepath):
            """Input file is a peptide."""
            if not os.path.isfile(filepath):
                raise YamlParseError('File path does not exist.')
            extension = os.path.splitext(filepath)[1]
            if extension == '.pdb':
                return True
            return False

        def is_small_molecule(filepath):
            """Input file is a small molecule."""
            file_formats = frozenset(['mol2', 'sdf', 'smiles', 'csv'])
            if not os.path.isfile(filepath):
                raise YamlParseError('File path does not exist.')
            extension = os.path.splitext(filepath)[1][1:]
            if extension in file_formats:
                return True
            return False

        validated_molecules = molecules_description.copy()

        # Define molecules Schema
        epik_schema = utils.generate_signature_schema(omt.schrodinger.run_epik,
                                                      update_keys={'select': int},
                                                      exclude_keys=['extract_range'])

        parameters_schema = {  # simple strings are converted to list of strings
            'parameters': And(Use(lambda p: [p] if isinstance(p, str) else p), [str])}
        common_schema = {Optional('leap'): parameters_schema, Optional('openeye'): {'quacpac': 'am1-bcc'},
                         Optional('antechamber'): {'charge_method': Or(str, None)},
                         Optional('epik'): epik_schema}
        molecule_schema = Or(
            utils.merge_dict({'smiles': str}, common_schema),
            utils.merge_dict({'name': str}, common_schema),
            utils.merge_dict({'filepath': is_small_molecule, Optional('select'): Or(int, 'all')},
                             common_schema),
            {'filepath': is_peptide, Optional('select'): Or(int, 'all'),
             Optional('leap'): parameters_schema, Optional('strip_protons'): bool}
        )

        # Schema validation
        for molecule_id, molecule_descr in utils.listitems(molecules_description):
            try:
                validated_molecules[molecule_id] = molecule_schema.validate(molecule_descr)

                # Check OpenEye charges - antechamber consistency
                if 'openeye' in validated_molecules[molecule_id]:
                    if not 'antechamber' in validated_molecules[molecule_id]:
                        raise YamlParseError('Cannot specify openeye charges without antechamber')
                    if validated_molecules[molecule_id]['antechamber']['charge_method'] is not None:
                        raise YamlParseError('Antechamber charge_method must be "null" to read '
                                             'OpenEye charges')

                # Convert epik "select" to "extract_range" which is accepted by run_epik()
                try:
                    extract_range = validated_molecules[molecule_id]['epik'].pop('select')
                    validated_molecules[molecule_id]['epik']['extract_range'] = extract_range
                except (AttributeError, KeyError):
                    pass

                # Create empty parameters list if not specified
                if 'leap' not in validated_molecules[molecule_id]:
                    validated_molecules[molecule_id]['leap'] = {'parameters': []}
            except SchemaError as e:
                raise YamlParseError('Molecule {}: {}'.format(molecule_id, e.autos[-1]))

        return validated_molecules

    @staticmethod
    def _validate_solvents(solvents_description):
        """Validate molecules syntax.

        Parameters
        ----------
        solvents_description : dict
            A dictionary representing solvents.

        Returns
        -------
        validated_solvents : dict
            The validated solvents description.

        Raises
        ------
        YamlParseError
            If the syntax for any solvent is not valid.

        """
        def to_explicit_solvent(str):
            """Check OpenMM explicit solvent."""
            openmm_app = to_openmm_app(str)
            if openmm_app == openmm.app.NoCutoff:
                raise ValueError('Nonbonded method cannot be NoCutoff.')
            return openmm_app

        def to_no_cutoff(str):
            """Check OpenMM implicit solvent or vacuum."""
            openmm_app = to_openmm_app(str)
            if openmm_app != openmm.app.NoCutoff:
                raise ValueError('Nonbonded method must be NoCutoff.')
            return openmm_app

        validated_solvents = solvents_description.copy()

        # Define solvents Schema
        explicit_schema = utils.generate_signature_schema(AmberPrmtopFile.createSystem,
                                update_keys={'nonbonded_method': Use(to_explicit_solvent)},
                                exclude_keys=['implicit_solvent'])
        explicit_schema.update({'clearance': Use(utils.to_unit_validator(unit.angstrom)),
                                Optional('positive_ion'): str, Optional('negative_ion'): str})
        implicit_schema = utils.generate_signature_schema(AmberPrmtopFile.createSystem,
                                update_keys={'implicit_solvent': Use(to_openmm_app),
                                             Optional('nonbonded_method'): Use(to_no_cutoff)},
                                exclude_keys=['rigid_water'])
        vacuum_schema = utils.generate_signature_schema(AmberPrmtopFile.createSystem,
                                update_keys={'nonbonded_method': Use(to_no_cutoff)},
                                exclude_keys=['rigid_water', 'implicit_solvent'])
        solvent_schema = Schema(Or(explicit_schema, implicit_schema, vacuum_schema))

        # Schema validation
        for solvent_id, solvent_descr in utils.listitems(solvents_description):
            try:
                validated_solvents[solvent_id] = solvent_schema.validate(solvent_descr)
            except SchemaError as e:
                raise YamlParseError('Solvent {}: {}'.format(solvent_id, e.autos[-1]))

        return validated_solvents

    @staticmethod
    def _validate_protocols(protocols_description):
        """Validate protocols.

        Parameters
        ----------
        protocols_description : dict
            A dictionary representing protocols.

        Returns
        -------
        validated_protocols : dict
            The validated protocols description.

        Raises
        ------
        YamlParseError
            If the syntax for any protocol is not valid.

        """
        def sort_protocol(protocol):
            """Reorder phases in dictionary to have complex/solvent1 first."""
            sortables = [('complex', 'solvent'), ('solvent1', 'solvent2')]
            for sortable in sortables:
                # Phases names must be unambiguous, they can't contain both names
                phase1 = [(k, v) for k, v in utils.listitems(protocol)
                          if (sortable[0] in k and sortable[1] not in k)]
                phase2 = [(k, v) for k, v in utils.listitems(protocol)
                          if (sortable[1] in k and sortable[0] not in k)]

                # Phases names must be unique
                if len(phase1) == 1 and len(phase2) == 1:
                    return collections.OrderedDict([phase1[0], phase2[0]])

            # Could not find any sortable
            raise SchemaError('Phases must contain either "complex" and "solvent"'
                              'or "solvent1" and "solvent2"')

        validated_protocols = protocols_description.copy()

        # Define protocol Schema
        lambda_list = [And(float, lambda l: 0.0 <= l <= 1.0)]
        alchemical_path_schema = {'alchemical_path': {'lambda_sterics': lambda_list,
                                                      'lambda_electrostatics': lambda_list,
                                                      Optional(str): lambda_list}}
        protocol_schema = Schema(And(
            lambda v: len(v) == 2, {str: alchemical_path_schema},
            Or(collections.OrderedDict, Use(sort_protocol))
        ))

        # Schema validation
        for protocol_id, protocol_descr in utils.listitems(protocols_description):
            try:
                validated_protocols[protocol_id] = protocol_schema.validate(protocol_descr)
            except SchemaError as e:
                raise YamlParseError('Protocol {}: {}'.format(protocol_id, e.autos[-1]))

        return validated_protocols

    def _validate_systems(self, systems_description):
        """Validate systems.

        Receptors, ligands, and solvents must be already loaded. If they are not
        found an exception is raised.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        Returns
        -------
        validated_systems : dict
            The validated systems description.

        Raises
        ------
        YamlParseError
            If the syntax for any experiment is not valid.

        """
        def is_known_molecule(molecule_id):
            if molecule_id in self._db.molecules:
                return True
            raise YamlParseError('Molecule ' + molecule_id + ' is unknown.')

        def is_known_solvent(solvent_id):
            if solvent_id in self._db.solvents:
                return True
            raise YamlParseError('Solvent ' + solvent_id + ' is unknown.')

        def system_files(type):
            def _system_files(files):
                """Paths to amber/gromacs files. Return them in alphabetical
                order of extension [*.inpcrd/gro, *.prmtop/top]."""
                extensions = [os.path.splitext(filepath)[1][1:] for filepath in files]
                correct_type = False
                if type == 'amber':
                    correct_type = sorted(extensions) == ['inpcrd', 'prmtop']
                elif type == 'gromacs':
                    correct_type = sorted(extensions) == ['gro', 'top']
                if not correct_type:
                    raise RuntimeError('Wrong system files type.')
                for filepath in files:
                    if not os.path.isfile(filepath):
                        raise YamlParseError('File path {} does not exist.'.format(filepath))
                return [filepath for (ext, filepath) in sorted(zip(extensions, files))]
            return _system_files

        # Define experiment Schema
        validated_systems = systems_description.copy()
        parameters_schema = {  # simple strings are converted to list of strings
            'parameters': And(Use(lambda p: [p] if isinstance(p, str) else p), [str])}
        system_schema = Schema(Or(
            {'receptor': is_known_molecule, 'ligand': is_known_molecule,
             'solvent': is_known_solvent, Optional('pack', default=False): bool,
             Optional('leap'): parameters_schema},

            {'solute': is_known_molecule, 'solvent1': is_known_solvent,
             'solvent2': is_known_solvent, Optional('leap'): parameters_schema},

            {'phase1_path': Use(system_files('amber')), 'phase2_path': Use(system_files('amber')),
             'ligand_dsl': str, 'solvent': is_known_solvent},

            {'phase1_path': Use(system_files('gromacs')), 'phase2_path': Use(system_files('gromacs')),
             'ligand_dsl': str, 'solvent': is_known_solvent, Optional('gromacs_include_dir'): os.path.isdir}
        ))

        # Schema validation
        for system_id, system_descr in utils.listitems(systems_description):
            try:
                validated_systems[system_id] = system_schema.validate(system_descr)

                # Create empty parameters list if not specified
                if 'leap' not in validated_systems[system_id]:
                    validated_systems[system_id]['leap'] = {'parameters': []}
            except SchemaError as e:
                raise YamlParseError('System {}: {}'.format(system_id, e.autos[-1]))

        return validated_systems

    def _parse_experiments(self, yaml_content):
        """Validate experiments.

        Perform dry run and validate system, protocol and options of every combination.

        Systems and protocols must be already loaded. If they are not found, an exception
        is raised. Experiments options are validated as well.

        Parameters
        ----------
        yaml_content : dict
            The dictionary representing the YAML script loaded by yaml.load()

        Raises
        ------
        YamlParseError
            If the syntax for any experiment is not valid.

        """
        def is_known_system(system_id):
            if system_id in self._db.systems:
                return True
            raise YamlParseError('System ' + system_id + ' is unknown.')

        def is_known_protocol(protocol_id):
            if protocol_id in self._protocols:
                return True
            raise YamlParseError('Protocol ' + protocol_id + ' is unknown')

        # Check if there is a sequence of experiments or a single one
        try:
            if isinstance(yaml_content['experiments'], list):
                self._experiments = {exp_name: utils.CombinatorialTree(yaml_content[exp_name])
                                     for exp_name in yaml_content['experiments']}
            else:
                self._experiments = {'experiments': utils.CombinatorialTree(yaml_content['experiments'])}
        except KeyError:
            self._experiments = {}
            return

        # Restraint schema
        restraint_schema = {'type': str}

        # Define experiment Schema
        experiment_schema = Schema({'system': is_known_system, 'protocol': is_known_protocol,
                                    Optional('options'): Use(YamlBuilder._validate_options),
                                    Optional('restraint'): restraint_schema})

        # Schema validation
        for experiment_id, experiment_descr in self._expand_experiments():
            try:
                experiment_schema.validate(experiment_descr)
            except SchemaError as e:
                raise YamlParseError('Experiment {}: {}'.format(experiment_id, e.autos[-1]))

    # --------------------------------------------------------------------------
    # File paths utilities
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Resuming
    # --------------------------------------------------------------------------

    def _check_resume_experiment(self, experiment_dir, protocol_id):
        """Check if Yank output files already exist.

        Parameters
        ----------
        experiment_dir : str
            The path to the directory that should contain the output files.
        protocol_id : str
            The ID of the protocol used in the experiment.

        Returns
        -------
        bool
            True if NetCDF output files already exist, False otherwise.

        """
        # Build phases .nc file paths
        phase_names = self._protocols[protocol_id].keys()
        phase_paths = [os.path.join(experiment_dir, name + '.nc') for name in phase_names]

        # Look for existing .nc files in the folder
        for phase_path in phase_paths:
            if not (os.path.isfile(phase_path) and os.path.getsize(phase_path) > 0):
                return False
        return True

    def _check_resume(self, check_setup=True, check_experiments=True):
        """Perform dry run to check if we are going to overwrite files.

        If we find folders that YamlBuilder should create we raise an exception
        unless resume_setup or resume_simulation are found, in which case we
        assume we need to use the existing files. We never overwrite files, the
        user is responsible to delete them or move them.

        It's important to check all possible combinations at the beginning to
        avoid interrupting the user simulation after few experiments.

        Parameters
        ----------
        check_setup : bool
            Check if we are going to overwrite setup files (default is True).
        check_experiments : bool
            Check if we are going to overwrite experiment files (default is True).

        Raises
        ------
        YamlParseError
            If files to write already exist and we resuming options are not set.

        """
        # This is hard disk intensive, only process 0 should do it
        if self._mpicomm is not None and self._mpicomm.rank != 0:
            self._mpicomm.barrier()  # Wait for root node to check
            return

        err_msg = ''

        for exp_sub_dir, combination in self._expand_experiments():
            # Determine and validate options
            exp_options = self._determine_experiment_options(combination)

            if check_experiments:
                resume_sim = exp_options['resume_simulation']
                experiment_dir = self._get_experiment_dir(exp_options, exp_sub_dir)
                if not resume_sim and self._check_resume_experiment(experiment_dir,
                                                                    combination['protocol']):
                    err_msg = 'experiment files in directory {}'.format(experiment_dir)
                    solving_option = 'resume_simulation'

            if check_setup and err_msg == '':
                resume_setup = exp_options['resume_setup']
                system_id = combination['system']

                # Check system and molecule setup dirs
                self._db.setup_dir = self._get_setup_dir(exp_options)
                is_sys_setup, is_sys_processed = self._db.is_system_setup(system_id)
                if is_sys_processed and not resume_setup:
                    system_dir = os.path.dirname(
                        self._db.get_system_files_paths(system_id)[0].position_path)
                    err_msg = 'system setup directory {}'.format(system_dir)
                elif not is_sys_setup:  # then this must go through the pipeline
                    try:  # binding free energy system
                        receptor_id = self._db.systems[system_id]['receptor']
                        ligand_id = self._db.systems[system_id]['ligand']
                        molecule_ids = [receptor_id, ligand_id]
                    except KeyError:  # partition/solvation free energy system
                        molecule_ids = [self._db.systems[system_id]['solute']]
                    for molecule_id in molecule_ids:
                        is_processed = self._db.is_molecule_setup(molecule_id)[1]
                        if is_processed and not resume_setup:
                            err_msg = 'molecule {} file'.format(molecule_id)
                            break

                if err_msg != '':
                    solving_option = 'resume_setup'

            # Check for errors
            if err_msg != '':
                err_msg += (' already exists; cowardly refusing to proceed. Move/delete '
                            'directory or set {} options').format(solving_option)
                raise YamlParseError(err_msg)

        # Signal resume to non-root other nodes
        if self._mpicomm is not None:
            self._mpicomm.barrier()

    # --------------------------------------------------------------------------
    # Experiment setup and execution
    # --------------------------------------------------------------------------

    def _setup_experiments(self):
        """Set up all experiments without running them.

        IMPORTANT: This does not check if we are about to overwrite files, nor it
        cd into the script directory! Use setup_experiments() for that.

        """
        # TODO parallelize setup
        # Only root node performs setup
        if self._mpicomm is not None and self._mpicomm.rank != 0:
            debug_msg = 'Node {}/{}: MPI barrier'.format(self._mpicomm.rank,
                                                         self._mpicomm.size)
            logger.debug(debug_msg + ' - waiting for the setup to be completed.')
            self._mpicomm.barrier()
            return

        for _, experiment in self._expand_experiments():
            # Set database path
            exp_opts = self._determine_experiment_options(experiment)
            self._db.setup_dir = self._get_setup_dir(exp_opts)

            # Force system and molecules setup
            system_id = experiment['system']
            sys_descr = self._db.systems[system_id]  # system description
            try:
                try:  # binding free energy system
                    components = (sys_descr['receptor'], sys_descr['ligand'], sys_descr['solvent'])
                except KeyError:  # partition/solvation free energy system
                    components = (sys_descr['solute'], sys_descr['solvent1'], sys_descr['solvent2'])
                logger.info('Setting up the systems for {}, {} and {}'.format(*components))
                self._db.get_system(system_id)
            except KeyError:  # system files are given directly by the user
                pass

        # Signal resume to child nodes
        if self._mpicomm is not None:
            debug_msg = 'Node {}/{}: MPI barrier'.format(self._mpicomm.rank,
                                                         self._mpicomm.size)
            logger.debug(debug_msg + ' - signal completed setup.')
            self._mpicomm.barrier()

    @staticmethod
    def _opencl_device_support_precision(precision_model):
        """
        Check if this device supports the given precision model for OpenCL platform.

        Some OpenCL devices do not support double precision. This offers a test
        function.

        Returns
        -------
        is_supported : bool
            True if this device supports double precision for OpenCL, False
            otherwise.

        """
        opencl_platform = openmm.Platform.getPlatformByName('OpenCL')

        # Platforms are singleton so we need to store
        # the old precision model before modifying it
        old_precision = opencl_platform.getPropertyDefaultValue('OpenCLPrecision')

        # Test support by creating a toy context
        opencl_platform.setPropertyDefaultValue('OpenCLPrecision', precision_model)
        system = openmm.System()
        system.addParticle(1.0 * unit.amu)  # system needs at least 1 particle
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        try:
            context = openmm.Context(system, integrator, opencl_platform)
            is_supported = True
        except Exception:
            is_supported = False
        else:
            del context
        del integrator

        # Restore old precision
        opencl_platform.setPropertyDefaultValue('OpenCLPrecision', old_precision)

        return is_supported

    @staticmethod
    def _determine_fastest_platform():
        """
        Return the fastest available platform.

        Returns
        -------
        platform : simtk.openmm.Platform
           The fastest available platform.

        """
        platform_speeds = np.array([openmm.Platform.getPlatform(i).getSpeed()
                                    for i in range(openmm.Platform.getNumPlatforms())])
        fastest_platform_id = int(np.argmax(platform_speeds))
        platform = openmm.Platform.getPlatform(fastest_platform_id)
        return platform

    def _configure_platform(self, platform_name, platform_precision):
        """
        Configure the platform to be used for simulation for the given precision.

        Parameters
        ----------
        platform_name : str
            The name of the platform to be used for execution. If 'fastest',
            the fastest available platform is used.
        platform_precision : str or None
            The precision to be used. If 'auto' the default value is used,
            which is always mixed precision except for Reference that only
            supports double precision, and OpenCL when the device supports
            only single precision. If None, the precision mode won't be
            set, so OpenMM default value will be used which is always
            'single' for CUDA and OpenCL.

        Returns
        -------
        platform : simtk.openmm.Platform
           The configured platform.

        Raises
        ------
        RuntimeError
            If the given precision model selected is not compatible with the
            platform.

        """
        # Determine the platform to configure
        if platform_name == 'fastest':
            platform = self._determine_fastest_platform()
            platform_name = platform.getName()
        else:
            platform = openmm.Platform.getPlatformByName(platform_name)

        # Use only a single CPU thread if we are using the CPU platform.
        # TODO: Since there is an environment variable that can control this,
        # TODO: we may want to avoid doing this.
        if platform_name == 'CPU' and self._mpicomm is not None:
            logger.debug("Setting 'CpuThreads' to 1 because MPI is active.")
            platform.setPropertyDefaultValue('CpuThreads', '1')

        # If user doesn't specify precision, determine default value
        if platform_precision == 'auto':
            if platform_name == 'CUDA':
                platform_precision = 'mixed'
            elif platform_name == 'OpenCL':
                if self._opencl_device_support_precision('mixed'):
                    platform_precision = 'mixed'
                else:
                    logger.info("This device does not support double precision for OpenCL. "
                                "Setting OpenCL precision to 'single'")
                    platform_precision = 'single'
            elif platform_name == 'Reference' or platform_name == 'CPU':
                platform_precision = None  # leave OpenMM default precision

        # Set platform precision
        if platform_precision is not None:
            logger.info("Setting {} platform to use precision model "
                        "'{}'.".format(platform_name, platform_precision))
            if platform_name == 'CUDA':
                platform.setPropertyDefaultValue('CudaPrecision', platform_precision)
            elif platform_name == 'OpenCL':
                # Some OpenCL devices do not support double precision so we need to test it
                if self._opencl_device_support_precision(platform_precision):
                    platform.setPropertyDefaultValue('OpenCLPrecision', platform_precision)
                else:
                    raise RuntimeError('This device does not support double precision for OpenCL.')
            elif platform_name == 'Reference':
                if platform_precision != 'double':
                    raise RuntimeError("Reference platform does not support precision model '{}';"
                                       "only 'double' is supported.".format(platform_precision))
            elif platform_name == 'CPU':
                if platform_precision != 'mixed':
                    raise RuntimeError("CPU platform does not support precision model '{}';"
                                       "only 'mixed' is supported.".format(platform_precision))
            else:  # This is an unkown platform
                raise RuntimeError("Found unknown platform '{}'.".format(platform_name))

        return platform

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
        sys_descr = self._db.systems[experiment['system']]  # system description

        # Molecules section data
        try:
            try:  # binding free energy
                molecule_ids = [sys_descr['receptor'], sys_descr['ligand']]
            except KeyError:  # partition/solvation free energy
                molecule_ids = [sys_descr['solute']]
            mol_section = {mol_id: self._raw_yaml['molecules'][mol_id]
                           for mol_id in molecule_ids}

            # Copy to avoid modifying _raw_yaml when updating paths
            mol_section = copy.deepcopy(mol_section)
        except KeyError:  # user provided directly system files
            mol_section = {}

        # Solvents section data
        try:  # binding free energy
            solvent_ids = [sys_descr['solvent']]
        except KeyError:  # partition/solvation free energy
            solvent_ids = [sys_descr['solvent1'], sys_descr['solvent2']]
        sol_section = {sol_id: self._raw_yaml['solvents'][sol_id]
                       for sol_id in solvent_ids}

        # Systems section data
        system_id = experiment['system']
        sys_section = {system_id: copy.deepcopy(self._raw_yaml['systems'][system_id])}

        # Protocols section data
        protocol_id = experiment['protocol']
        prot_section = {protocol_id: self._raw_yaml['protocols'][protocol_id]}

        # We pop the options section in experiment and merge it to the general one
        exp_section = experiment.copy()
        opt_section = self._raw_yaml['options'].copy()
        opt_section.update(exp_section.pop('options', {}))

        # Convert relative paths to new script directory
        for molecule in utils.listvalues(mol_section):
            if 'filepath' in molecule and not os.path.isabs(molecule['filepath']):
                molecule['filepath'] = os.path.relpath(molecule['filepath'], yaml_dir)

        try:  # systems for which user has specified directly system files
            for phase in ['phase2_path', 'phase1_path']:
                for path in sys_section[system_id][phase]:
                    sys_section[system_id][path] = os.path.relpath(path, yaml_dir)
        except KeyError:  # system went through pipeline
            pass

        try:  # output directory
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
        if mol_section:
            yaml_content += yaml.dump({'molecules': mol_section},  **dump_options)
        yaml_content += yaml.dump({'solvents': sol_section},  **dump_options)
        yaml_content += yaml.dump({'systems': sys_section},  **dump_options)
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
        for phase_name, phase in utils.listitems(self._protocols[protocol_id]):
            # Separate lambda variables names from their associated lists
            lambdas, values = zip(*utils.listitems(phase['alchemical_path']))

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
        protocol_id = experiment['protocol']
        exp_name = 'experiments' if experiment_dir == '' else os.path.basename(experiment_dir)

        # Get and validate experiment sub-options
        exp_opts = self._determine_experiment_options(experiment)
        yank_opts = self._isolate_yank_options(exp_opts)

        # Set database path
        self._db.setup_dir = self._get_setup_dir(exp_opts)

        # Create directory and determine if we need to resume the simulation
        results_dir = self._get_experiment_dir(exp_opts, experiment_dir)
        resume = os.path.isdir(results_dir)
        if self._mpicomm is None or self._mpicomm.rank == 0:
            if not resume:
                os.makedirs(results_dir)
            else:
                resume = self._check_resume_experiment(results_dir, protocol_id)
        if self._mpicomm:  # process 0 send result to other processes
            resume = self._mpicomm.bcast(resume, root=0)

        # Configure logger for this experiment
        utils.config_root_logger(exp_opts['verbose'], os.path.join(results_dir, exp_name + '.log'),
                                 self._mpicomm)

        # Configure platform and precision
        platform = self._configure_platform(exp_opts['platform'], exp_opts['precision'])

        # Initialize simulation
        yank = Yank(results_dir, mpicomm=self._mpicomm, platform=platform, **yank_opts)

        if resume:
            yank.resume()
        else:
            if self._mpicomm is None or self._mpicomm.rank == 0:
                system_id = experiment['system']

                # Export YAML file for reproducibility
                self._generate_yaml(experiment, os.path.join(results_dir, exp_name + '.yaml'))

                # Get molecule resname for alchemical atom selection
                if self._db.is_system_setup(system_id)[1]:  # system went through pipeline
                    try:  # binding free energy calculation
                        alchemical_molecule_id = self._db.systems[system_id]['ligand']
                    except KeyError:  # partition/solvation free energy calculation
                        alchemical_molecule_id = self._db.systems[system_id]['solute']

                    # Make sure that molecule filepath points to the mol2 file
                    self._db.is_molecule_setup(alchemical_molecule_id)
                    ligand_descr = self._db.molecules[alchemical_molecule_id]
                    ligand_dsl = utils.Mol2File(ligand_descr['filepath']).resname
                    if ligand_dsl is None:
                        ligand_dsl = 'MOL'
                    ligand_dsl = 'resname ' + ligand_dsl
                else:  # system files given directly by user
                    ligand_dsl = self._db.systems[system_id]['ligand_dsl']
                logger.debug('DSL string for the ligand: "{}"'.format(ligand_dsl))

                # Determine complex and solvent phase solvents
                try:  # binding free energy calculations
                    solvent_ids = [self._db.systems[system_id]['solvent'],
                                   self._db.systems[system_id]['solvent']]
                except KeyError:  # partition/solvation free energy calculations
                    solvent_ids = [self._db.systems[system_id]['solvent1'],
                                   self._db.systems[system_id]['solvent2']]

                # Get protocols as list of AlchemicalStates
                alchemical_paths = self._get_alchemical_paths(protocol_id)
                system_files_paths = self._db.get_system(system_id)
                gromacs_include_dir = self._db.systems[system_id].get('gromacs_include_dir', None)

                # Prepare Yank arguments
                phases = [None, None]
                for i, phase_name in enumerate(self._protocols[protocol_id]):
                    # self._protocols[protocol_id] is an OrderedDict so phases are in the
                    # correct order (e.g. [complex, solvent] or [solvent1, solvent2])
                    solvent_id = solvent_ids[i]
                    positions_file_path = system_files_paths[i].position_path
                    topology_file_path = system_files_paths[i].topology_path
                    system_options = utils.merge_dict(self._db.solvents[solvent_id], exp_opts)

                    logger.info("Reading phase {}".format(phase_name))
                    phases[i] = pipeline.prepare_phase(positions_file_path, topology_file_path, ligand_dsl,
                                                       system_options, gromacs_include_dir=gromacs_include_dir)
                    phases[i].name = phase_name
                    phases[i].protocol = alchemical_paths[phase_name]

                for phase in phases:
                    if len(phase.atom_indices['ligand']) == 0:
                        raise RuntimeError('DSL string "{}" did not select any atom for '
                                           'the ligand in phase {}.'.format(ligand_dsl, phase.name))

                # Dump analysis script
                analysis = [[phases[0].name, 1], [phases[1].name, -1]]
                analysis_script_path = os.path.join(results_dir, 'analysis.yaml')
                with open(analysis_script_path, 'w') as f:
                    yaml.dump(analysis, f)

                # Create thermodynamic state
                thermodynamic_state = ThermodynamicState(temperature=exp_opts['temperature'],
                                                         pressure=exp_opts['pressure'])

                # Determine restraint
                try:
                    restraint_type = experiment['restraint']['type']
                except (KeyError, TypeError):  # restraint unspecified or None
                    restraint_type = None

                # Create new simulation
                yank.create(thermodynamic_state, *phases, restraint_type=restraint_type)

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
