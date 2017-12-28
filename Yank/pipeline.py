#!/usr/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Pipeline
========

Utility functions to help setting up Yank configurations.

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import os
import re
import sys
import copy
import inspect
import logging
import itertools
import collections

import mdtraj
import numpy as np
import openmmtools as mmtools
import openmoltools as moltools
from pdbfixer import PDBFixer
from simtk import openmm, unit
from simtk.openmm.app import PDBFile

from . import utils, mpi

logger = logging.getLogger(__name__)


# ==============================================================================
# Utility functions
# ==============================================================================

def compute_squared_distances(molecule1_positions, molecule2_positions):
    """Compute the squared distances between the atoms of two molecules.

    All the positions must be expressed in the same unit of measure.

    Parameters
    ----------
    molecule1_positions : numpy.ndarray
        An Nx3 array where, N is the number of atoms, containing the positions of
        the atoms of molecule1.
    molecule2_positions : numpy.ndarray
        An Mx3 array where, M is the number of atoms, containing the positions of
        the atoms of the molecule2.

    Returns
    -------
    squared_distances : numpy.ndarray
        An NxM array of squared distances. distances_squared[i][j] is the squared
        distance between atom i of molecule1 and atom j of molecule 2.

    """
    squared_distances = np.array([((molecule2_positions - atom_pos)**2).sum(1)
                                  for atom_pos in molecule1_positions])
    return squared_distances


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
        The minimum distance between ``mol_positions`` and the other set of positions

    """
    for argmol_positions in args:
        # Compute squared distances.  Each row is an array of distances
        # from a mol_positions atom to all argmol_positions atoms.
        distances2 = compute_squared_distances(mol_positions, argmol_positions)

        # Find closest atoms and their distance
        min_idx = np.unravel_index(distances2.argmin(), distances2.shape)
        try:
            min_dist = min(min_dist, np.sqrt(distances2[min_idx]))
        except UnboundLocalError:
            min_dist = np.sqrt(distances2[min_idx])
    return min_dist


def compute_min_max_dist(mol_positions, *args):
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
    >>> min_dist, max_dist = compute_min_max_dist(mol1_pos, mol2_pos, mol3_pos)
    >>> min_dist == np.linalg.norm(mol1_pos[1] - mol2_pos[0])
    True
    >>> max_dist == np.linalg.norm(mol1_pos[1] - mol3_pos[1])
    True

    """
    min_dist = None

    for argmol_positions in args:
        # Compute squared distances of all atoms. Each row is an array of distances
        # from an atom in argmol_positions to all the atoms in mol_positions.
        distances2 = compute_squared_distances(argmol_positions, mol_positions)

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


def compute_radius_of_gyration(positions):
    """
    Compute the radius of gyration of the specified coordinate set.

    Parameters
    ----------
    positions : simtk.unit.Quantity with units compatible with angstrom
       The coordinate set (natoms x 3) for which the radius of gyration is to be computed.

    Returns
    -------
    radius_of_gyration : simtk.unit.Quantity with units compatible with angstrom
       The radius of gyration

    """
    unit = positions.unit

    # Get dimensionless receptor positions.
    x = positions / unit

    # Get dimensionless restrained atom coordinate.
    xref = x.mean(0)
    xref = np.reshape(xref, (1,3)) # (1,3) array

    # Compute distances from restrained atom.
    natoms = x.shape[0]
    # Distances[i] is the distance from the centroid to particle i
    distances = np.sqrt(((x - np.tile(xref, (natoms, 1)))**2).sum(1))

    # Compute std dev of distances from restrained atom.
    radius_of_gyration = distances.std() * unit

    return radius_of_gyration


def compute_net_charge(system, atom_indices):
    """Compute the total net charge of a subset of atoms in the system.

    Parameters
    ----------
    system : simtk.openmm.System
        The system object containing the atoms of interest.
    atom_indices : list of int
        Indices of the atoms of interest.

    Returns
    -------
    net_charge : int
        Total net charge as the sum of the partial charges of the atoms.

    """
    atom_indices = set(atom_indices)  # convert to set to speed up searching
    net_charge = 0.0 * unit.elementary_charge
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if isinstance(force, openmm.NonbondedForce):
            for particle_index in range(force.getNumParticles()):
                if particle_index in atom_indices:
                    net_charge += force.getParticleParameters(particle_index)[0]
                    atom_indices.remove(particle_index)
    assert len(atom_indices) == 0
    net_charge = int(round(net_charge / unit.elementary_charge))
    return net_charge


def find_alchemical_counterions(system, topography, region_name):
    """Return the atom indices of the ligand or solute counter ions.

    In periodic systems, the solvation box needs to be neutral, and
    if the decoupled molecule is charged, it will cause trouble. This
    can be used to find a set of ions in the system that neutralize
    the molecule, so that the solvation box will remain neutral all
    the time.

    Parameters
    ----------
    system : simtk.openmm.System
        The system object containing the atoms of interest.
    topography : yank.Topography
        The topography object holding the indices of the ions and the
        ligand (for binding free energy) or solute (for transfer free
        energy).
    region_name : str
        The region name in the topography (e.g. "ligand_atoms") for
        which to find counter ions.

    Returns
    -------
    counterions_indices : list of int
        The list of atom indices in the system of the counter ions
        neutralizing the region.

    Raises
    ------
    ValueError
        If the topography region has no atoms, or if it impossible
        to neutralize the region with the ions in the system.

    """
    # Check whether we need to find counterions of ligand or solute.
    atom_indices = getattr(topography, region_name)
    if len(atom_indices) == 0:
        raise ValueError("Cannot find counterions for region {}. "
                         "The region has no atoms.")

    # If the net charge of alchemical atoms is 0, we don't need counterions.
    mol_net_charge = compute_net_charge(system, atom_indices)
    logger.debug('{} net charge: {}'.format(region_name, mol_net_charge))
    if mol_net_charge == 0:
        return []

    # Find net charge of all ions in the system.
    ions_net_charges = {ion_id: compute_net_charge(system, [ion_id])
                        for ion_id in topography.ions_atoms}
    topology = topography.topology
    ions_names_charges = [(topology.atom(ion_id).residue.name, ions_net_charges[ion_id])
                          for ion_id in ions_net_charges]
    logger.debug('Ions net charges: {}'.format(ions_names_charges))

    # Find minimal subset of counterions whose charges sums to -mol_net_charge.
    for n_ions in range(1, len(ions_net_charges) + 1):
        for ion_subset in itertools.combinations(ions_net_charges.items(), n_ions):
            counterions_indices, counterions_charges = zip(*ion_subset)
            if sum(counterions_charges) == -mol_net_charge:
                return counterions_indices

    # We couldn't find any subset of counterions neutralizing the region.
    raise ValueError('Impossible to find a solution for region {}. '
                     'Net charge: {}, system ions: {}.'.format(
        region_name, mol_net_charge, ions_names_charges))


# See Amber manual Table 4.1 http://ambermd.org/doc12/Amber15.pdf
_OPENMM_TO_TLEAP_PBRADII = {'HCT': 'mbondi', 'OBC1': 'mbondi2', 'OBC2': 'mbondi2',
                            'GBn': 'bondi', 'GBn2': 'mbondi3'}


def get_leap_recommended_pbradii(implicit_solvent):
    """Return the recommended PBradii setting for LeAP.

    Parameters
    ----------
    implicit_solvent : str
        The implicit solvent model.

    Returns
    -------
    pbradii : str or object
        The LeAP recommended PBradii for the model.

    Raises
    ------
    ValueError
        If the implicit solvent model is not supported by OpenMM.

    Examples
    --------
    >>> get_leap_recommended_pbradii('OBC2')
    'mbondi2'
    >>> from simtk.openmm.app import HCT
    >>> get_leap_recommended_pbradii(HCT)
    'mbondi'

    """
    try:
        return _OPENMM_TO_TLEAP_PBRADII[str(implicit_solvent)]
    except KeyError:
        raise ValueError('Implicit solvent {} is not supported.'.format(implicit_solvent))


_NONPERIODIC_NONBONDED_METHODS = [openmm.app.NoCutoff, openmm.app.CutoffNonPeriodic]


def create_system(parameters_file, box_vectors, create_system_args, system_options):
    """Create and return an OpenMM system.

    Parameters
    ----------
    parameters_file : simtk.openmm.app.AmberPrmtopFile or GromacsTopFile
        The file used to create they system.
    box_vectors : list of Vec3
        The default box vectors of the system will be set to this value.
    create_system_args : dict of str
        The kwargs accepted by the ``createSystem()`` function of the ``parameters_file``.
    system_options : dict
        The kwargs to forward to ``createSystem()``.

    Returns
    -------
    system : simtk.openmm.System
        The system created.

    """
    # Prepare createSystem() options
    # OpenMM adopts camel case convention so we need to change the options format.
    # Then we filter system options according to specific createSystem() args
    system_options = {utils.underscore_to_camelcase(key): value
                      for key, value in system_options.items()}
    system_options = {arg: system_options[arg] for arg in create_system_args
                      if arg in system_options}

    # Determine if this will be an explicit or implicit solvent simulation.
    if box_vectors is not None:
        is_periodic = True
    else:
        is_periodic = False

    # Adjust nonbondedMethod
    # TODO: Ensure that selected method is appropriate.
    if 'nonbondedMethod' not in system_options:
        if is_periodic:
            system_options['nonbondedMethod'] = openmm.app.CutoffPeriodic
        else:
            system_options['nonbondedMethod'] = openmm.app.NoCutoff

    # Check for solvent configuration inconsistencies
    # TODO: Check to make sure both files agree on explicit/implicit.
    err_msg = ''
    nonbonded_method = system_options['nonbondedMethod']
    if is_periodic:
        if 'implicitSolvent' in system_options and system_options['implicitSolvent'] is not None:
            err_msg = 'Found periodic box in positions file and implicitSolvent specified.'
        if nonbonded_method in _NONPERIODIC_NONBONDED_METHODS:
            err_msg = ('Found periodic box in positions file but '
                       'nonbondedMethod is {}'.format(nonbonded_method))
    else:
        if nonbonded_method not in _NONPERIODIC_NONBONDED_METHODS:
            err_msg = ('nonbondedMethod {} is periodic but could not '
                       'find periodic box in positions file.'.format(nonbonded_method))
    if len(err_msg) != 0:
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Create system and update box vectors (if needed)
    system = parameters_file.createSystem(removeCMMotion=False, **system_options)
    if is_periodic:
        system.setDefaultPeriodicBoxVectors(*box_vectors)

    return system


def read_system_files(positions_file_path, parameters_file_path, system_options,
                      gromacs_include_dir=None):
    """Create a Yank arguments for a phase from system files.

    Parameters
    ----------
    positions_file_path : str
        Path to system position file (e.g. 'complex.inpcrd/.gro/.pdb').
    parameters_file_path : str
        Path to system parameters file (e.g. 'complex.prmtop/.top/.xml').
    system_options : dict
        ``system_options[phase]`` is a a dictionary containing options to
        pass to ``createSystem()``. If the parameters file is an OpenMM
        system in XML format, this will be ignored.
    gromacs_include_dir : str, optional
        Path to directory in which to look for other files included
        from the gromacs top file.

    Returns
    -------
    system : simtk.openmm.System
        The OpenMM System built from the given files.
    topology : openmm.app.Topology
        The OpenMM Topology built from the given files.
    sampler_state : openmmtools.states.SamplerState
        The sampler state containing the positions of the atoms.

    """
    # Load system files
    parameters_file_extension = os.path.splitext(parameters_file_path)[1]

    # Read OpenMM XML and PDB files.
    if parameters_file_extension == '.xml':
        logger.debug("xml: {}".format(parameters_file_path))
        logger.debug("pdb: {}".format(positions_file_path))

        positions_file = openmm.app.PDBFile(positions_file_path)
        parameters_file = positions_file  # Needed for topology.
        with open(parameters_file_path, 'r') as f:
            serialized_system = f.read()

        system = openmm.XmlSerializer.deserialize(serialized_system)

    # Read Amber prmtop and inpcrd files.
    elif parameters_file_extension == '.prmtop':
        logger.debug("prmtop: {}".format(parameters_file_path))
        logger.debug("inpcrd: {}".format(positions_file_path))

        parameters_file = openmm.app.AmberPrmtopFile(parameters_file_path)
        positions_file = openmm.app.AmberInpcrdFile(positions_file_path)
        box_vectors = positions_file.boxVectors
        create_system_args = set(inspect.getargspec(openmm.app.AmberPrmtopFile.createSystem).args)

        system = create_system(parameters_file, box_vectors, create_system_args, system_options)

    # Read Gromacs top and gro files.
    elif parameters_file_extension == '.top':
        logger.debug("top: {}".format(parameters_file_path))
        logger.debug("gro: {}".format(positions_file_path))

        positions_file = openmm.app.GromacsGroFile(positions_file_path)

        # gro files must contain box vectors, so we must determine whether system
        # is non-periodic or not from provided nonbonded options
        # WARNING: This uses the private API for GromacsGroFile, and may break.
        if ('nonbonded_method' in system_options and
                system_options['nonbonded_method'] in _NONPERIODIC_NONBONDED_METHODS):
            logger.info('nonbonded_method = {}, so removing periodic box vectors '
                        'from gro file'.format(system_options['nonbonded_method']))
            for frame, box_vectors in enumerate(positions_file._periodicBoxVectors):
                positions_file._periodicBoxVectors[frame] = None

        box_vectors = positions_file.getPeriodicBoxVectors()
        parameters_file = openmm.app.GromacsTopFile(parameters_file_path,
                                                    periodicBoxVectors=box_vectors,
                                                    includeDir=gromacs_include_dir)
        create_system_args = set(inspect.getargspec(openmm.app.GromacsTopFile.createSystem).args)

        system = create_system(parameters_file, box_vectors, create_system_args, system_options)

    # Unsupported file format.
    else:
        raise ValueError('Unsupported format for parameter file {}'.format(parameters_file_extension))

    # Store numpy positions and create SamplerState.
    positions = positions_file.getPositions(asNumpy=True)
    sampler_state = mmtools.states.SamplerState(positions=positions)

    # Check to make sure number of atoms match between prmtop and inpcrd.
    n_atoms_system = system.getNumParticles()
    n_atoms_positions = positions.shape[0]
    if n_atoms_system != n_atoms_positions:
        err_msg = "Atom number mismatch: {} has {} atoms; {} has {} atoms.".format(
            parameters_file_path, n_atoms_system, positions_file_path, n_atoms_positions)
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    return system, parameters_file.topology, sampler_state


# =============================================================================
# SETUP PIPELINE UTILITY FUNCTIONS
# =============================================================================

# Map the OpenMM-style name for a solvent to the tleap
# name compatible with the solvateBox command.
_OPENMM_LEAP_SOLVENT_MODELS_MAP = {
    'tip3p': 'TIP3PBOX',
    'tip3pfb': 'TIP3PFBOX',
    'tip4pew': 'TIP4PEWBOX',
    'tip5p': 'TIP5PBOX',
    'spce': 'SPCBOX',
}

# Map the OpenMM-style name for solvent to the tleap
# name for a list of files which would enable the
# solvent model to work. Servers as error checking,
# but is not foolproof

_OPENMM_LEAP_SOLVENT_FILES_MAP = {
    'tip3p': 'leaprc.water.tip3p',
    'tip3pfb': 'leaprc.water.tip3p',
    'tip4pew': 'leaprc.water.tip4pew',
    'tip5p': 'leaprc.water.tip4pew',  # Enables the EP atom type
    'spce': 'leaprc.water.spce',
}


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
        Rq = mmtools.mcmc.MCRotationMove.generate_random_rotation_matrix()
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
    min_dist, max_dist = compute_min_max_dist(mol1_pos, mol2_pos)
    while min_dist < min_distance or max_distance <= max_dist:
        # Select random atom of fixed molecule and use it to propose new x0 position
        mol1_atom_idx = np.random.random_integers(0, len(mol1_pos) - 1)
        translation = mol1_pos[mol1_atom_idx] + max_distance * np.random.randn(3) - x0

        # Generate random rotation matrix
        Rq = mmtools.mcmc.MCRotationMove.generate_random_rotation_matrix()

        # Apply random transformation and test
        x = ((Rq * np.matrix(mol2_pos - x0).T).T + x0).A + translation
        min_dist, max_dist = compute_min_max_dist(mol1_pos, x)

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
    """
    Remove all hydrogens from PDB file and save the result.

    Input and output file cannot be the same file

    Parameters
    ----------
    input_file_path : str
        Full file path to the file to read, including extensions
    output_file_path : str
        Full file path to the file to save, including extensions
    """
    output_file = open(output_file_path, 'w')
    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            if not (line[:6] == 'ATOM  ' and (line[12] == 'H' or line[13] == 'H')):
                output_file.write(line)
    output_file.close()

# For mutate_protein
_three_letter_code = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR'
}

_one_letter_code = dict()
for one_letter in _three_letter_code.keys():
    three_letter = _three_letter_code[one_letter]
    _one_letter_code[three_letter] = one_letter


def decompose_mutation(mutation):
    match = re.match('(\D)(\d+)(\D)', mutation)
    try:
        original_residue_name = _three_letter_code[match.group(1)]
        residue_index = int(match.group(2))
        mutated_residue_name = _three_letter_code[match.group(3)]
    except AttributeError:
        error_string = 'Mutation "{}" could not be parsed! '.format(mutation)
        error_string += 'Should be of form {single letter}{integer}{another single letter}'
        raise ValueError(error_string)
    return original_residue_name, residue_index, mutated_residue_name


def generate_pdbfixer_mutation_code(original_residue_name, residue_index, mutated_residue_name):
    return '{0:s}-{1:d}-{2:s}'.format(original_residue_name, residue_index, mutated_residue_name)


def apply_pdbfixer(input_file_path, output_file_path, directives):
    """
    Apply PDBFixer to make changes to the specified molecule.

    Single mutants are supported in the form "T315I"
    Double mutants are supported in the form "L858R/T790M"

    The string "WT" still pushes the molecule through PDBFixer, but makes no mutations.
    This is useful for testing.

    Original PDB file numbering scheme is used.

    Currently, only PDB files are supported.
    pdbfixer is used to make the mutations

    Parameters
    ----------
    input_file_path : str
        Full file path to the file to read, including extensions
    output_file_path : str
        Full file path to the file to save, including extensions
    directives : dict
        Dict containing directives for PDBFixer.
    """
    # Make a copy since we will delete from the dictionary to validate
    directives = copy.deepcopy(directives)

    # Create a PDBFixer object
    fixer = PDBFixer(input_file_path)

    # TODO: Refactor this to minimize code duplication

    # Extract pH if specified
    pH = 7.4
    option = 'ph'
    if option in directives:
        try:
            pH = float(directives[option])
            logger.info('pdbfixer: Will use user-specified pH {}'.format(pH))
        except:
            raise ValueError("'ph' must be a floating-point number")
        # Delete the key once we've processed it
        del directives[option]

    # Set default atom addition method
    if 'add_missing_atoms' not in directives:
        directives['add_missing_atoms'] = 'heavy'

    # Add missing residues
    option = 'add_missing_residues'
    fixer.missingResidues = {}
    if option in directives:
        value = directives[option]
        # Validate options
        allowed_values = ['yes', 'no']
        if value not in allowed_values:
            raise ValueError("'{}' must be one of {}".format(option, allowed_values))
        # Apply options
        if value == 'yes':
            fixer.findMissingResidues()
            logger.info('pdbfixer: Will add missing residues specified in SEQRES')
        # Delete the key once we've processed it
        del directives[option]

    # Apply mutations
    option = 'apply_mutations'
    if option in directives:
        value = directives[option]
        # Validate options
        if type(value) is not dict:
            raise ValueError("'apply_mutations' must have a 'mutations:' node")
        # Extract chain id
        chain_id = None
        if 'chain_id' in value:
            chain_id = value['chain_id']
            if chain_id == 'none':
                chain_id = None
        # Extract mutations
        mutations = value['mutations']
        # Convert mutations to PDBFixer format
        if mutations != 'WT':
            pdbfixer_mutations = [generate_pdbfixer_mutation_code(*decompose_mutation(mutation))
                                  for mutation in mutations.split('/')]
            logger.info('pdbfixer: Will make mutations {} to chain_id {}.'.format(pdbfixer_mutations, chain_id))
            fixer.applyMutations(pdbfixer_mutations, chain_id)
        else:
            logger.info('pdbfixer: No mutations will be applied since "WT" specified.')
        # Delete the key once we've processed it
        del directives[option]

    # Replace nonstandard residues
    option = 'replace_nonstandard_residues'
    if option in directives:
        value = directives[option]
        # Validate options
        allowed_values = ['yes', 'no']
        if value not in allowed_values:
            raise ValueError("'{}' must be one of {}".format(option, allowed_values))
        # Apply options
        if value == 'yes':
            logger.info('pdbfixer: Will replace nonstandard residues.')
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
        # Delete the key once we've processed it
        del directives[option]

    # Remove heterogens
    option = 'remove_heterogens'
    if option in directives:
        value = directives[option]
        # Validate options
        allowed_values = ['all', 'water', 'none']
        if value not in allowed_values:
            raise ValueError("'{}' must be one of {}".format(option, allowed_values))
        # Apply options
        if value == 'water':
            logger.info('pdbfixer: Will remove heterogens, retaining water.')
            fixer.removeHeterogens(keepWater=True)
        elif value == 'all':
            logger.info('pdbfixer: Will remove heterogens, discarding water.')
            fixer.removeHeterogens(keepWater=False)
        # Delete the key once we've processed it
        del directives[option]

    # Add missing residues
    option = 'add_missing_atoms'
    if option in directives:
        value = directives[option]
        # Validate options
        allowed_values = ['all', 'heavy', 'hydrogens', 'none']
        if value not in allowed_values:
            raise ValueError("'{}' must be one of {}".format(option, allowed_values))
        # Apply options
        fixer.findMissingAtoms()
        if value not in ('all', 'heavy'):
            fixer.missingAtoms = {}
            fixer.missingTerminals = {}
        logger.info('pdbfixer: Will add missing atoms: {}.'.format(value))
        fixer.addMissingAtoms()
        if value in ('all', 'hydrogens'):
            logger.info('pdbfixer: Will add hydrogens in default protonation state for pH {}.'.format(pH))
            fixer.addMissingHydrogens(pH)
        # Delete the key once we've processed it
        del directives[option]

    # Check that there were no extra options
    if len(directives) > 0:
        raise ValueError("The 'pdbfixer:' block contained some nodes that it didn't know how to process: {}".format(directives))

    # Write the final structure
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_file_path, 'w'))

def read_csv_lines(file_path, lines):
    """Return a list of CSV records.

    The function takes care of ignoring comments and blank lines.

    Parameters
    ----------
    file_path : str
        The path to the CSV file.
    lines : 'all' or int
        The index of the line to read or 'all' to return
        the list of all lines.

    Returns
    -------
    records : str or list of str
        The CSV record if lines is an integer, or a list of CSV
        records if it is 'all'.

    """
    # Read all lines ignoring blank lines and comments.
    with open(file_path, 'r') as f:
        all_records = [line for line in f
                       if bool(line) and not line.strip().startswith('#')]

    if lines == 'all':
        return all_records
    return all_records[lines]


# ==============================================================================
# SETUP DATABASE
# ==============================================================================

class SetupDatabase:
    """Provide utility functions to set up systems and molecules.

    The object allows to access molecules, systems and solvents by 'id' and takes
    care of parametrizing molecules and creating the AMBER prmtop and inpcrd files
    describing systems.

    Parameters
    ----------
    setup_dir : str
        Path to the main setup directory. Changing this means changing the database.
    molecules : dict, Optional. Default: None
        YAML description of the molecules.
        Dictionary should be of form {molecule_id : molecule YAML description}
    solvents : dict, Optional. Default: None
        YAML description of the solvents.
        Dictionary should be of form {solvent_id : solvent YAML description}
    systems : dict, Optional. Default: None
        YAML description of the systems.
        Dictionary should be of form {system_id : system YAML description}

    """

    SYSTEMS_DIR = 'systems'  #: Stock system's sub-directory name
    MOLECULES_DIR = 'molecules'  #: Stock Molecules sub-directory name
    CLASH_THRESHOLD = 1.5  #: distance in Angstroms to consider two atoms clashing

    def __init__(self, setup_dir, molecules=None, solvents=None, systems=None):
        """Initialize the database."""
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
            inpcrd, gro, or pdb) and parameters_path (e.g. prmtop, top, or xml).

        """
        Paths = collections.namedtuple('Paths', ['position_path', 'parameters_path'])
        system_dir = os.path.join(self.setup_dir, self.SYSTEMS_DIR, system_id)

        if 'receptor' in self.systems[system_id]:
            system_files_paths = [
                Paths(position_path=os.path.join(system_dir, 'complex.inpcrd'),
                      parameters_path=os.path.join(system_dir, 'complex.prmtop')),
                Paths(position_path=os.path.join(system_dir, 'solvent.inpcrd'),
                      parameters_path=os.path.join(system_dir, 'solvent.prmtop'))
            ]
        elif 'solute' in self.systems[system_id]:
            system_files_paths = [
                Paths(position_path=os.path.join(system_dir, 'solvent1.inpcrd'),
                      parameters_path=os.path.join(system_dir, 'solvent1.prmtop')),
                Paths(position_path=os.path.join(system_dir, 'solvent2.inpcrd'),
                      parameters_path=os.path.join(system_dir, 'solvent2.prmtop'))
            ]
        else:
            parameter_file_extensions = {'prmtop', 'top', 'xml'}
            system_files_paths = []
            for phase_path_name in ['phase1_path', 'phase2_path']:
                file_paths = self.systems[system_id][phase_path_name]
                assert len(file_paths) == 2

                # Make sure that the position file is first.
                first_file_extension = os.path.splitext(file_paths[0])[1][1:]
                if first_file_extension in parameter_file_extensions:
                    file_paths = list(reversed(file_paths))

                # Append Paths object.
                system_files_paths.append(Paths(position_path=file_paths[0],
                                                parameters_path=file_paths[1]))
        return system_files_paths

    def is_molecule_setup(self, molecule_id):
        """Check whether the molecule has been processed previously.

        The molecule must be set up if it needs to be parametrize by antechamber
        (and the gaff.mol2 and frcmod files do not exist), if the molecule must be
        generated by OpenEye, or if it needs to be extracted by a multi-molecule file.

        An example to clarify the difference between the two return values: a protein
        in a single-frame pdb does not have to be processed (since it does not go through
        antechamber) thus the function will return ``is_setup=True`` and ``is_processed=False``.

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

        # If we have to make mutations, a new PDB should be created
        elif 'pdbfixer' in molecule_descr:
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
                        os.path.exists(system_files_paths[0].parameters_path) and
                        os.path.exists(system_files_paths[1].position_path) and
                        os.path.exists(system_files_paths[1].parameters_path))
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
            inpcrd, gro, or pdb) and parameters_path (e.g. prmtop, top, or xml).

        """
        # Check if system has been already processed
        system_files_paths = self.get_system_files_paths(system_id)
        if self.is_system_setup(system_id)[0]:
            return system_files_paths

        system_descr = self.systems[system_id]
        log_message = 'Setting up the systems for {}, {} and {}'

        if 'receptor' in system_descr:  # binding free energy calculation
            receptor_id = system_descr['receptor']
            ligand_id = system_descr['ligand']
            solvent_id = system_descr['solvent']
            system_parameters = system_descr['leap']['parameters']
            logger.info(log_message.format(receptor_id, ligand_id, solvent_id))

            # solvent phase
            logger.debug('Setting up solvent phase')
            self._setup_system(system_files_paths[1].position_path, False,
                               0, system_parameters, solvent_id, ligand_id)

            try:
                alchemical_charge = int(round(self.molecules[ligand_id]['net_charge']))
            except KeyError:
                alchemical_charge = 0

            # complex phase
            logger.debug('Setting up complex phase')
            self._setup_system(system_files_paths[0].position_path,
                               system_descr['pack'], alchemical_charge,
                               system_parameters, solvent_id, receptor_id,
                               ligand_id)
        else:  # partition/solvation free energy calculation
            solute_id = system_descr['solute']
            solvent1_id = system_descr['solvent1']
            solvent2_id = system_descr['solvent2']
            system_parameters = system_descr['leap']['parameters']
            logger.info(log_message.format(solute_id, solvent1_id, solvent2_id))

            # solvent1 phase
            logger.debug('Setting up solvent1 phase')
            self._setup_system(system_files_paths[0].position_path, False,
                               0, system_parameters, solvent1_id, solute_id)

            # solvent2 phase
            logger.debug('Setting up solvent2 phase')
            self._setup_system(system_files_paths[1].position_path, False,
                               0, system_parameters, solvent2_id, solute_id)

        return system_files_paths

    def setup_all_systems(self):
        """Setup all molecules and systems in the database.

        The method supports parallelization through MPI.

        """
        # Find all molecules that need to be set up.
        molecules_to_setup = []
        for molecule_id in self.molecules:
            if not self.is_molecule_setup(molecule_id)[0]:
                molecules_to_setup.append(molecule_id)
        molecules_to_setup.sort()

        # Parallelize generation of all molecules among nodes.
        mpi.distribute(self._setup_molecules,
                       distributed_args=molecules_to_setup,
                       send_results_to=None, group_size=1, sync_nodes=True)

        # Find all systems that need to be set up.
        systems_to_setup = []
        for system_id in self.systems:
            if not self.is_system_setup(system_id)[0]:
                systems_to_setup.append(system_id)
        systems_to_setup.sort()

        # Parallelize generation of all systems among nodes.
        mpi.distribute(self.get_system,
                       distributed_args=systems_to_setup,
                       send_results_to=None, group_size=1, sync_nodes=True)

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
                molecule = moltools.openeye.iupac_to_oemol(mol_descr['name'])
            elif 'smiles' in mol_descr:
                molecule = moltools.openeye.smiles_to_oemol(mol_descr['smiles'])
            molecule = moltools.openeye.get_charges(molecule, keep_confs=1)
        except ImportError as e:
            error_msg = ('requested molecule generation from name or smiles but '
                         'could not find OpenEye toolkit: ' + str(e))
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return molecule

    def _generate_residue_name(self, molecule_id):
        """Generates a residue name for a molecule.

        The function guarantees to not generate twice the same residue name.
        Purely numeric residue names mess up the pipeline, so we generate
        residue names of the form YXX, where Y is a letter and X are digits
        (e.g. A01, A02, ..., Z99).

        The order of the generated residue name is not guaranteed.

        WARNING: The algorithm may fail when new molecules are added to
        self.molecules after construction. This is not the case right now,
        but it's good to keep in mind.

        Parameters
        ----------
        molecule_id : str
            The molecule identifier.

        Returns
        -------
        residue_name : str
            A three-character residue name.
        """
        # We need to associate a unique number to this molecule, and do
        # so in such a way that distributing molecule setups over multiple
        # MPI process still ends up in unique residue names for each molecule.
        molecule_ids = sorted(self.molecules.keys())
        n_molecule = molecule_ids.index(molecule_id)
        assert n_molecule < 2600

        # Build 3-character identifier.
        character = chr(n_molecule // 100 + 65)
        digits = str(n_molecule % 100)
        residue_name = character + digits.zfill(2)
        return residue_name

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
                print('Molecule has already been set up')
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
                    # Extract the correct line and save it in a new file.
                    smiles_line = read_csv_lines(mol_descr['filepath'], lines=model_idx)
                    with open(single_file_path, 'w') as f:
                        f.write(smiles_line)

                elif extension == '.mol2' or extension == '.sdf':
                    if not utils.is_openeye_installed(oetools=('oechem',)):
                        raise RuntimeError('Cannot support {} files selection without OpenEye'.format(
                                extension[1:]))
                    oe_molecule = utils.load_oe_molecules(mol_descr['filepath'], molecule_idx=model_idx)
                    if extension == '.mol2':
                        mol_names = list(utils.Mol2File(mol_descr['filepath']).resnames)
                        utils.write_oe_molecule(oe_molecule, single_file_path, mol2_resname=mol_names[model_idx])
                    else:
                        utils.write_oe_molecule(oe_molecule, single_file_path)

                else:
                    raise RuntimeError('Model selection is not supported for {} files'.format(extension[1:]))

                # Save new file path
                mol_descr['filepath'] = single_file_path

            # Strip off protons if required
            if 'strip_protons' in mol_descr and mol_descr['strip_protons']:
                if extension != '.pdb':
                    raise RuntimeError('Cannot strip protons from {} files.'.format(extension[1:]))
                output_file_path = os.path.join(mol_dir, mol_id + '.pdb')
                strip_protons(mol_descr['filepath'], output_file_path)
                mol_descr['filepath'] = output_file_path

            # Apply PDBFixer if requested
            if 'pdbfixer' in mol_descr:
                if extension not in ['.pdb', '.PDB']:
                    raise RuntimeError('Cannot apply PDBFixer to {} files; a .pdb file is required.'.format(extension[1:]))
                output_file_path = os.path.join(mol_dir, mol_id + '.pdb')
                apply_pdbfixer(mol_descr['filepath'], output_file_path, mol_descr['pdbfixer'])
                mol_descr['filepath'] = output_file_path

            # Generate missing molecules with OpenEye. At the end of parametrization
            # we update the 'filepath' key also for OpenEye-generated molecules so
            # we don't need to keep track of the molecules we have already generated
            if extension is None or extension == '.smiles' or extension == '.csv':
                if not utils.is_openeye_installed(oetools=('oechem', 'oeiupac', 'oequacpac', 'oeomega')):
                    if extension is None:
                        raise RuntimeError('Cannot generate molecule {} without OpenEye licensed with '
                                           'OEChem, OEIUPAC, OEOmega, and OEQuacPack.'.format(mol_id))
                    else:
                        raise RuntimeError('Cannot support {} files without OpenEye licensed with '
                                           'OEChem, OEIUPAC, OEOmega, and OEQuacPack.'.format(extension[1:]))

                # Retrieve the first SMILES string (eventually extracted
                # while handling of the 'select' keyword above)
                if extension is not None:
                    # Get first record in CSV file.
                    first_line = read_csv_lines(mol_descr['filepath'], lines=0)

                    # Automatically detect if delimiter is comma or semicolon
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
                            break
                        except (ValueError, RuntimeError):
                            oe_molecule = None

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

                # Generate a residue name for the SMILES molecule.
                residue_name = self._generate_residue_name(mol_id)
                moltools.openeye.molecule_to_mol2(oe_molecule, mol_descr['filepath'],
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
                moltools.schrodinger.run_epik(mol_descr['filepath'], epik_mae_file, **epik_kwargs)
                moltools.schrodinger.run_structconvert(epik_mae_file, epik_sdf_file)
                moltools.schrodinger.run_structconvert(epik_mae_file, epik_mol2_file)

                # Save new net charge from the i_epik_Tot_Q property
                net_charge = int(moltools.schrodinger.run_proplister(epik_sdf_file)[0]['i_epik_Tot_Q'])

                # Keep filepath consistent
                mol_descr['filepath'] = epik_mol2_file

            # Antechamber does not support sdf files so we need to convert them
            extension = os.path.splitext(mol_descr['filepath'])[1]
            if extension == '.sdf':
                if not utils.is_openeye_installed(oetools=('oechem',)):
                    raise RuntimeError('Cannot support sdf files without OpenEye OEChem')
                mol2_file_path = os.path.join(mol_dir, mol_id + '.mol2')
                oe_molecule = utils.load_oe_molecules(mol_descr['filepath'], molecule_idx=0)

                # Generate a residue name for the sdf molecule.
                residue_name = self._generate_residue_name(mol_id)
                moltools.openeye.molecule_to_mol2(oe_molecule, mol2_file_path,
                                                  residue_name=residue_name)

                # Update filepath information
                mol_descr['filepath'] = mol2_file_path

            # Parametrize the molecule with antechamber
            if 'antechamber' in mol_descr:
                # Generate charges with OpenEye if requested
                if 'openeye' in mol_descr:
                    if not utils.is_openeye_installed(oetools=('oechem', 'oequacpac', 'oeomega')):
                        err_msg = ('Cannot find OpenEye toolkit with OEChem and OEQuacPac to compute charges '
                                   'for molecule {}').format(mol_id)
                        logger.error(err_msg)
                        raise RuntimeError(err_msg)
                    mol2_file_path = os.path.join(mol_dir, mol_id + '.mol2')
                    oe_molecule = utils.load_oe_molecules(mol_descr['filepath'], molecule_idx=0)

                    # Setting keep_confs = None keeps the original conformation
                    oe_molecule = moltools.openeye.get_charges(oe_molecule, keep_confs=None)
                    residue_name = utils.Mol2File(mol_descr['filepath']).resname
                    moltools.openeye.molecule_to_mol2(oe_molecule, mol2_file_path,
                                                      residue_name=residue_name)

                    utils.Mol2File(mol2_file_path).round_charge()  # normalize charges
                    net_charge = None  # we don't need Epik's net charge
                    mol_descr['filepath'] = mol2_file_path

                # Generate parameters
                charge_method = mol_descr['antechamber']['charge_method']
                input_mol_path = os.path.abspath(mol_descr['filepath'])
                # Use Gaff in parameters, otherwise default to gaff2
                gaff = 'gaff2' if 'leaprc.gaff2' in mol_descr['leap']['parameters'] else 'gaff'
                with moltools.utils.temporary_cd(mol_dir):
                    moltools.amber.run_antechamber(mol_id, input_mol_path,
                                                   charge_method=charge_method,
                                                   net_charge=net_charge,
                                                   gaff_version=gaff)

                # Save new parameters paths
                mol_descr['filepath'] = os.path.join(mol_dir, mol_id + '.gaff.mol2')
                mol_descr['leap']['parameters'].append(os.path.join(mol_dir, mol_id + '.frcmod'))

                # Normalize charges if not done before
                if 'openeye' not in mol_descr:
                    utils.Mol2File(mol_descr['filepath']).round_charge()

            # Determine small molecule net charge
            extension = os.path.splitext(mol_descr['filepath'])[1]
            if extension == '.mol2':
                # TODO what if this is a peptide? this should be computed in get_system()
                mol_descr['net_charge'] = utils.Mol2File(mol_descr['filepath']).net_charge

            # Keep track of processed molecule
            self._processed_mols.add(mol_id)

    def _setup_system(self, system_file_path, pack, alchemical_charge,
                      system_parameters, solvent_id, *molecule_ids, **kwargs):
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
        ignore_ionic_strength : bool, optional
            If True, no ions will be added to reach the ionic strength (default
            is False).
        save_amber_files : bool, optional
            If False, prmtop and inpcrd files are not saved (default is True).

        Other Parameters
        ----------------
        *molecule_ids : list-like of str
            List the IDs of the molecules to pack together in the system.

        """

        # Get kwargs
        ignore_ionic_strength = kwargs.pop('ignore_ionic_strength', False)
        save_amber_files = kwargs.pop('save_amber_files', True)
        assert len(kwargs) == 0

        # Make sure molecules are set up
        self._setup_molecules(*molecule_ids)
        solvent = self.solvents[solvent_id]

        # Start error tracking variables
        # Water
        known_solvent_files = [file for file in _OPENMM_LEAP_SOLVENT_FILES_MAP.values()]
        loaded_water_files = []  # Detected loaded water files

        def extend_list_of_waters(leap_parameters):
            """Extend the loaded_water_files list given the leap_parameters"""
            loaded_water_files.extend([water for water
                                       in leap_parameters
                                       if water in known_solvent_files])

        # Create tleap script
        tleap = utils.TLeap()

        # Load all parameters
        # --------------------
        tleap.new_section('Load parameters')

        for mol_id in molecule_ids:
            molecule_parameters = self.molecules[mol_id]['leap']['parameters']
            # Track loaded water models
            extend_list_of_waters(molecule_parameters)
            tleap.load_parameters(*molecule_parameters)

        extend_list_of_waters(system_parameters)
        tleap.load_parameters(*system_parameters)
        solvent_leap = solvent['leap']['parameters']
        extend_list_of_waters(solvent_leap)
        tleap.load_parameters(*solvent_leap)

        # Load molecules and create complexes
        # ------------------------------------
        tleap.new_section('Load molecules')
        for mol_id in molecule_ids:
            tleap.load_unit(unit_name=mol_id, file_path=self.molecules[mol_id]['filepath'])

        if len(molecule_ids) > 1:
            # Check that molecules don't have clashing atoms. Also, if the ligand
            # is too far away from the molecule we want to pull it closer
            # TODO this check should be available even without OpenEye
            if pack and utils.is_openeye_installed(oetools=('oechem',)):

                # Load atom positions of all molecules
                positions = [0 for _ in molecule_ids]
                for i, mol_id in enumerate(molecule_ids):
                    if mol_id not in self._pos_cache:
                        self._pos_cache[mol_id] = utils.get_oe_mol_positions(
                                utils.load_oe_molecules(self.molecules[mol_id]['filepath'], molecule_idx=0))
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
                    get_leap_recommended_pbradii(implicit_solvent)))
        else:  # explicit solvent
            tleap.new_section('Solvate systems')

            # Solvate unit. Solvent models different than tip3p need parameter modifications.
            solvent_model = solvent['solvent_model']
            # Check that solvent model has loaded the appropriate leap parameters.
            # This does guarantee a failure, but is a good sign of it.
            if _OPENMM_LEAP_SOLVENT_FILES_MAP[solvent_model] not in loaded_water_files:
                solvent_warning = ("WARNING: The solvent_model {} may not work for loaded "
                                   "leaprc.water.X files.\n We expected {} to make your "
                                   "solvent model work, but did not find it.\n "
                                   "This is okay for tip4pew leaprc file with tip3p solvent_model, "
                                   "but not the other way around.\nThis does "
                                   "not mean the system will not build, but it may throw "
                                   "an error.".format(solvent_model,
                                                      _OPENMM_LEAP_SOLVENT_FILES_MAP[solvent_model]))
                logger.warning(solvent_warning)
            leap_solvent_model = _OPENMM_LEAP_SOLVENT_MODELS_MAP[solvent_model]
            clearance = float(solvent['clearance'].value_in_unit(unit.angstroms))
            tleap.solvate(unit_name=unit_to_solvate, solvent_model=leap_solvent_model, clearance=clearance)

            # First, determine how many ions we need to add for the ionic strength.
            if not ignore_ionic_strength and solvent['ionic_strength'] != 0.0*unit.molar:
                # Currently we support only monovalent ions.
                for ion_name in [solvent['positive_ion'], solvent['negative_ion']]:
                    assert '2' not in ion_name and '3' not in ion_name

                logger.debug('Estimating number of water molecules in the box.')
                n_waters = self._get_number_box_waters(pack, alchemical_charge, system_parameters,
                                                       solvent_id, *molecule_ids)
                logger.debug('Estimated number of water molecules: {}'.format(n_waters))

                # Water molarity at room temperature: 998.23g/L / 18.01528g/mol ~= 55.41M
                n_ions_ionic_strength = int(np.round(n_waters * solvent['ionic_strength'] / (55.41*unit.molar)))

                logging.debug('Adding {} ions in {} water molecules to reach ionic strength '
                              'of {}'.format(n_ions_ionic_strength, n_waters, solvent['ionic_strength']))
            else:
                n_ions_ionic_strength = 0

            # Add alchemically modified ions that we don't already add for ionic strength.
            if abs(alchemical_charge) > n_ions_ionic_strength:
                n_alchemical_ions = abs(alchemical_charge) - n_ions_ionic_strength
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
                tleap.add_ions(unit_name=unit_to_solvate, ion=ion,
                               num_ions=n_alchemical_ions, replace_solvent=True)
                logging.debug('Adding {} {} ion to neutralize ligand charge of {}'
                              ''.format(n_alchemical_ions, ion, alchemical_charge))

            # Neutralizing solvation box
            if 'positive_ion' in solvent:
                tleap.add_ions(unit_name=unit_to_solvate, ion=solvent['positive_ion'],
                               replace_solvent=True)
            if 'negative_ion' in solvent:
                tleap.add_ions(unit_name=unit_to_solvate, ion=solvent['negative_ion'],
                               replace_solvent=True)

            # Ions for the ionic strength must be added AFTER neutralization.
            if n_ions_ionic_strength != 0:
                tleap.add_ions(unit_name=unit_to_solvate, ion=solvent['positive_ion'],
                               num_ions=n_ions_ionic_strength, replace_solvent=True)
                tleap.add_ions(unit_name=unit_to_solvate, ion=solvent['negative_ion'],
                               num_ions=n_ions_ionic_strength, replace_solvent=True)

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
        if save_amber_files:
            tleap.save_unit(unit_to_solvate, system_file_path)
        tleap.save_unit(unit_to_solvate, base_file_path + '.pdb')

        # Save tleap script for reference
        tleap.export_script(base_file_path + '.leap.in')

        # Run tleap and log warnings
        # Handle common errors we know of
        try:
            warnings = tleap.run()
        except RuntimeError as e:
            error = RuntimeError('Solvent {}: {}'.format(solvent_id, str(e)))
            error.with_traceback(sys.exc_info()[2])
            raise error

        for warning in warnings:
            logger.warning('TLeap: ' + warning)

    def _get_number_box_waters(self, *args):
        """Build a system in a temporary directory and count the number of waters."""
        with mmtools.utils.temporary_directory() as tmp_dir:
            system_file_path = os.path.join(tmp_dir, 'temp_system.prmtop')
            self._setup_system(system_file_path, *args, ignore_ionic_strength=True,
                               save_amber_files=False)

            # Count number of waters of created system.
            system_file_path = os.path.join(tmp_dir, 'temp_system.pdb')
            system_traj = mdtraj.load(system_file_path)
            n_waters = sum([1 for res in system_traj.topology.residues if res.is_water])

        return n_waters


# ==============================================================================
# FUNCTIONS FOR ALCHEMICAL PATH OPTIMIZATION
# ==============================================================================

def trailblaze_alchemical_protocol(thermodynamic_state, sampler_state, mcmc_move, state_parameters,
                                   std_energy_threshold=0.5, threshold_tolerance=0.05,
                                   n_samples_per_state=100):
    """
    Find an alchemical path by placing alchemical states at a fixed distance.

    The distance between two states is estimated by collecting ``n_samples_per_state``
    configurations through the MCMCMove in one of the two alchemical states,
    and computing the standard deviation of the difference of potential energies
    between the two states at those configurations.

    Two states are chosen for the protocol if their standard deviation is
    within ``std_energy_threshold +- threshold_tolerance``.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.CompoundThermodynamicState
        The state of the alchemically modified system.
    sampler_state : openmmtools.states.SamplerState
        The sampler states including initial positions and box vectors.
    mcmc_move : openmmtools.mcmc.MCMCMove
        The MCMCMove to use for propagation.
    state_parameters : list of tuples (str, [float, float])
        Each element of this list is a tuple containing first the name
        of the parameter to be modified (e.g. ``lambda_electrostatics``,
        ``lambda_sterics``) and a list specifying the initial and final
        values for the path.
    std_energy_threshold : float
        The threshold that determines how to separate the states between
        each others.
    threshold_tolerance : float
        The tolerance on the found standard deviation.
    n_samples_per_state : int
        How many samples to collect to estimate the overlap between two
        states.

    Returns
    -------
    optimal_protocol : dict {str, list of floats}
        The estimated protocol. Each dictionary key is one of the
        parameters in ``state_parameters``, and its values is the
        list of values that it takes in each state of the path.

    """
    # Make sure that the state parameters to optimize have a clear order.
    assert(isinstance(state_parameters, list) or isinstance(state_parameters, tuple))

    # Make sure that thermodynamic_state is in correct state
    # and initialize protocol with starting value.
    optimal_protocol = {}
    for parameter, values in state_parameters:
        setattr(thermodynamic_state, parameter, values[0])
        optimal_protocol[parameter] = [values[0]]

    # We change only one parameter at a time.
    for state_parameter, values in state_parameters:
        logger.debug('Determining alchemical path for parameter {}'.format(state_parameter))

        # Is this a search from 0 to 1 or from 1 to 0?
        search_direction = np.sign(values[1] - values[0])
        # If the parameter doesn't change, continue to the next one.
        if search_direction == 0:
            continue

        # Gather data until we get to the last value.
        while optimal_protocol[state_parameter][-1] != values[-1]:
            # Simulate current thermodynamic state to obtain energies.
            sampler_states = []
            simulated_energies = np.zeros(n_samples_per_state)
            for i in range(n_samples_per_state):
                mcmc_move.apply(thermodynamic_state, sampler_state)
                context, _ = mmtools.cache.global_context_cache.get_context(thermodynamic_state)
                sampler_state.apply_to_context(context, ignore_velocities=True)
                simulated_energies[i] = thermodynamic_state.reduced_potential(context)
                sampler_states.append(copy.deepcopy(sampler_state))

            # Find first state that doesn't overlap with simulated one
            # with std(du) within std_energy_threshold +- threshold_tolerance.
            # We stop anyway if we reach the last value of the protocol.
            std_energy = 0.0
            current_parameter_value = optimal_protocol[state_parameter][-1]
            while (abs(std_energy - std_energy_threshold) > threshold_tolerance and
                   not (current_parameter_value == values[1] and std_energy < std_energy_threshold)):
                # Determine next parameter value to compute.
                if np.isclose(std_energy, 0.0):
                    # This is the first iteration or the two state overlap significantly
                    # (e.g. small molecule in vacuum). Just advance by a +- 0.05 step.
                    old_parameter_value = current_parameter_value
                    current_parameter_value += (values[1] - values[0]) / 20.0
                else:
                    # Assume std_energy(parameter_value) is linear to determine next value to try.
                    derivative_std_energy = ((std_energy - old_std_energy) /
                                             (current_parameter_value - old_parameter_value))
                    old_parameter_value = current_parameter_value
                    current_parameter_value += (std_energy_threshold - std_energy) / derivative_std_energy

                # Keep current_parameter_value inside bound interval.
                if search_direction * current_parameter_value > values[1]:
                    current_parameter_value = values[1]
                assert search_direction * (optimal_protocol[state_parameter][-1] - current_parameter_value) < 0

                # Get context in new thermodynamic state.
                setattr(thermodynamic_state, state_parameter, current_parameter_value)
                context, integrator = mmtools.cache.global_context_cache.get_context(thermodynamic_state)

                # Compute the energies at the sampled positions.
                reweighted_energies = np.zeros(n_samples_per_state)
                for i, sampler_state in enumerate(sampler_states):
                    sampler_state.apply_to_context(context, ignore_velocities=True)
                    reweighted_energies[i] = thermodynamic_state.reduced_potential(context)

                # Compute standard deviation of the difference.
                old_std_energy = std_energy
                denergies = reweighted_energies - simulated_energies
                std_energy = np.std(denergies)
                logger.debug('trailblazing: state_parameter {}, simulated_value {}, current_parameter_value {}, '
                             'std_du {}'.format(state_parameter, optimal_protocol[state_parameter][-1],
                                                current_parameter_value, std_energy))

            # Update the optimal protocol with the new value of this parameter.
            # The other parameters remain fixed.
            for par_name in optimal_protocol:
                # Make sure we append to a Python float to the list.
                # Lists of numpy types sometimes give problems.
                if par_name == state_parameter:
                    protocol_value = float(current_parameter_value)
                else:
                    protocol_value = float(optimal_protocol[par_name][-1])
                optimal_protocol[par_name].append(protocol_value)

    logger.debug('Alchemical path found: {}'.format(optimal_protocol))
    return optimal_protocol


if __name__ == '__main__':
    import doctest
    doctest.testmod()
