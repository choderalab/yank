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

import collections
import copy
import inspect
import itertools
import json
import logging
import os
import re
import sys

import mdtraj
import mpiplus
import numpy as np
import openmmtools as mmtools
import openmoltools as moltools
import yaml
from pdbfixer import PDBFixer
from simtk import openmm, unit
from simtk.openmm.app import PDBFile


from . import utils

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

def order_counterions(system, topography, sampler_state):
    """Order the counterions in decreasing distance to the ligand/solute.

    Parameters
    ----------
    system  : simtk.openmm.System
        The system object containing the atoms of interest.
    topography : Yank topography
        The topography object holding the indices of the ions and the
        ligand (for binding free energy) or solute (for transfer free
        energy).
    sampler_state : openmmtools.mdtools.sampler_state
        State with positions 

    Returns
    -------
    ion_index, charge : list
        List of ion index and charge

    """
    pos_unit = sampler_state.positions.unit
    # atoms of the ligand/solute which are part of the alchemical transformation
    alchemical_atoms = topography.ligand_atoms
    # if no ligand present but a charged solute
    if not alchemical_atoms:
        alchemical_atoms = topography.solute_atoms
    alchemical_positions = sampler_state.positions[alchemical_atoms] / pos_unit
    # compute minimal distance between each ion and the alchemical atoms
    ions_distances = np.zeros(len(topography.ions_atoms))
    for i, ion_id in enumerate(topography.ions_atoms):
        ion_position = sampler_state.positions[ion_id] / pos_unit 
        ions_distances[i] = compute_min_dist([ion_position], alchemical_positions)
        logger.debug('Min distance of ion {} {}: {}'.format(ion_id,
                                                            topography.topology.atom(ion_id).residue.name,
                                                            ions_distances[i]))
    # list with ion_id ordered in decreasing distance of the ligand
    ordered_ion_id = [topography.ions_atoms[ion_id] for ion_id in np.argsort(ions_distances)[::-1]]
    return [(ion_id, compute_net_charge(system, [ion_id]))
                        for ion_id in ordered_ion_id]

def find_alchemical_counterions(system, topography, sampler_state, region_name):
    """Return the atom indices of the ligand or solute counter ions.

    In periodic systems, the solvation box needs to be neutral, and
    if the decoupled molecule is charged, it will cause trouble. This
    can be used to find a set of ions in the system that neutralize
    the molecule, so that the solvation box will remain neutral all
    the time. Ions are selected starting with the ones with the largest
    distance to the ligand.

    Parameters
    ----------
    system : simtk.openmm.System
        The system object containing the atoms of interest.
    topography : yank.Topography
        The topography object holding the indices of the ions and the
        ligand (for binding free energy) or solute (for transfer free
        energy).
    sampler_state : openmmtools.mdtools.sampler_state
        State with positions 
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

    # Find net charge of all ions in the system and order them according to the 
    # largest distance from the ligand/solute
    ions_net_charges = order_counterions(system, topography, sampler_state)
    topology = topography.topology
    ions_names_charges = [(topology.atom(ion_id).residue.name, ion_net_charge)
                          for ion_id, ion_net_charge in ions_net_charges]


    # Find minimal subset of counterions whose charges sums to -mol_net_charge.
    for n_ions in range(1, len(ions_net_charges) + 1):
        for ion_subset in itertools.combinations(ions_net_charges, n_ions):
            counterions_indices, counterions_charges = zip(*ion_subset)
            if sum(counterions_charges) == -mol_net_charge:
                logger.debug('Index of alchemical counter ion {}'.format(counterions_indices))
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
                      gromacs_include_dir=None, charmm_parameter_files=None):
    """Create a Yank arguments for a phase from system files.

    Parameters
    ----------
    positions_file_path : str
        Path to system position file (e.g. 'complex.inpcrd/.gro/.pdb').
    parameters_file_path : str
        Path to system parameters file (e.g. 'complex.prmtop/.top/.xml/.psf').
    system_options : dict
        ``system_options[phase]`` is a a dictionary containing options to
        pass to ``createSystem()``. If the parameters file is an OpenMM
        system in XML format, this will be ignored.
    gromacs_include_dir : str, optional
        Path to directory in which to look for other files included
        from the gromacs top file.
    charmm_parameter_files : str, optional
        Path to additional parameter files

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
        box_vectors = positions_file.topology.getPeriodicBoxVectors()
        if box_vectors is None:
            box_vectors = system.getDefaultPeriodicBoxVectors()

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

    # Read CHARMM format psf and pdb files
    elif parameters_file_extension == '.psf':
        logger.debug("psf: {}".format(parameters_file_path))
        logger.debug("pdb: {}".format(positions_file_path))

        parameters_file = openmm.app.CharmmPsfFile(parameters_file_path)
        positions_file = openmm.app.PDBFile(positions_file_path)
        params = openmm.app.CharmmParameterSet(*charmm_parameter_files)

        box_vectors = positions_file.topology.getPeriodicBoxVectors()
        if box_vectors is None:
            box_vectors = system.getDefaultPeriodicBoxVectors()

        parameters_file.setBox(box_vectors[0][0], box_vectors[1][1], box_vectors[2][2])
        create_system_args = set(inspect.getargspec(openmm.app.CharmmPsfFile.createSystem).args)
        system_options['params'] = params
        system = create_system(parameters_file, box_vectors, create_system_args, system_options)
            
    # Unsupported file format.
    else:
        raise ValueError('Unsupported format for parameter file {}'.format(parameters_file_extension))

    # Store numpy positions and create SamplerState.
    positions = positions_file.getPositions(asNumpy=True)
    sampler_state = mmtools.states.SamplerState(positions=positions, box_vectors=box_vectors)

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


def process_tool_directive(directives, option, dispatch, allowed_values, yields_value=False):
    """Process a directive.

    Parameters
    ----------
    option : str
        The name of the option to be processed.
        Will remove this option from `directives` once processed.
    dispatch : function
        The function to call.
    allowed_values : list
        If not None, the value of directives[option] will be checked against this list
    yields_value : boolean, default False
        Tells this function to expect a return from the dispatch function and give it back as needed
    """
    if option in directives:
        value = directives[option]
        # Validate options
        if allowed_values is not None:
            if value not in allowed_values:
                raise ValueError("'{}' must be one of {}".format(option, allowed_values))
        # Dispatch
        output = dispatch(value)
        # Delete the key once we've processed it
        del directives[option]
        if yields_value:
            return output
        return

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
    DEFAULT_PH = 7.4 # default pH

    # Make a copy since we will delete from the dictionary to validate
    directives = copy.deepcopy(directives)

    # Create a PDBFixer object
    fixer = PDBFixer(input_file_path)
    fixer.missingResidues = {}

    # Dispatch functions
    # These won't be documented individually because they are so short
    def dispatch_pH(value):
        pH = DEFAULT_PH
        try:
            pH = float(value)
            logger.info('pdbfixer: Will use user-specified pH {}'.format(pH))
        except:
            raise ValueError("'ph' must be a floating-point number: found '{}'".format(value))
        return pH

    pH = process_tool_directive(directives, 'ph', dispatch_pH, None, yields_value=True)

    def add_missing_residues(value):
        if value == 'yes':
            fixer.findMissingResidues()
            logger.info('pdbfixer: Will add missing residues specified in SEQRES')

    def apply_mutations(value):
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

    def replace_nonstandard_residues(value):
        if value == 'yes':
            logger.info('pdbfixer: Will replace nonstandard residues.')
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()

    def remove_heterogens(value):
        if value == 'water':
            logger.info('pdbfixer: Will remove heterogens, retaining water.')
            fixer.removeHeterogens(keepWater=True)
        elif value == 'all':
            logger.info('pdbfixer: Will remove heterogens, discarding water.')
            fixer.removeHeterogens(keepWater=False)

    def add_missing_atoms(value):
        fixer.findMissingAtoms()
        if value not in ('all', 'heavy'):
            fixer.missingAtoms = {}
            fixer.missingTerminals = {}
        logger.info('pdbfixer: Will add missing atoms: {}.'.format(value))
        fixer.addMissingAtoms()
        if value in ('all', 'hydrogens'):
            logger.info('pdbfixer: Will add hydrogens in default protonation state for pH {}.'.format(pH))
            fixer.addMissingHydrogens(pH)

    # Set default atom addition method
    if 'add_missing_atoms' not in directives:
        directives['add_missing_atoms'] = 'heavy'

    # Dispatch directives
    process_tool_directive(directives, 'add_missing_residues', add_missing_residues, [True, False])
    process_tool_directive(directives, 'apply_mutations', apply_mutations, None)
    process_tool_directive(directives, 'replace_nonstandard_residues', replace_nonstandard_residues, [True, False])
    process_tool_directive(directives, 'remove_heterogens', remove_heterogens, ['all', 'water', 'none'])
    process_tool_directive(directives, 'add_missing_atoms', add_missing_atoms, ['all', 'heavy', 'hydrogens', 'none'])

    # Check that there were no extra options
    if len(directives) > 0:
        raise ValueError("The 'pdbfixer:' block contained some nodes that it didn't know how to process: {}".format(directives))

    # Write the final structure
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_file_path, 'w'))


def apply_modeller(input_file_path, output_file_path, directives):
    """
    Apply Salilab Modeller to make changes to the specified molecule.

    Single mutants are supported in the form "T315I"
    Double mutants are not currently supported.

    The string "WT" makes no mutations.

    Original PDB file numbering scheme is used.

    Currently, only PDB files are supported.
    modeller is used to make the mutations. You must have a license file installed for this to work

    Parameters
    ----------
    input_file_path : str
        Full file path to the file to read, including extensions
    output_file_path : str
        Full file path to the file to save, including extensions
    directives : dict
        Dict containing directives for modeller.
    """

    if not utils.is_modeller_installed():
        raise ImportError('Modeller and license must be installed to use this feature.')
    import modeller

    directives = copy.deepcopy(directives)

    # Silence unnecessary output to the log files
    modeller.log.none()

    # Create modeller environment and point it to the PDB file
    env = modeller.environ()
    atom_files_directory = os.path.dirname(input_file_path)
    atom_file_name = os.path.basename(input_file_path)

    # Read in topology and parameter files
    env.libs.topology.read(file='$(LIB)/top_heav.lib')
    env.libs.parameters.read(file='$(LIB)/par.lib')

    env.io.atom_files_directory = [atom_files_directory]
    alignment = modeller.alignment(env)
    model = modeller.model(env, file=atom_file_name)
    model_original_numbering = modeller.model(env, file=atom_file_name)
    alignment.append_model(model, atom_files=atom_file_name, align_codes=atom_file_name)

    def apply_mutations_modeller(value):
        # Extract chain id
        chain_id = None
        if 'chain_id' in value:
            chain_id = value['chain_id']
            if chain_id == 'none':
                chain_id = 0
        # Extract mutations
        mutations = value['mutations']
        # Convert mutations to PDBFixer format
        if mutations != 'WT':
            modeller_mutations = [generate_pdbfixer_mutation_code(*decompose_mutation(mutation))
                                  for mutation in mutations.split('/')]
            if len(modeller_mutations) > 1:
                raise ValueError('{} is a double mutant and not supported by Modeller currently.'.format(mutations))
            else:
                logger.info('modeller: Will make mutations {} to chain_id {}.'.format(modeller_mutations, chain_id))
                sel = modeller.selection(model.chains[chain_id].residues[modeller_mutations[0].split('-')[1]])
                sel.mutate(residue_type=modeller_mutations[0].split('-')[2])
                alignment.append_model(model, align_codes=modeller_mutations[0])
                model.clear_topology()
                model.generate_topology(alignment[modeller_mutations[0]])
                model.transfer_xyz(alignment)
                model.build(initialize_xyz=False, build_method='INTERNAL_COORDINATES')

        else:
            logger.info('modeller: No mutations will be applied since "WT" specified.')
            alignment.append_model(model, align_codes='WT')

    process_tool_directive(directives, 'apply_mutations', apply_mutations_modeller, None)

    # Check that there were no extra options
    if len(directives) > 0:
        raise ValueError("The 'modeller:' block contained some nodes that it didn't know how to process: {}".format(directives))

    # Write the final model
    model.res_num_from(model_original_numbering, alignment)
    model.write(file=output_file_path)


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
            parameter_file_extensions = {'prmtop', 'top', 'xml', 'psf'}
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

        # If we have to make mutations using modeller, a new PDB should be created
        elif 'modeller' in molecule_descr:
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
        log_message = 'Setting up the systems for {} and {} using solvent {}'

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
        mpiplus.distribute(self._setup_molecules,
                           distributed_args=molecules_to_setup,
                           send_results_to=None, group_size=1, sync_nodes=True)

        # Find all systems that need to be set up.
        systems_to_setup = []
        for system_id in self.systems:
            if not self.is_system_setup(system_id)[0]:
                systems_to_setup.append(system_id)
        systems_to_setup.sort()

        # Parallelize generation of all systems among nodes.
        mpiplus.distribute(self.get_system,
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

            # Apply modeller if requested
            if 'modeller' in mol_descr:
                if extension not in ['.pdb', '.PDB']:
                    raise RuntimeError('Cannot apply modeller to {} files; a .pdb file is required.'.format(extension[1:]))
                output_file_path = os.path.join(mol_dir, mol_id + '.pdb')
                apply_modeller(mol_descr['filepath'], output_file_path, mol_descr['modeller'])
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
                    # We don't need Epik's or input net charge as antechamber will
                    # infer the net charge from the sum of the OpenEye charges.
                    net_charge = None
                    mol_descr['filepath'] = mol2_file_path
                # Check if use specified a net_charge, but don't overwrite Epik's protonation state.
                elif net_charge is not None:
                    net_charge = mol_descr['antechamber'].get('net_charge', None)

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
            clearance = solvent['clearance']
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

class _DCDTrajectoryFile(mdtraj.formats.dcd.DCDTrajectoryFile):
    """Convenience class extending MDTraj DCD trajectory file.

    This handles units and allow reading/writing SamplerStates instead
    of positions, cell_lengths, and cell_angles.

    """

    def read(self, *args, **kwargs):
        positions, cell_lengths, cell_angles = super().read(*args, **kwargs)
        # Add standard DCD units.
        return positions*unit.angstrom, cell_lengths*unit.angstrom, cell_angles*unit.degree

    def read_as_sampler_states(self, *args, **kwargs):
        positions, cell_lengths, cell_angles = self.read(*args, **kwargs)
        sampler_states = []
        for i in range(len(positions)):
            box_vectors = mdtraj.utils.lengths_and_angles_to_box_vectors(
                *(cell_lengths[i]/unit.nanometer),
                *(cell_angles[i]/unit.degree)
            ) * unit.nanometer
            sampler_state = mmtools.states.SamplerState(positions[i], box_vectors=box_vectors)
            sampler_states.append(sampler_state)
        return sampler_states

    def write(self, xyz, cell_lengths=None, cell_angles=None, **kwargs):
        # Convert to standard DCD units.
        super().write(xyz/unit.angstrom, cell_lengths/unit.angstrom, cell_angles/unit.degree)

    def write_sampler_state(self, sampler_state):
        a, b, c, alpha, beta, gamma = mdtraj.utils.box_vectors_to_lengths_and_angles(
            *(np.array(sampler_state.box_vectors / unit.angstrom)))
        super().write(sampler_state.positions / unit.angstrom,
                      (a, b, c), (alpha, beta, gamma))


def read_trailblaze_checkpoint_coordinates(checkpoint_dir_path, redistributed=True):
    """Read positions and box vectors stored as checkpoint by the trailblaze algorithm.

    Parameters
    ----------
    checkpoint_dir_path : str
        The path to the directory containing the checkpoint information.
    redistributed : bool, optional
        If True, the function will check if the states were redistributed,
        and will returned the set of coordinates that are more representative
        of the redistributed protocol.

    Returns
    -------
    sampler_states : List[openmmtools.states.SamplerState], optional
        ``sampler_states[i]`` contain positions and box vectors for the
        intermediate state i generated by the trailblaze algorithm.

    Raises
    ------
    FileNotFoundError
        If no file with the coordinates was found.
    """
    positions_file_path = os.path.join(checkpoint_dir_path, 'coordinates.dcd')
    states_map_file_path = os.path.join(checkpoint_dir_path, 'states_map.json')

    # Open the file if it exist.
    try:
        trajectory_file = _DCDTrajectoryFile(positions_file_path, 'r')
    except OSError as e:
        raise FileNotFoundError(str(e))

    # Read info.
    try:
        sampler_states = trajectory_file.read_as_sampler_states()
    finally:
        trajectory_file.close()

    # If the protocol was redistributed, use the states map to create
    # a new set of sampler states that can be used as starting conditions
    # for the redistributed protocol.
    if redistributed:
        try:
            with open(states_map_file_path, 'r') as f:
                states_map = json.load(f)
            sampler_states = [sampler_states[i] for i in states_map]
        except FileNotFoundError:
            pass

    return sampler_states


def _resume_thermodynamic_trailblazing(checkpoint_dir_path, initial_protocol):
    """Resume a previously-run trailblaze execution.

    Parameters
    ----------
    checkpoint_dir_path : str
        The path to the directory used to store the trailblaze information.
    initial_protocol : Dict[str, List[float]]
        The initial protocol containing only the first state of the path.
        If no checkpoint file for the protocol is found, the file will
        be initialized using this state.

    Returns
    -------
    resumed_protocol : Dict[str, List[float]]
        The resumed optimal protocol.
    trajectory_file : _DCDTrajectoryFile
        A DCD file open for appening.
    sampler_state : SamplerState or None
        The last saved SamplerState or None if no frame was saved yet.

    """
    # We save the protocol in a YAML file and the positions in netcdf.
    protocol_file_path = os.path.join(checkpoint_dir_path, 'protocol.yaml')
    stds_file_path = os.path.join(checkpoint_dir_path, 'states_stds.json')
    positions_file_path = os.path.join(checkpoint_dir_path, 'coordinates.dcd')

    # Create the directory, if it doesn't exist.
    os.makedirs(checkpoint_dir_path, exist_ok=True)

    # Load protocol and stds checkpoint file.
    try:
        # Parse the previously calculated optimal_protocol dict.
        with open(protocol_file_path, 'r') as file_stream:
            resumed_protocol = yaml.load(file_stream, Loader=yaml.FullLoader)

        # Load the energy difference stds.
        with open(stds_file_path, 'r') as f:
            states_stds = json.load(f)
    except FileNotFoundError:
        resumed_protocol = initial_protocol
        states_stds = [[], []]

    # Check if there's an existing positions information.
    try:
        # We want the coordinates of the states that were sampled
        # during the search not the states after redistribution.
        sampler_states = read_trailblaze_checkpoint_coordinates(
            checkpoint_dir_path, redistributed=False)
    except FileNotFoundError:
        len_trajectory = 0
    else:
        len_trajectory = len(sampler_states)

    # Raise an error if the algorithm was interrupted *during*
    # writing on disk. We store only the states that we simulated
    # so there should be one less here.
    for state_values in resumed_protocol.values():
        if len_trajectory < len(state_values) - 1:
            err_msg = ("The trailblaze algorithm was interrupted while "
                       "writing the checkpoint file and it is now unable "
                       "to resume. Please delete the files "
                       f"in {checkpoint_dir_path} and restart.")
            raise RuntimeError(err_msg)

    # When this is resumed, but the trajectory is already completed,
    # the frame of the final end state has been already written, but
    # we don't want to add it twice at the end of the trailblaze function.
    len_trajectory = len(state_values) - 1

    # Whether the file exist or not, MDTraj doesn't support appending
    # files so we open a new one and rewrite the configurations we
    # have generated in the previous run.
    trajectory_file = _DCDTrajectoryFile(positions_file_path, 'w',
                                         force_overwrite=True)
    if len_trajectory > 0:
        for i in range(len_trajectory):
            trajectory_file.write_sampler_state(sampler_states[i])
        # Make sure the next search starts from the last saved position
        # unless the previous calculation was interrupted before the
        # first position could be saved.
        sampler_state = sampler_states[-1]
    else:
        sampler_state = None

    return resumed_protocol, states_stds, trajectory_file, sampler_state


def _cache_trailblaze_data(checkpoint_dir_path, optimal_protocol, states_stds,
                           trajectory_file, sampler_state):
    """Store on disk current state of the trailblaze run."""

    # Determine the file paths of the stored data.
    protocol_file_path = os.path.join(checkpoint_dir_path, 'protocol.yaml')
    stds_file_path = os.path.join(checkpoint_dir_path, 'states_stds.json')

    # Update protocol.
    with open(protocol_file_path, 'w') as file_stream:
        yaml.dump(optimal_protocol, file_stream)

    # Update the stds between states.
    with open(stds_file_path, 'w') as f:
        json.dump(states_stds, f)

    # Append positions of the state that we just simulated.
    trajectory_file.write_sampler_state(sampler_state)


def _redistribute_trailblaze_states(old_protocol, states_stds, thermodynamic_distance):
    """Redistribute the states using a bidirectional estimate of the thermodynamic length.

    Parameters
    ----------
    old_protocol : Dict[str, List[float]]
        The unidirectional optimal protocol.
    states_stds : List[List[float]]
        states_stds[j][i] is the standard deviation of the potential
        difference between states i-1 and i computed in the j direction.
    thermodynamic_distance : float
        The distance between each pair of states.

    Returns
    -------
    new_protocol : Dict[str, List[float]]
        The new estimate of the optimal protocol.
    states_map : List[int]
        states_map[i] is the index of the state in the old protocol that
        is closest to the i-th state in the new protocol. This allows to
        map coordinates generated during trailblazing to the redistributed
        protocol.
    """
    # The parameter names in a fixed order.
    parameter_names = [par_name for par_name in old_protocol]

    # Initialize the new protocol from the first state the optimal protocol.
    new_protocol = {par_name: [values[0]] for par_name, values in old_protocol.items()}

    # The first state of the new protocol always maps to the first state of the old one.
    states_map = [0]

    def _get_old_protocol_state(state_idx):
        """Return a representation of the thermo state as a list of parameter values."""
        return np.array([old_protocol[par_name][state_idx] for par_name in parameter_names])

    def _add_state_to_new_protocol(state):
        for parameter_name, new_state_value in zip(parameter_names, state.tolist()):
            new_protocol[parameter_name].append(new_state_value)

    # The thermodynamic length at 0 is 0.0.
    states_stds[0] = [0.0] + states_stds[0]
    states_stds[1] = [0.0] + states_stds[1]

    # We don't have the energy difference std in the
    # direction opposite to the search direction so we
    # pad the list.
    states_stds[1].append(states_stds[0][-1])
    states_stds = np.array(states_stds)

    # Compute a bidirectional estimate of the thermodynamic length.
    old_protocol_thermo_length = np.cumsum(np.mean(states_stds, axis=0))

    # Trailblaze again interpolating the thermodynamic length function.
    current_state_idx = 0
    current_state = _get_old_protocol_state(0)
    last_state = _get_old_protocol_state(-1)
    new_protocol_cum_thermo_length = 0.0
    while (current_state != last_state).any():
        # Find first state for which the accumulated standard
        # deviation is greater than the thermo length threshold.
        try:
            while old_protocol_thermo_length[current_state_idx+1] - new_protocol_cum_thermo_length <= thermodynamic_distance:
                current_state_idx += 1
        except IndexError:
            # If we got to the end, we just add the last state
            # to the protocol and stop the while loop.
            _add_state_to_new_protocol(last_state)
            break

        # Update current state.
        current_state = _get_old_protocol_state(current_state_idx)

        # The thermodynamic length from the last redistributed state to current state.
        pair_thermo_length = old_protocol_thermo_length[current_state_idx] - new_protocol_cum_thermo_length

        # Now interpolate between the current state state and the next to find
        # the exact state for which the thermo length equal the threshold.
        next_state = _get_old_protocol_state(current_state_idx+1)
        differential = thermodynamic_distance - pair_thermo_length
        differential /= old_protocol_thermo_length[current_state_idx+1] - old_protocol_thermo_length[current_state_idx]
        new_state = current_state + differential * (next_state - current_state)

        # Update cumulative thermo length.
        new_protocol_cum_thermo_length += thermodynamic_distance

        # Update states map.
        closest_state_idx = current_state_idx if differential <= 0.5 else current_state_idx+1
        states_map.append(closest_state_idx)

        # Update redistributed protocol.
        _add_state_to_new_protocol(new_state)

    # The last state of the new protocol always maps to the last state of the old one.
    states_map.append(len(old_protocol_thermo_length)-1)

    return new_protocol, states_map


def run_thermodynamic_trailblazing(
        thermodynamic_state, sampler_state, mcmc_move, state_parameters,
        parameter_setters=None, thermodynamic_distance=1.0,
        distance_tolerance=0.05, n_samples_per_state=100,
        reversed_direction=False, bidirectional_redistribution=True,
        bidirectional_search_thermo_dist='auto',
        global_parameter_functions=None, function_variables=tuple(),
        checkpoint_dir_path=None
):
    """
    Find an alchemical path by placing alchemical states at a fixed distance.

    The distance between two states is estimated by collecting ``n_samples_per_state``
    configurations through the MCMCMove in one of the two alchemical states,
    and computing the standard deviation of the difference of potential energies
    between the two states at those configurations.

    The states of the protocol are chosen so that each pair has a distance
    (in thermodynamic length) of ``thermodynamic_distance +- distance_tolerance``.
    The thermodynamic length estimate (in kT) is based on the standard deviation
    of the difference in potential energy between the two states.

    The function is capable of resuming when interrupted if ``checkpoint_dir_path``
    is specified. This will create two files called 'protocol.yaml' and
    'coordinates.dcd' storing the protocol and initial positions and box
    vectors for each state that are generated while running the algorithm.

    It is also possible to discretize a path specified through maathematical
    expressions through the arguments ``global_parameter_function`` and
    ``function_variables``.

    Parameters
    ----------
    thermodynamic_state : openmmtools.states.CompoundThermodynamicState
        The state of the alchemically modified system.
    sampler_state : openmmtools.states.SamplerState
        The sampler states including initial positions and box vectors.
    mcmc_move : openmmtools.mcmc.MCMCMove
        The MCMCMove to use for propagation.
    state_parameters : List[Tuple[str, List[float]]]
        Each element of this list is a tuple containing first the name
        of the parameter to be modified (e.g. ``lambda_electrostatics``,
        ``lambda_sterics``) and a list specifying the initial and final
        values for the path.
    parameter_setters : Dict[str, Callable], optional
        If the parameter cannot be set in the ``thermodynamic_state``
        with a simple call to ``setattr``, you can pass a dictionary
        mapping the parameter name to a function
        ``setter(thermodynamic_state, parameter_name, value)``. This
        is useful for example to set global parameter function variables
        with ``openmmtools.states.GlobalParameterState.set_function_variable``.
    thermodynamic_distance : float, optional
        The target distance (in thermodynamic length) between each pair of
        states in kT. Default is 1.0 (kT).
    distance_tolerance : float, optional
        The tolerance on the found standard deviation. Default is 0.05 (kT).
    n_samples_per_state : int, optional
        How many samples to collect to estimate the overlap between two
        states. Default is 100.
    reversed_direction : bool, optional
        If ``True``, the algorithm starts from the final state and traverses
        the path from the end to the beginning. The returned path
        discretization will still be ordered from the beginning to the
        end following the order in ``state_parameters``. Default is ``False``.
    bidirectional_redistribution : bool, optional
        If ``True``, the states will be redistributed using the standard
        deviation of the potential difference between states in both
        directions. Default is ``True``.
    bidirectional_search_thermo_dist : float or 'auto', optional
        If ``bidirectional_redistribution`` is ``True``, the thermodynamic
        distance between the sampled states used to collect data along
        the path can be different than the thermodynamic distance after
        redistribution. The default ('auto') caps the thermodynamic
        distance used for trailblazing at 1 kT. Keeping this value small
        lower the chance of obtaining very large stds in the opposite direction
        due to rare, dominating events in sections of the path where the overlap
        decreases quickly, which in turn may results in unreasonably long
        protocols.
    global_parameter_functions : Dict[str, Union[str, openmmtools.states.GlobalParameterFunction]], optional
        Map a parameter name to a mathematical expression as a string
        or a ``openmmtools.states.GlobalParameterFunction`` object.
    function_variables : List[str], optional
        A list of function variables entering the mathematical
        expressions.
    checkpoint_dir_path : str, optional
        The path to the directory used to store the trailblaze information.
        If this is given and the files exist, the algorithm will use this
        information to resume in case it was previously interrupted. If
        ``None``, no information is stored and it won't be possible to
        resume. Default is ``None``.

    Returns
    -------
    optimal_protocol : Dict[str, List[float]]
        The estimated protocol. Each dictionary key is one of the
        parameters in ``state_parameters``, and its values is the
        list of values that it takes in each state of the path.

    """
    def _state_parameter_setter(state, parameter_name, value):
        """Helper function to set state parameters."""
        setattr(state, parameter_name, value)

    def _function_variable_setter(state, parameter_name, value):
        """Helper function to set global parameter function variables."""
        state.set_function_variable(parameter_name, value)

    # Make sure that the state parameters to optimize have a clear order.
    assert (isinstance(state_parameters, list) or isinstance(state_parameters, tuple))

    # Determine the thermo distance to achieve during the search.
    if not bidirectional_redistribution:
        search_thermo_dist = thermodynamic_distance
    else:
        if bidirectional_search_thermo_dist == 'auto':
            search_thermo_dist = min(1.0, thermodynamic_distance)
        else:
            search_thermo_dist = bidirectional_search_thermo_dist

    # Create unordered helper variable.
    state_parameter_dict = {x[0]: x[1] for x in state_parameters}

    # Do not modify original thermodynamic_state.
    thermodynamic_state = copy.deepcopy(thermodynamic_state)

    # Handle mutable default arguments.
    if parameter_setters is None:
        parameter_setters = {}
    if global_parameter_functions is None:
        global_parameter_functions = {}

    # Make sure that the same parameter was not listed both as
    # a function and a parameter to iterate.
    for parameter_name, _ in state_parameters:
        if parameter_name in global_parameter_functions:
            raise ValueError(f"Cannot specify {parameter_name} in "
                              "'state_parameters' and 'global_parameter_functions'")

    # Make sure all function variables are in state_parameters.
    for function_variable in function_variables:
        if function_variable not in state_parameter_dict:
            raise ValueError(f"The variable '{function_variable}' must be given in 'state_parameters'")

    # Use special setters for the function variables.
    for function_variable in function_variables:
        if parameter_name not in parameter_setters:
            parameter_setters[function_variable] = _function_variable_setter
    # Create default setters for all other state parameters to later avoid if/else or try/except.
    for parameter_name, _ in state_parameters:
        if parameter_name not in parameter_setters:
            parameter_setters[parameter_name] = _state_parameter_setter

    # Initialize all function variables in the thermodynamic state.
    for function_variable in function_variables:
        value = state_parameter_dict[function_variable][0]
        thermodynamic_state.set_function_variable(function_variable, value)

    # Assign global parameter functions.
    for parameter_name, global_parameter_function in global_parameter_functions.items():
        # If the user doesn't pass an instance of function class, create one.
        if isinstance(global_parameter_function, str):
            global_parameter_function = mmtools.states.GlobalParameterFunction(global_parameter_function)
        setattr(thermodynamic_state, parameter_name, global_parameter_function)

    # Reverse the direction of the algorithm if requested.
    if reversed_direction:
        state_parameters = [(par_name, end_states.__class__(reversed(end_states)))
                            for par_name, end_states in reversed(state_parameters)]

    # Initialize protocol with the starting value.
    optimal_protocol = {par: [values[0]] for par, values in state_parameters}

    # Keep track of potential std between states in both directions
    # of the path so that we can redistribute the states later.
    # At the end of the protocol this will have the same length
    # of the protocol minus one. The inner lists are for the forward
    # and reversed direction stds respectively.
    states_stds = [[], []]

    # Check to see whether a trailblazing algorithm is already in progress,
    # and if so, restore to the previously checkpointed state.
    if checkpoint_dir_path is not None:
        optimal_protocol, states_stds, trajectory_file, resumed_sampler_state = _resume_thermodynamic_trailblazing(
            checkpoint_dir_path, optimal_protocol)
        # Start from the last saved conformation.
        if resumed_sampler_state is not None:
            sampler_state = resumed_sampler_state

    # We keep track of the previous state in the optimal protocol
    # that we'll use to compute the stds in the opposite direction.
    if len(states_stds[0]) == 0:
        previous_thermo_state = None
    else:
        previous_thermo_state = copy.deepcopy(thermodynamic_state)

    # Make sure that thermodynamic_state is in the last explored
    # state, whether the algorithm was resumed or not.
    for state_parameter in optimal_protocol:
        parameter_setters[state_parameter](thermodynamic_state, state_parameter,
                                           optimal_protocol[state_parameter][-1])
        if previous_thermo_state is not None:
            parameter_setters[state_parameter](previous_thermo_state, state_parameter,
                                               optimal_protocol[state_parameter][-2])

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
            # Simulate current thermodynamic state to collect samples.
            sampler_states = []
            simulated_energies = np.zeros(n_samples_per_state)
            for i in range(n_samples_per_state):
                mcmc_move.apply(thermodynamic_state, sampler_state)
                sampler_states.append(copy.deepcopy(sampler_state))

            # Keep track of the thermo state we use for the reweighting.
            reweighted_thermo_state = None

            # Find first state that doesn't overlap with simulated one
            # with std(du) within search_thermo_dist +- distance_tolerance.
            # We stop anyway if we reach the last value of the protocol.
            std_energy = 0.0
            current_parameter_value = optimal_protocol[state_parameter][-1]
            while (abs(std_energy - search_thermo_dist) > distance_tolerance and
                   not (current_parameter_value == values[1] and std_energy < search_thermo_dist)):

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
                    current_parameter_value += (search_thermo_dist - std_energy) / derivative_std_energy

                # Keep current_parameter_value inside bound interval.
                if search_direction * current_parameter_value > values[1]:
                    current_parameter_value = values[1]
                assert search_direction * (optimal_protocol[state_parameter][-1] - current_parameter_value) < 0

                # Determine the thermo states at which we need to compute the energies.
                # If this is the first attempt, compute also the reduced potential of
                # the simulated energies and the previous state to estimate the standard
                # deviation in the opposite direction.
                if reweighted_thermo_state is None:
                    # First attempt.
                    reweighted_thermo_state = copy.deepcopy(thermodynamic_state)
                    computed_thermo_states = [reweighted_thermo_state, thermodynamic_state]
                    if previous_thermo_state is not None:
                        computed_thermo_states.append(previous_thermo_state)
                else:
                    computed_thermo_states = [reweighted_thermo_state]

                # Set the reweighted state to the current parameter value.
                parameter_setters[state_parameter](reweighted_thermo_state, state_parameter, current_parameter_value)

                # Compute all energies.
                energies = np.empty(shape=(len(computed_thermo_states), n_samples_per_state))
                for i, sampler_state in enumerate(sampler_states):
                    energies[:,i] = mmtools.states.reduced_potential_at_states(
                        sampler_state, computed_thermo_states, mmtools.cache.global_context_cache)

                # Cache the simulated energies for the next iteration.
                if len(computed_thermo_states) > 1:
                    simulated_energies = energies[1]

                # Compute the energy difference std in the direction: simulated state -> previous state.
                if len(computed_thermo_states) > 2:
                    denergies = energies[2] - simulated_energies
                    states_stds[1].append(float(np.std(denergies, ddof=1)))

                # Compute the energy difference std between the currently simulated and the reweighted states.
                old_std_energy = std_energy
                denergies = energies[0] - simulated_energies
                std_energy = np.std(denergies, ddof=1)
                logger.debug('trailblazing: state_parameter {}, simulated_value {}, current_parameter_value {}, '
                             'std_du {}'.format(state_parameter, optimal_protocol[state_parameter][-1],
                                                current_parameter_value, std_energy))

            # Store energy difference std in the direction: simulated state -> reweighted state.
            states_stds[0].append(float(std_energy))

            # Update variables for next iteration.
            previous_thermo_state = copy.deepcopy(thermodynamic_state)
            thermodynamic_state = reweighted_thermo_state

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

            # Save the updated checkpoint file to disk.
            if checkpoint_dir_path is not None:
                _cache_trailblaze_data(checkpoint_dir_path, optimal_protocol, states_stds,
                                       trajectory_file, sampler_state)

    if checkpoint_dir_path is not None:
        # We haven't simulated the last state so we just set the positions of the second to last.
        trajectory_file.write_sampler_state(sampler_state)
        trajectory_file.close()

    # Redistribute the states using the standard deviation estimates in both directions.
    if bidirectional_redistribution:
        optimal_protocol, states_map = _redistribute_trailblaze_states(
            optimal_protocol, states_stds, thermodynamic_distance)

    # Save the states map for reading the coordinates correctly.
    if checkpoint_dir_path is not None:
        states_map_file_path = os.path.join(checkpoint_dir_path, 'states_map.json')

        if bidirectional_redistribution:
            with open(states_map_file_path, 'w') as f:
                json.dump(states_map, f)
        elif os.path.isfile(states_map_file_path):
            # If there's an old file because the path was previously redistributed,
            # we delete it so that read_trailblaze_checkpoint_coordinates will
            # return the coordinates associated to the most recently-generated path.
            os.remove(states_map_file_path)

    # If we have traversed the path in the reversed direction, re-invert
    # the order of the discretized path.
    if reversed_direction:
        for par_name, values in optimal_protocol.items():
            optimal_protocol[par_name] = values.__class__(reversed(values))

    # If we used global parameter functions, the optimal_protocol at this
    # point is a discretization of the function_variables, not the original
    # parameters so we convert it back.
    len_protocol = len(next(iter(optimal_protocol.values())))
    function_variables_protocol = {var: optimal_protocol.pop(var) for var in function_variables}
    original_parameters_protocol = {par: [] for par in global_parameter_functions}

    # Rebuild the optimal discretization for the original parameters.
    for state_idx in range(len_protocol):
        # Set the function variable value so that the function is computed.
        for function_variable in function_variables:
            value = function_variables_protocol[function_variable][state_idx]
            thermodynamic_state.set_function_variable(function_variable, value)

        # Recover original parameters.
        for parameter_name in original_parameters_protocol:
            value = getattr(thermodynamic_state, parameter_name)
            original_parameters_protocol[parameter_name].append(value)

    # Update the total protocol.
    optimal_protocol.update(original_parameters_protocol)

    logger.debug('Alchemical path found: {}'.format(optimal_protocol))
    return optimal_protocol


if __name__ == '__main__':
    import doctest
    doctest.testmod()
