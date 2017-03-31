#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Utility functions to help setting up Yank configurations.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import inspect
import logging
import itertools

import numpy as np
import openmmtools as mmtools
from simtk import openmm, unit

from . import utils

logger = logging.getLogger(__name__)


# ==============================================================================
# Utility functions
# ==============================================================================

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
        distances = np.sqrt(((x - np.tile(xref, (natoms, 1)))**2).sum(1)) #  distances[i] is the distance from the centroid to particle i

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
    """Return the atom indices of the ligand or solute counterions.

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
        which to find counterions.

    Returns
    -------
    counterions_indices : list of int
        The list of atom indices in the system of the counterions
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
    print(ions_net_charges)
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
        The kwargs accepted by the createSystem() function of the file.
    system_options : dict
        The kwargs to forward to createSystem().

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
        if 'implicitSolvent' in system_options:
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
        system_options[phase] is a a dictionary containing options to
        pass to createSystem(). If the parameters file is an OpenMM
        system in XML format, this will be ignored.
    gromacs_include_dir : str, optional
        Path to directory in which to look for other files included
        from the gromacs top file.
    verbose : bool
        Whether or not to log information (default is False).

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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
