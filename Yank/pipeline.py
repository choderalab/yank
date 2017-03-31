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

import mdtraj
import numpy as np
import openmmtools as mmtools
from simtk import openmm, unit

from . import utils
from .yank import AlchemicalPhase

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


# Common solvent and ions.
_SOLVENT_RESNAMES = frozenset(['118', '119', '1AL', '1CU', '2FK', '2HP', '2OF', '3CO', '3MT',
        '3NI', '3OF', '4MO', '543', '6MO', 'ACT', 'AG', 'AL', 'ALF', 'ATH',
        'AU', 'AU3', 'AUC', 'AZI', 'Ag', 'BA', 'BAR', 'BCT', 'BEF', 'BF4',
        'BO4', 'BR', 'BS3', 'BSY', 'Be', 'CA', 'CA+2', 'Ca+2', 'CAC', 'CAD',
        'CAL', 'CD', 'CD1', 'CD3', 'CD5', 'CE', 'CES', 'CHT', 'CL', 'CL-',
        'CLA', 'Cl-', 'CO', 'CO3', 'CO5', 'CON', 'CR', 'CS', 'CSB', 'CU',
        'CU1', 'CU3', 'CUA', 'CUZ', 'CYN', 'Cl-', 'Cr', 'DME', 'DMI', 'DSC',
        'DTI', 'DY', 'E4N', 'EDR', 'EMC', 'ER3', 'EU', 'EU3', 'F', 'FE', 'FE2',
        'FPO', 'GA', 'GD3', 'GEP', 'HAI', 'HG', 'HGC', 'HOH', 'IN', 'IOD',
        'ION', 'IR', 'IR3', 'IRI', 'IUM', 'K', 'K+', 'KO4', 'LA', 'LCO', 'LCP',
        'LI', 'LIT', 'LU', 'MAC', 'MG', 'MH2', 'MH3', 'MLI', 'MMC', 'MN',
        'MN3', 'MN5', 'MN6', 'MO1', 'MO2', 'MO3', 'MO4', 'MO5', 'MO6', 'MOO',
        'MOS', 'MOW', 'MW1', 'MW2', 'MW3', 'NA', 'NA+2', 'NA2', 'NA5', 'NA6',
        'NAO', 'NAW', 'Na+2', 'NET', 'NH4', 'NI', 'NI1', 'NI2', 'NI3', 'NO2',
        'NO3', 'NRU', 'Na+', 'O4M', 'OAA', 'OC1', 'OC2', 'OC3', 'OC4', 'OC5',
        'OC6', 'OC7', 'OC8', 'OCL', 'OCM', 'OCN', 'OCO', 'OF1', 'OF2', 'OF3',
        'OH', 'OS', 'OS4', 'OXL', 'PB', 'PBM', 'PD', 'PER', 'PI', 'PO3', 'PO4',
        'POT', 'PR', 'PT', 'PT4', 'PTN', 'RB', 'RH3', 'RHD', 'RU', 'RUB', 'Ra',
        'SB', 'SCN', 'SE4', 'SEK', 'SM', 'SMO', 'SO3', 'SO4', 'SOD', 'SR',
        'Sm', 'Sn', 'T1A', 'TB', 'TBA', 'TCN', 'TEA', 'THE', 'TL', 'TMA',
        'TRA', 'UNX', 'V', 'V2+', 'VN3', 'VO4', 'W', 'WO5', 'Y1', 'YB', 'YB2',
        'YH', 'YT3', 'ZN', 'ZN2', 'ZN3', 'ZNA', 'ZNO', 'ZO3'])

_NONPERIODIC_NONBONDED_METHODS = [openmm.app.NoCutoff, openmm.app.CutoffNonPeriodic]


def find_components(system, topology, ligand_dsl, solvent_dsl=None):
    """Determine the atom indices of the system components.

    Ligand atoms are specified through MDTraj domain-specific language (DSL),
    while receptor atoms are everything else excluding solvent.

    If ligand_net_charge is non-zero, the function also isolates the indices
    of ligand-neutralizing counterions in the 'ligand_counterions' component.

    Parameters
    ----------
    system : simtk.openmm.app.System
        The system object containing partial charges to perceive the net charge
        of the ligand.
    topology : mdtraj.Topology, simtk.openmm.app.Topology
        The topology object specifying the system. If simtk.openmm.app.Topology
        is passed instead this will be converted.
    ligand_dsl : str
        DSL specification of the ligand atoms.
    solvent_dsl : str, optional
        Optional DSL specification of the ligand atoms. If None, a list of
        common solvent residue names will be used to automatically detect
        solvent atoms (default is None).

    Returns
    -------
    atom_indices : dict of list of int
        atom_indices[component] is a list of atom indices belonging to that
        component, where component is one of 'receptor', 'ligand', 'solvent',
        'complex', and 'ligand_counterions'.

    """
    atom_indices = {}

    # Determine if we need to convert the topology to mdtraj
    # I'm using isinstance() to check this and not duck typing with hasattr() in
    # case openmm Topology will implement a select() method in the future
    if isinstance(topology, mdtraj.Topology):
        mdtraj_top = topology
    else:
        mdtraj_top = mdtraj.Topology.from_openmm(topology)

    # Determine ligand atoms
    atom_indices['ligand'] = mdtraj_top.select(ligand_dsl).tolist()

    # Determine solvent and receptor atoms
    # I did some benchmarking and this is still faster than a single for loop
    # through all the atoms in mdtraj_top.atoms
    if solvent_dsl is not None:
        solvent_indices = mdtraj_top.select(solvent_dsl).tolist()
        solvent_resnames = [mdtraj_top.atom(i).residue.name for i in solvent_indices]
        atom_indices['solvent'] = solvent_indices
    else:
        solvent_resnames = _SOLVENT_RESNAMES
        atom_indices['solvent'] = [atom.index for atom in mdtraj_top.atoms
                                   if atom.residue.name in solvent_resnames]
    not_receptor_set = frozenset(atom_indices['ligand'] + atom_indices['solvent'])
    atom_indices['receptor'] = [atom.index for atom in mdtraj_top.atoms
                                if atom.index not in not_receptor_set]
    atom_indices['complex'] = atom_indices['receptor'] + atom_indices['ligand']

    # Perceive ligand net charge
    ligand_net_charge = compute_net_charge(system, atom_indices['ligand'])
    logger.debug('Ligand net charge: {}'.format(ligand_net_charge))

    # Isolate ligand-neutralizing counterions.
    if ligand_net_charge != 0:
        if ligand_net_charge > 0:
            counterions_set = {name for name in solvent_resnames if '-' in name}
        elif ligand_net_charge < 0:
            counterions_set = {name for name in solvent_resnames if '+' in name}

        ligand_counterions = [atom.index for atom in mdtraj_top.atoms
                              if atom.residue.name in counterions_set]
        atom_indices['ligand_counterions'] = ligand_counterions[:abs(ligand_net_charge)]
        logger.debug('Found {} ligand counterions.'.format(len(atom_indices['ligand_counterions'])))

        # Eliminate ligand counterions indices from solvent component
        atom_indices['solvent'] = [i for i in atom_indices['solvent']
                                   if i not in atom_indices['ligand_counterions']]
    else:
        atom_indices['ligand_counterions'] = []

    return atom_indices


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


def prepare_phase(positions_file_path, parameters_file_path, ligand_dsl, system_options,
                  solvent_dsl=None, gromacs_include_dir=None, verbose=False):
    """Create a Yank arguments for a phase from system files.

    Parameters
    ----------
    positions_file_path : str
        Path to system position file (e.g. 'complex.inpcrd/.gro/.pdb').
    parameters_file_path : str
        Path to system parameters file (e.g. 'complex.prmtop/.top/.xml').
    ligand_dsl : str
        MDTraj DSL string that specify the ligand atoms.
    system_options : dict
        system_options[phase] is a a dictionary containing options to
        pass to createSystem(). If the parameters file is an OpenMM
        system in XML format, this will be ignored.
    solvent_dsl : str, optional
        Optional DSL specification of the ligand atoms. If None, a list of
        common solvent residue names will be used to automatically detect
        solvent atoms (default is None).
    gromacs_include_dir : str, optional
        Path to directory in which to look for other files included
        from the gromacs top file.
    verbose : bool
        Whether or not to log information (default is False).

    Returns
    -------
    alchemical_phase : AlchemicalPhase
        The alchemical phase for Yank calculation with unspecified name, and protocol.

    """
    # Load system files
    parameters_file_extension = os.path.splitext(parameters_file_path)[1]
    if parameters_file_extension == '.xml':
        # Read Amber prmtop and inpcrd files
        if verbose:
            logger.info("xml: %s" % parameters_file_path)
            logger.info("pdb: %s" % positions_file_path)
        with open(parameters_file_path, 'r') as f:
            serialized_system = f.read()
        system = openmm.XmlSerializer.deserialize(serialized_system)
        positions_file = openmm.app.PDBFile(positions_file_path)
        parameters_file = positions_file  # needed for topology
    elif parameters_file_extension == '.prmtop':
        # Read Amber prmtop and inpcrd files
        if verbose:
            logger.info("prmtop: %s" % parameters_file_path)
            logger.info("inpcrd: %s" % positions_file_path)
        parameters_file = openmm.app.AmberPrmtopFile(parameters_file_path)
        positions_file = openmm.app.AmberInpcrdFile(positions_file_path)
        box_vectors = positions_file.boxVectors
        create_system_args = set(inspect.getargspec(openmm.app.AmberPrmtopFile.createSystem).args)
        system = create_system(parameters_file, box_vectors, create_system_args, system_options)
    elif parameters_file_extension == '.top':
        # Read Gromacs top and gro files
        if verbose:
            logger.info("top: %s" % parameters_file_path)
            logger.info("gro: %s" % positions_file_path)

        positions_file = openmm.app.GromacsGroFile(positions_file_path)
        if ('nonbonded_method' in system_options) and (system_options['nonbonded_method'] in _NONPERIODIC_NONBONDED_METHODS):
            # gro files must contain box vectors, so we must determine whether system is non-periodic or not from provided nonbonded options
            # WARNING: This uses the private API for GromacsGroFile, and may break.
            logger.info('nonbonded_method = %s, so removing periodic box vectors from gro file' % system_options['nonbonded_method'])
            for (frame, box_vectors) in enumerate(positions_file._periodicBoxVectors):
                positions_file._periodicBoxVectors[frame] = None
        box_vectors = positions_file.getPeriodicBoxVectors()
        parameters_file = openmm.app.GromacsTopFile(parameters_file_path,
                                                    periodicBoxVectors=box_vectors,
                                                    includeDir=gromacs_include_dir)
        create_system_args = set(inspect.getargspec(openmm.app.GromacsTopFile.createSystem).args)
        system = create_system(parameters_file, box_vectors, create_system_args, system_options)
    else:
        raise ValueError('Unsupported format for parameter file {}'.format(parameters_file_extension))

    # Store numpy positions
    positions = positions_file.getPositions(asNumpy=True)

    # Check to make sure number of atoms match between prmtop and inpcrd.
    system_natoms = system.getNumParticles()
    positions_natoms = positions.shape[0]
    if system_natoms != positions_natoms:
        err_msg = "Atom number mismatch: {} has {} atoms; {} has {} atoms.".format(
            parameters_file_path, system_natoms, positions_file_path, positions_natoms)
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Find ligand atoms and receptor atoms
    atom_indices = find_components(system, parameters_file.topology, ligand_dsl,
                                   solvent_dsl=solvent_dsl)

    alchemical_phase = AlchemicalPhase('', system, parameters_file.topology,
                                       positions, atom_indices, None)
    return alchemical_phase


if __name__ == '__main__':
    import doctest
    doctest.testmod()
