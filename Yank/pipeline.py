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
import logging
logger = logging.getLogger(__name__)

import mdtraj
from simtk import openmm

#=============================================================================================
# UTILITY FUNCTIONS
#=============================================================================================

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


def find_components(topology, ligand_dsl, ligand_net_charge=0,
                    solvent_resnames=_SOLVENT_RESNAMES):
    """Determine the atom indices of the system components.

    Ligand atoms are specified through MDTraj domain-specific language (DSL),
    while receptor atoms are everything else excluding solvent.

    If ligand_net_charge is non-zero, the function also isolates the indices
    of ligand-neutralizing counterions in the 'ligand_counterions' component.

    Parameters
    ----------
    topology : mdtraj.Topology, simtk.openmm.app.Topology
        The topology object specifying the system. If simtk.openmm.app.Topology
        is passed instead this will be converted.
    ligand_dsl : str
        DSL specification of the ligand atoms.
    ligand_net_charge : int, optional
        The net charge of the ligand (default is 0).
    solvent_resnames : list of str, optional
        List of solvent residue names to exclude from receptor definition (default
        is _SOLVENT_RESNAME).

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
    atom_indices['solvent'] = [atom.index for atom in mdtraj_top.atoms
                               if atom.residue.name in solvent_resnames]
    not_receptor_set = frozenset(atom_indices['ligand'] + atom_indices['solvent'])
    atom_indices['receptor'] = [atom.index for atom in mdtraj_top.atoms
                                if atom.index not in not_receptor_set]
    atom_indices['complex'] = atom_indices['receptor'] + atom_indices['ligand']

    # Isolate ligand-neutralizing counterions
    if ligand_net_charge != 0:
        if ligand_net_charge > 0:
            counterions_set = {name for name in solvent_resnames if '-' in name}
        elif ligand_net_charge < 0:
            counterions_set = {name for name in solvent_resnames if '+' in name}

        ligand_counterions = [atom.index for atom in mdtraj_top.atoms
                              if atom.residue.name in counterions_set]
        atom_indices['ligand_counterions'] = ligand_counterions[:abs(ligand_net_charge)]

        # Eliminate ligand counterions indices from solvent component
        atom_indices['solvent'] = [i for i in atom_indices['solvent']
                                   if i not in atom_indices['ligand_counterions']]
    else:
        atom_indices['ligand_counterions'] = []

    return atom_indices


def prepare_amber(system_dir, ligand_dsl, system_parameters, ligand_net_charge=0, verbose=False):
    """Create a system from prmtop and inpcrd files.

    Parameters
    ----------
    system_dir : str
        Path to the directory containing the prmtop and inpcrd files.
    ligand_dsl : str
        MDTraj DSL string that specify the ligand atoms.
    system_parameters : dict
        A kwargs dictionary to pass to openmm.app.AmberPrmtopFile.createSystem().
    ligand_net_charge : int, optional
       The net charge of the ligand. If non-zero, the function will look in the
       system for ligand-neutralizing counterions that will be included in atom_indices
       (default is 0).
    verbose : bool
        Whether or not to log informations (default is False).

    Returns
    -------
    phases : list of str
        Phases (thermodynamic legs) of the calculation.
    systems : dict
        systems[phase] is the OpenMM System reference object for phase 'phase'.
    positions : dict
        positions[phase] is a numpy array of positions for initializing replicas.
    atom_indices : dict
        atom_indices[phase][component] is list of atom indices for component 'component'
        in phase 'phase' as returned by find_components().

    """
    systems = {}  # systems[phase] is the System object associated with phase 'phase'
    positions = {}  # positions[phase] is a list of coordinates associated with phase 'phase'
    atom_indices = {}  # ligand_atoms[phase] is a list of ligand atom indices of phase 'phase'

    # Prepare phases of calculation.
    for phase_prefix in ['complex', 'solvent']:
        prmtop_file_path = os.path.join(system_dir, '{}.prmtop'.format(phase_prefix))
        inpcrd_file_path = os.path.join(system_dir, '{}.inpcrd'.format(phase_prefix))
        if verbose:
            logger.info("reading phase %s: " % phase_prefix)
            logger.info("prmtop: %s" % prmtop_file_path)
            logger.info("inpcrd: %s" % inpcrd_file_path)

        # Read Amber prmtop and inpcrd files
        prmtop = openmm.app.AmberPrmtopFile(prmtop_file_path)
        inpcrd = openmm.app.AmberInpcrdFile(inpcrd_file_path)

        # Determine if this will be an explicit or implicit solvent simulation.
        if inpcrd.boxVectors is not None:
            is_periodic = True
            phase_suffix = 'explicit'
        else:
            is_periodic = False
            phase_suffix = 'implicit'
        phase = '{}-{}'.format(phase_prefix, phase_suffix)

        # Adjust nonbondedMethod
        # TODO: Ensure that selected method is appropriate.
        if 'nonbondedMethod' not in system_parameters:
            if is_periodic:
                system_parameters['nonbondedMethod'] = openmm.app.CutoffPeriodic
            else:
                system_parameters['nonbondedMethod'] = openmm.app.NoCutoff

        # Check for solvent configuration inconsistencies
        # TODO: Check to make sure both prmtop and inpcrd agree on explicit/implicit.
        err_msg = ''
        if is_periodic:
            if 'implicitSolvent' in system_parameters:
                err_msg = 'Found periodic box in inpcrd file and implicitSolvent specified.'
            if system_parameters['nonbondedMethod'] == openmm.app.NoCutoff:
                err_msg = 'Found periodic box in inpcrd file but nonbondedMethod is NoCutoff'
        else:
            if system_parameters['nonbondedMethod'] != openmm.app.NoCutoff:
                err_msg = 'nonbondedMethod is NoCutoff but could not find periodic box in inpcrd.'
        if len(err_msg) != 0:
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        # Create system and update box vectors (if needed)
        systems[phase] = prmtop.createSystem(removeCMMotion=False, **system_parameters)
        if is_periodic:
            systems[phase].setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)

        # Store numpy positions
        positions[phase] = inpcrd.getPositions(asNumpy=True)

        # Check to make sure number of atoms match between prmtop and inpcrd.
        prmtop_natoms = systems[phase].getNumParticles()
        inpcrd_natoms = positions[phase].shape[0]
        if prmtop_natoms != inpcrd_natoms:
            err_msg = "Atom number mismatch: prmtop {} has {} atoms; inpcrd {} has {} atoms.".format(
                prmtop_file_path, prmtop_natoms, inpcrd_file_path, inpcrd_natoms)
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        # Find ligand atoms and receptor atoms
        atom_indices[phase] = find_components(prmtop.topology, ligand_dsl, ligand_net_charge)

    phases = systems.keys()

    return [phases, systems, positions, atom_indices]