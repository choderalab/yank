#!/usr/local/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Set up YANK calculations.

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os, os.path
import logging
logger = logging.getLogger(__name__)

import mdtraj

from simtk import unit
from simtk.openmm import app

from yank import utils
from yank.yank import Yank # TODO: Fix this weird import path to something more sane, like 'from yank import Yank'
from yank.repex import ThermodynamicState # TODO: Fix this weird import path to something more sane, like 'from yank.repex import ThermodynamicState'

#=============================================================================================
# SUBROUTINES
#=============================================================================================

#=============================================================================================
# COMMAND DISPATCH
#=============================================================================================

def dispatch(args):
    if args['binding']:
        return dispatch_binding(args)
    else:
        return False

#=============================================================================================
# SET UP BINDING FREE ENERGY CALCULATION
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

def find_components(topology, ligand_dsl, solvent_resnames=_SOLVENT_RESNAMES):
    """
    Return list of receptor and ligand atoms.
    Ligand atoms are specified through MDTraj domain-specific language (DSL),
    while receptor atoms are everything else excluding water.

    Parameters
    ----------
    topology : mdtraj.Topology, simtk.openmm.app.Topology
       The topology object specifying the system. If simtk.openmm.app.Topology
       is passed instead this will be converted.
    ligand_dsl : str
       DSL specification of the ligand atoms.
    solvent_resnames : list of str, optional, default=['WAT', 'TIP', 'HOH']
       List of solvent residue names to exclude from receptor definition.

    Returns
    -------
    atom_indices : dict of list of int
       atom_indices[component] is a list of atom indices belonging to that component, where
       component is one of 'receptor', 'ligand', 'solvent', 'complex'

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

    return atom_indices

def process_unit_bearing_argument(args, argname, compatible_units):
    """
    Process a unit-bearing command-line argument to produce a Quantity.

    Parameters
    ----------
    args : dict
       Command-line arguments dict from docopt.
    argname : str
       Key to use to extract value from args.
    compatible_units : simtk.unit.Unit
       The result will be checked for compatibility with specified units, and an exception raised if not compatible.

    Returns
    -------
    quantity : simtk.unit.Quantity
       The specified parameter, returned as a Quantity.

    """

    # WARNING: This is dangerous!
    # See: http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
    # TODO: Can we use a safer form of (or alternative to) 'eval' here?
    quantity = eval(args[argname], unit.__dict__)
    # Unpack quantity if it was surrounded by quotes.
    if isinstance(quantity, str):
        quantity = eval(quantity, unit.__dict__)
    # Check to make sure units are compatible with expected units.
    try:
        quantity.unit.is_compatible(compatible_units)
    except:
        raise Exception("Argument %s does not have units attached." % args[argname])
    # Check that units are compatible with what we expect.
    if not quantity.unit.is_compatible(compatible_units):
        raise Exception("Argument %s must be compatible with units %s" % (agname, str(compatible_units)))
    # Return unit-bearing quantity.
    return quantity

def setup_binding_amber(args):
    """
    Set up ligand binding free energy calculation using AMBER prmtop/inpcrd files.

    Parameters
    ----------
    args : dict
       Command-line arguments dict from docopt.

    Returns
    -------
    phases : list of str
       Phases (thermodynamic legs) of the calculation.
    systems : dict
       systems[phase] is the OpenMM System reference object for phase 'phase'.
    positions : dict
       positions[phase] is a set of positions (or list of positions) for initializing replicas.
    atom_indices : dict
       atom_indices[phase][component] is list of atom indices for component 'component' in phase 'phase'.

    """
    verbose = args['--verbose']

    # Implicit solvent
    if args['--gbsa']:
        implicitSolvent = getattr(app, args['--gbsa'])
    else:
        implicitSolvent = None

    # Select nonbonded treatment
    # TODO: Carefully check whether input file is periodic or not.
    if args['--nbmethod']:
        nonbondedMethod = getattr(app, args['--nbmethod'])
    else:
        nonbondedMethod = None

    # Constraints
    if args['--constraints']:
        constraints = getattr(app, args['--constraints'])
    else:
        constraints = None

    # COM removal
    removeCMMotion = False

    # Prepare phases of calculation.
    phase_prefixes = ['solvent', 'complex'] # list of calculation phases (thermodynamic legs) to set up
    components = ['ligand', 'receptor', 'solvent'] # components of the binding system
    systems = dict() # systems[phase] is the System object associated with phase 'phase'
    positions = dict() # positions[phase] is a list of coordinates associated with phase 'phase'
    atom_indices = dict() # ligand_atoms[phase] is a list of ligand atom indices associated with phase 'phase'
    setup_directory = args['--setupdir'] # Directory where prmtop/inpcrd files are to be found
    for phase_prefix in phase_prefixes:
        if verbose: logger.info("reading phase %s: " % phase_prefix)
        # Read Amber prmtop and create System object.
        prmtop_filename = os.path.join(setup_directory, '%s.prmtop' % phase_prefix)
        if verbose: logger.info("prmtop: %s" % prmtop_filename)
        prmtop = app.AmberPrmtopFile(prmtop_filename)
        # Read Amber inpcrd and load positions.
        inpcrd_filename = os.path.join(setup_directory, '%s.inpcrd' % phase_prefix)
        if verbose: logger.info("inpcrd: %s" % inpcrd_filename)
        inpcrd = app.AmberInpcrdFile(inpcrd_filename)
        # Determine if this will be an explicit or implicit solvent simulation.
        phase_suffix = 'implicit'
        is_periodic = False
        if inpcrd.boxVectors is not None:
            is_periodic = True
            phase_suffix = 'explicit'
        # Check if both periodic box and implicit solvent are defined
        if is_periodic and implicitSolvent is not None:
            logger.error('Detected both a periodic box and an implicit solvent.')
            raise Exception('Detected both a periodic box and an implicit solvent.')
        # Adjust nonbondedMethod.
        # TODO: Ensure that selected method is appropriate.
        if nonbondedMethod == None:
            if is_periodic:
                nonbondedMethod = app.CutoffPeriodic
            else:
                nonbondedMethod = app.NoCutoff
        # TODO: Check to make sure both prmtop and inpcrd agree on explicit/implicit.
        phase = '%s-%s' % (phase_prefix, phase_suffix)
        systems[phase] = prmtop.createSystem(nonbondedMethod=nonbondedMethod, implicitSolvent=implicitSolvent, constraints=constraints, removeCMMotion=removeCMMotion)
        positions[phase] = inpcrd.getPositions(asNumpy=True)
        # Update box vectors (if needed).
        if is_periodic:
            systems[phase].setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
        # Check to make sure number of atoms match between prmtop and inpcrd.
        prmtop_natoms = systems[phase].getNumParticles()
        inpcrd_natoms = positions[phase].shape[0]
        if prmtop_natoms != inpcrd_natoms:
            raise Exception("Atom number mismatch: prmtop %s has %d atoms; inpcrd %s has %d atoms." % (prmtop_filename, prmtop_natoms, inpcrd_filename, inpcrd_natoms))

        # Find ligand atoms and receptor atoms.
        ligand_dsl = args['--ligand'] # MDTraj DSL that specifies ligand atoms
        atom_indices[phase] = find_components(prmtop.topology, ligand_dsl)

    phases = systems.keys()

    return [phases, systems, positions, atom_indices]

def setup_systembuilder(args):
    """
    Set up ligand binding free energy calculation using OpenMM app and openmoltools.

    Parameters
    ----------
    args : dict
       Command-line arguments dict from docopt.

    Returns
    -------
    phases : list of str
       Phases (thermodynamic legs) of the calculation.
    systems : dict
       systems[phase] is the OpenMM System reference object for phase 'phase'.
    positions : dict
       positions[phase] is a set of positions (or list of positions) for initializing replicas.
    atom_indices : dict
       atom_indices[phase][component] is list of atom indices for component 'component' in phase 'phase'.

    """
    # TODO: Merge in systembuilder setup from PBG and JDC.
    raise Exception("Not implemented.")

def dispatch_binding(args):
    """
    Set up a binding free energy calculation.

    Parameters
    ----------
    args : dict
       Command-line arguments from docopt.

    """

    verbose = args['--verbose']
    store_dir = args['--store']
    utils.config_root_logger(verbose, log_file_path=os.path.join(store_dir, 'prepare.log'))

    #
    # Determine simulation options.
    #

    # Specify thermodynamic parameters.
    temperature = process_unit_bearing_argument(args, '--temperature', unit.kelvin)
    pressure = process_unit_bearing_argument(args, '--pressure', unit.atmospheres)
    thermodynamic_state = ThermodynamicState(temperature=temperature, pressure=pressure)

    # Create systems according to specified setup/import method.
    if args['amber']:
        [phases, systems, positions, atom_indices] = setup_binding_amber(args)
    elif args['systembuilder']:
        [phases, systems, positions, atom_indices] = setup_binding_systembuilder(args)
    else:
        logger.error("No valid binding free energy calculation setup command specified: Must be one of ['amber', 'systembuilder'].")
        # Trigger help argument to be returned.
        return False

    # Report some useful properties.
    if verbose:
        if 'complex-explicit' in atom_indices:
            phase = 'complex-explicit'
        else:
            phase = 'complex-implicit'
        logger.info("TOTAL ATOMS      : %9d" % len(atom_indices[phase]['complex']))
        logger.info("receptor         : %9d" % len(atom_indices[phase]['receptor']))
        logger.info("ligand           : %9d" % len(atom_indices[phase]['ligand']))
        if phase == 'complex-explicit':
            logger.info("solvent and ions : %9d" % len(atom_indices[phase]['solvent']))

    # Initialize YANK object.
    yank = Yank(store_dir)

    # Set options.
    options = dict()
    if args['--iterations']:
        options['number_of_iterations'] = int(args['--iterations'])
    if args['--equilibrate']:
        options['number_of_equilibration_iterations'] = int(args['--equilibrate'])
    if args['--online-analysis']:
        options['online_analysis'] = True
    if args['--restraints']:
        yank.restraint_type = args['--restraints']
    if args['--randomize-ligand']:
        options['randomize_ligand'] = True
    if args['--minimize']:
        options['minimize'] = True

    # Create new simulation.
    yank.create(phases, systems, positions, atom_indices, thermodynamic_state, options=options)

    # Report success.
    return True
